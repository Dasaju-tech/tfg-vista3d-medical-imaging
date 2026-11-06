"""Versioned continuation; parent tensors and cache are read-only inputs."""
from __future__ import annotations

import math
from pathlib import Path
import time
from uuid import uuid4

import numpy as np

from vista3d_tfg.training import contract65 as c, contract65b as b
from vista3d_tfg.training.cache65 import CacheIndex
from vista3d_tfg.training.supervised64 import build_model, state_digest
from vista3d_tfg.training.validation65 import validate_epoch
from vista3d_tfg.training.engine65 import augmented_patch, commit_checkpoint, epoch_receipt
from vista3d_tfg.training.amp65b import optimizer_step


def train(root: Path, plan: dict, runtime: dict, *, epochs: int, device: str="cuda") -> dict:
    import torch
    from monai.losses import DiceCELoss

    if type(epochs) is not int or epochs<0:raise ValueError("Epoch budget must be nonnegative.")
    config=plan["settings"];output=c.output_root(root,plan)
    parent=c.read_json(root/c.CONFIG)
    source=c.output_root(root,parent)
    parent_records=b.history(root,parent)
    if parent_records[-1]["epoch"] != plan["parent_last_epoch"]:
        raise ValueError("Parent checkpoint boundary changed.")
    index=CacheIndex(root,parent)
    cache_identity=c.canonical_sha256(index.receipts)
    expected_parent={"runtime":runtime,"device":device,"cache_sha256":cache_identity,
                     "experiment_sha256":parent["experiment_sha256"]}
    if c.read_json(source/"training_identity.json") != expected_parent:
        raise ValueError("Parent runtime/cache identity changed.")
    c.frozen_json(output/"training_identity.json",{"runtime":runtime,"device":device,"cache_sha256":cache_identity,
                                                 "experiment_sha256":plan["experiment_sha256"]})
    torch.manual_seed(config["seed"])
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True,warn_only=True)
    if device=="cuda":torch.cuda.manual_seed_all(config["seed"])
    model=build_model(config).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config["optimizer"]["lr"],weight_decay=config["optimizer"]["weight_decay"])
    scaler=torch.amp.GradScaler("cuda",enabled=device=="cuda",init_scale=128.)
    loss_fn=DiceCELoss(include_background=False,to_onehot_y=True,softmax=True,lambda_dice=1.,lambda_ce=1.)
    completed=[]
    for epoch in range(plan["parent_last_epoch"]+1,config["epochs"]+1):
        receipt=epoch_receipt(output,plan,epoch)
        if receipt is None:
            if any((output/"epochs"/f"epoch_{later:03d}"/"complete.json").exists() for later in range(epoch+1,config["epochs"]+1)):
                raise ValueError("Gap in committed epoch sequence; restore missing evidence before resuming.")
            break
        if (receipt["stats"].get("recovery_revision")!=b.REVISION
                or receipt["stats"].get("parent_experiment_sha256")!=parent["experiment_sha256"]):
            raise ValueError("Continuation checkpoint lineage changed.")
        completed.append(receipt)
    latest=completed[-1] if completed else parent_records[-1]
    last_epoch=latest["epoch"]
    checkpoint_output=output if completed else source
    checkpoint_plan=plan if completed else parent
    checkpoint=torch.load(checkpoint_output/"epochs"/f"epoch_{last_epoch:03d}"/latest["file"],map_location="cpu",weights_only=True)
    if (checkpoint["runtime"]!=runtime or checkpoint["experiment_sha256"]!=checkpoint_plan["experiment_sha256"]
            or checkpoint["epoch"]!=last_epoch or checkpoint["stats"]!=latest["stats"]):
        raise ValueError("Resume checkpoint identity mismatch.")
    model.load_state_dict(checkpoint["model"],strict=True);optimizer.load_state_dict(checkpoint["optimizer"]);scaler.load_state_dict(checkpoint["scaler"])
    if state_digest(model)!=checkpoint["model_state_sha256"]:raise ValueError("Resume model state mismatch.")
    torch.set_rng_state(checkpoint["torch_rng"])
    if device=="cuda":torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    del checkpoint
    selected=b.summaries(root,parent,parent_records)
    # A committed epoch can lack validation after an interruption. Finish that
    # validation before advancing; complete case receipts are reused.
    for entry in completed:
        ep=entry["epoch"]
        if ep in config["validation_epochs"]:
            if ep!=last_epoch:
                summary=c.read_json(output/"validation"/f"epoch_{ep:03d}"/"summary.json")
                c.check_digest(summary,"summary_sha256")
                if (summary["checkpoint_sha256"]!=entry["sha256"] or summary["n_cases"]!=len(c.partition_ids(plan,"validation"))
                        or summary["experiment_sha256"]!=plan["experiment_sha256"] or summary["epoch"]!=ep or summary["status"]!="complete"):
                    raise ValueError("Historical validation summary changed.")
            else:summary=validate_epoch(root,plan,index,model,ep,entry["sha256"],device)
            selected.append(summary)
    target=min(config["epochs"],last_epoch+epochs)
    ids=c.partition_ids(plan,"development")
    for epoch in range(last_epoch+1,target+1):
        started=time.perf_counter();model.train()
        if device=="cuda":torch.cuda.reset_peak_memory_stats()
        rng=np.random.default_rng(config["seed"]+epoch)
        ordered=[ids[int(i)] for i in rng.permutation(len(ids))]
        lr=config["optimizer"]["lr"]*.5*(1.+math.cos(math.pi*(epoch-1)/config["epochs"]))
        for group in optimizer.param_groups:group["lr"]=lr
        amp_retries=0;scale_start=float(scaler.get_scale())
        losses=[];class_centers={str(k):0 for k in range(4)};case_losses=[]
        for number,cid in enumerate(ordered):
            image,label,counts,_=index.load("development",cid)
            case_values=[]
            for patch_index in range(config["patches_per_case"]):
                x_np,y_np,k=augmented_patch(image,label,counts,config,rng)
                x=torch.from_numpy(x_np)[None,None].to(device);y=torch.from_numpy(y_np)[None,None].to(device)
                def record_retry(event):
                    event={**event,"epoch":epoch,"case_id":cid,"patch_index":patch_index,
                           "experiment_sha256":plan["experiment_sha256"]}
                    c.atomic_json(output/"amp_events"/("event_"+uuid4().hex+".json"),event)
                    print(f"  AMP retry {event['retry']}: {cid}; scale {event['scale_before']:g} -> {event['scale_after']:g}; same patch",flush=True)
                try:
                    value,retries=optimizer_step(model,optimizer,scaler,loss_fn,x,y,
                                                 config["max_gradient_norm"],device,on_retry=record_retry)
                except (ValueError,RuntimeError) as exc:
                    raise RuntimeError(f"Epoch {epoch}, case {cid}, patch {patch_index+1}: {exc}") from exc
                amp_retries+=retries
                losses.append(value);case_values.append(value);class_centers[str(k)]+=1
                del x,y
            case_losses.append({"case_id":cid,"mean_loss":float(np.mean(case_values))})
            print(f"Epoch {epoch:03d}/{config['epochs']:03d} [{number+1:03d}/{len(ids):03d}] {cid}; loss={np.mean(case_values):.6f}",flush=True)
            c.atomic_json(output/"training_progress.json",{"status":"running","epoch_in_progress":epoch,"last_committed_epoch":epoch-1,
                                                         "processed_cases_this_epoch":number+1,"test_reads":False})
        if device=="cuda":torch.cuda.synchronize()
        stats={"epoch":epoch,"optimizer_steps":epoch*len(ids)*config["patches_per_case"],"steps_this_epoch":len(losses),
               "amp_retries":amp_retries,"amp_scale_start":scale_start,"amp_scale_end":float(scaler.get_scale()),
               "effective_optimizer_steps_this_epoch":len(losses),
               "parent_experiment_sha256":parent["experiment_sha256"],"recovery_revision":b.REVISION,
               "mean_loss":float(np.mean(losses)),"lr":lr,"case_order":ordered,"case_losses":case_losses,
               "class_centers":class_centers,"wall_seconds":time.perf_counter()-started,
               "peak_allocated_bytes":torch.cuda.max_memory_allocated() if device=="cuda" else None,
               "peak_reserved_bytes":torch.cuda.max_memory_reserved() if device=="cuda" else None}
        latest=commit_checkpoint(output,plan,runtime,model,optimizer,scaler,epoch,stats,device)
        if epoch in config["validation_epochs"]:
            selected.append(validate_epoch(root,plan,index,model,epoch,latest["sha256"],device))
        last_epoch=epoch
    best=max(selected,key=lambda r:(r["selection_score"],-r["epoch"])) if selected else None
    selection={"experiment_sha256":plan["experiment_sha256"],"final":last_epoch==config["epochs"],
               "best_epoch":best["epoch"] if best else None,"best_score":best["selection_score"] if best else None,
               "evaluated_epochs":[r["epoch"] for r in selected],"test_reads":False,
               "best_checkpoint_experiment_sha256":best["experiment_sha256"] if best else None,
               "best_checkpoint_sha256":best["checkpoint_sha256"] if best else None}
    c.atomic_json(output/"selection.json",selection)
    result={"status":"complete" if last_epoch==config["epochs"] else "partial","completed_epochs":last_epoch,
            "planned_epochs":config["epochs"],"optimizer_steps":last_epoch*len(ids)*config["patches_per_case"],
            "selection":selection,"test_reads":False,"device":device}
    c.atomic_json(output/"training_progress.json",result)
    return result


def train_until(root, plan, runtime, *, until_epoch, device="cuda"):
    """An absolute target cannot accidentally add epochs when rerun."""
    if type(until_epoch) is not int or not 1 <= until_epoch <= plan["settings"]["epochs"]:
        raise ValueError("Target epoch is outside the frozen schedule.")
    last = plan["parent_last_epoch"]
    output = c.output_root(root, plan)
    for epoch in range(last+1, plan["settings"]["epochs"]+1):
        if epoch_receipt(output, plan, epoch) is None:
            break
        last = epoch
    return train(root, plan, runtime, epochs=max(0, until_epoch-last), device=device)
