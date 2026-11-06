"""Resumable epoch training and fixed validation-based checkpoint selection."""
from __future__ import annotations

import math
from pathlib import Path
import time
from uuid import uuid4

import numpy as np

from vista3d_tfg.training import contract65 as c
from vista3d_tfg.training.cache65 import CacheIndex
from vista3d_tfg.training.supervised64 import build_model, state_digest
from vista3d_tfg.training.validation65 import validate_epoch


def sample_patch(image, label, counts, size, rng):
    present=np.flatnonzero(counts.sum(axis=1))
    k=int(rng.choice(present))
    cumulative=counts[k].cumsum();rank=int(rng.integers(int(cumulative[-1])))
    z=int(np.searchsorted(cumulative,rank,side="right"))
    offset=rank-(int(cumulative[z-1]) if z else 0)
    flat=np.flatnonzero(label[:,:,z]==k)[offset]
    center=(*np.unravel_index(flat,label.shape[:2]),z)
    src,dst=[],[]
    for value,n,dim in zip(center,size,label.shape):
        start=int(value)-n//2;lower,upper=max(start,0),min(start+n,dim)
        src.append(slice(lower,upper));dst.append(slice(lower-start,upper-start))
    x=np.full(size,-1.,dtype=np.float32);y=np.zeros(size,dtype=np.int64)
    x[tuple(dst)]=image[tuple(src)];y[tuple(dst)]=label[tuple(src)]
    return x,y,k


def augmented_patch(image,label,counts,config,rng):
    x,y,k=sample_patch(image,label,counts,config["patch_size"],rng)
    aug=config["augmentation"]
    for axis in range(3):
        if rng.random()<aug["flip_probability_per_axis"]:x=np.flip(x,axis);y=np.flip(y,axis)
    x=x*rng.uniform(*aug["intensity_scale"])+rng.uniform(*aug["intensity_shift"])
    return np.ascontiguousarray(np.clip(x,-1.,1.),dtype=np.float32),np.ascontiguousarray(y),k


def commit_checkpoint(output: Path, plan: dict, runtime: dict, model, optimizer, scaler, epoch: int, stats: dict, device: str):
    import torch
    payload={"experiment_sha256":plan["experiment_sha256"],"runtime":runtime,"epoch":epoch,
             "model":model.state_dict(),"optimizer":optimizer.state_dict(),"scaler":scaler.state_dict(),
             "torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all() if device=="cuda" else [],
             "stats":stats,"model_state_sha256":state_digest(model)}
    folder=output/"epochs"/f"epoch_{epoch:03d}";folder.mkdir(parents=True,exist_ok=True)
    if (folder/"complete.json").exists():raise ValueError("Refusing to replace a completed epoch.")
    path=folder/("checkpoint_"+uuid4().hex[:12]+".pt")
    torch.save(payload,path)
    loaded=torch.load(path,map_location="cpu",weights_only=True)
    if loaded["model_state_sha256"]!=payload["model_state_sha256"] or any(
            not torch.equal(t.detach().cpu(),loaded["model"][name]) for name,t in model.state_dict().items()):
        raise ValueError("Checkpoint tensor round-trip failed.")
    receipt={"experiment_sha256":plan["experiment_sha256"],"epoch":epoch,
             "file":path.name,"sha256":c.sha256_file(path),"stats":stats}
    receipt["receipt_sha256"]=c.canonical_sha256(receipt)
    c.atomic_json(folder/"complete.json",receipt)
    return receipt


def epoch_receipt(output: Path, plan: dict, epoch: int) -> dict | None:
    import re
    folder=output/"epochs"/f"epoch_{epoch:03d}"
    if not (folder/"complete.json").exists():return None
    r=c.read_json(folder/"complete.json");c.check_digest(r,"receipt_sha256")
    if (r["experiment_sha256"]!=plan["experiment_sha256"] or r["epoch"]!=epoch
            or not re.fullmatch(r"checkpoint_[0-9a-f]{12}\.pt",r["file"]) or c.sha256_file(folder/r["file"])!=r["sha256"]):
        raise ValueError("Epoch checkpoint changed.")
    return r


def train(root: Path, plan: dict, runtime: dict, *, epochs: int, device: str="cuda") -> dict:
    import torch
    from monai.losses import DiceCELoss

    if type(epochs) is not int or epochs<1:raise ValueError("Epoch budget must be positive.")
    config=plan["settings"];output=c.output_root(root,plan)
    index=CacheIndex(root,plan)
    cache_identity=c.canonical_sha256(index.receipts)
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
    for epoch in range(config["epochs"]+1):
        receipt=epoch_receipt(output,plan,epoch)
        if receipt is None:
            if any((output/"epochs"/f"epoch_{later:03d}"/"complete.json").exists() for later in range(epoch+1,config["epochs"]+1)):
                raise ValueError("Gap in committed epoch sequence; restore missing evidence before resuming.")
            break
        completed.append(receipt)
    if not completed:
        completed=[commit_checkpoint(output,plan,runtime,model,optimizer,scaler,0,{"optimizer_steps":0},device)]
    latest=completed[-1];last_epoch=latest["epoch"]
    checkpoint=torch.load(output/"epochs"/f"epoch_{last_epoch:03d}"/latest["file"],map_location="cpu",weights_only=True)
    if checkpoint["runtime"]!=runtime or checkpoint["experiment_sha256"]!=plan["experiment_sha256"] or checkpoint["epoch"]!=last_epoch:
        raise ValueError("Resume checkpoint identity mismatch.")
    model.load_state_dict(checkpoint["model"],strict=True);optimizer.load_state_dict(checkpoint["optimizer"]);scaler.load_state_dict(checkpoint["scaler"])
    if state_digest(model)!=checkpoint["model_state_sha256"]:raise ValueError("Resume model state mismatch.")
    torch.set_rng_state(checkpoint["torch_rng"])
    if device=="cuda":torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    del checkpoint
    selected=[]
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
        losses=[];class_centers={str(k):0 for k in range(4)};case_losses=[]
        for number,cid in enumerate(ordered):
            image,label,counts,_=index.load("development",cid)
            case_values=[]
            for _ in range(config["patches_per_case"]):
                x_np,y_np,k=augmented_patch(image,label,counts,config,rng)
                x=torch.from_numpy(x_np)[None,None].to(device);y=torch.from_numpy(y_np)[None,None].to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device,dtype=torch.float16,enabled=device=="cuda"):logits=model(x)
                loss=loss_fn(logits.float(),y)
                if not bool(torch.isfinite(loss)):raise ValueError("Non-finite training loss.")
                scaler.scale(loss).backward();scaler.unscale_(optimizer)
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config["max_gradient_norm"],error_if_nonfinite=True)
                if float(norm)<=0:raise ValueError("Zero training gradient.")
                scaler.step(optimizer);scaler.update()
                value=float(loss.detach());losses.append(value);case_values.append(value);class_centers[str(k)]+=1
                del x,y,logits,loss
            case_losses.append({"case_id":cid,"mean_loss":float(np.mean(case_values))})
            print(f"Epoch {epoch:03d}/{config['epochs']:03d} [{number+1:03d}/{len(ids):03d}] {cid}; loss={np.mean(case_values):.6f}",flush=True)
            c.atomic_json(output/"training_progress.json",{"status":"running","epoch_in_progress":epoch,"last_committed_epoch":epoch-1,
                                                         "processed_cases_this_epoch":number+1,"test_reads":False})
        if device=="cuda":torch.cuda.synchronize()
        stats={"epoch":epoch,"optimizer_steps":epoch*len(ids)*config["patches_per_case"],"steps_this_epoch":len(losses),
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
               "evaluated_epochs":[r["epoch"] for r in selected],"test_reads":False}
    c.atomic_json(output/"selection.json",selection)
    result={"status":"complete" if last_epoch==config["epochs"] else "partial","completed_epochs":last_epoch,
            "planned_epochs":config["epochs"],"optimizer_steps":last_epoch*len(ids)*config["patches_per_case"],
            "selection":selection,"test_reads":False,"device":device}
    c.atomic_json(output/"training_progress.json",result)
    return result
