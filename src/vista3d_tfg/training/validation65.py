"""Whole-volume validation on the original CT grid, with explicit empty classes."""
from __future__ import annotations

from pathlib import Path
import time

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

from vista3d_tfg.training import contract65 as c


def full_volume_prediction(model, image: np.ndarray, meta: dict, config: dict, device: str) -> np.ndarray:
    import torch
    from monai.inferers import sliding_window_inference

    inference=config["inference"]
    model.eval()
    def predictor(patches):
        with torch.autocast(device_type=device,dtype=torch.float16,enabled=device=="cuda"):
            logits=model(patches)
        if not bool(torch.isfinite(logits).all()):raise ValueError("Non-finite validation logits.")
        return logits.float()
    with torch.inference_mode():
        x=torch.from_numpy(np.array(image, dtype=np.float32, copy=True))[None,None]
        logits=sliding_window_inference(x,roi_size=inference["roi_size"],sw_batch_size=inference["sw_batch_size"],
                                       predictor=predictor,overlap=inference["overlap"],mode="gaussian",
                                       cval=-1.,sw_device=device,device="cpu",progress=False)
        if logits.shape!=(1,4,*image.shape):raise ValueError("Unexpected stitched prediction shape.")
        predicted=logits.argmax(dim=1)[0].numpy().astype(np.uint8)
    # The CT metadata supplies the target grid. No reference mask is an input.
    restored=resample_from_to(nib.Nifti1Image(predicted,np.asarray(meta["affine"])),
                             (tuple(meta["original_shape"]),np.asarray(meta["original_affine"])),order=0,cval=0)
    return np.asarray(restored.dataobj,dtype=np.uint8)


def voxel_metrics(gt: np.ndarray, pred: np.ndarray, affine) -> dict:
    if gt.shape!=pred.shape or not np.isin(gt,(0,1,2,3)).all() or not np.isin(pred,(0,1,2,3)).all():
        raise ValueError("Invalid reference/prediction labels or geometry.")
    voxel_ml=abs(float(np.linalg.det(np.asarray(affine)[:3,:3])))/1000.
    result={}
    for key,labels in (("kidney",(1,)),("tumor",(2,)),("cyst",(3,)),("renal_envelope",(1,2,3))):
        ref=np.isin(gt,labels);prediction=np.isin(pred,labels)
        nr,npred=int(ref.sum()),int(prediction.sum());intersection=int(np.count_nonzero(ref&prediction))
        result[key]={"reference_voxels":nr,"predicted_voxels":npred,"intersection_voxels":intersection,
                     "dice":2.*intersection/(nr+npred) if nr+npred else 1.,
                     "false_positive_ml":(npred-intersection)*voxel_ml,
                     "false_negative_ml":(nr-intersection)*voxel_ml}
    return result


def summarize(reports: list[dict], ids: list[str]) -> dict:
    if [r["case_id"] for r in reports]!=ids:raise ValueError("Selection requires every frozen validation case.")
    classes={}
    for label in ("kidney","tumor","cyst","renal_envelope"):
        entries=[r["metrics"][label] for r in reports]
        positive=[r["dice"] for r in entries if r["reference_voxels"]>0]
        negative=[r for r in entries if r["reference_voxels"]==0]
        classes[label]={"mean_dice":float(np.mean([r["dice"] for r in entries])),
                        "reference_positive_n":len(positive),
                        "reference_positive_mean_dice":float(np.mean(positive)) if positive else None,
                        "reference_negative_n":len(negative),
                        "reference_negative_with_prediction_n":sum(r["predicted_voxels"]>0 for r in negative),
                        "reference_negative_false_positive_ml":sum(r["false_positive_ml"] for r in negative)}
    return {"n_cases":len(ids),"classes":classes,
            "selection_score":sum(classes[k]["mean_dice"] for k in ("kidney","tumor","cyst"))/3.}


def validate_epoch(root: Path, plan: dict, index, model, epoch: int, checkpoint_sha: str, device: str) -> dict:
    output=c.output_root(root,plan)/"validation"/f"epoch_{epoch:03d}"
    reports=[];ids=c.partition_ids(plan,"validation")
    for number,cid in enumerate(ids):
        c.verify_input(root,plan,"validation",cid)
        folder=output/cid
        identity={"experiment_sha256":plan["experiment_sha256"],"epoch":epoch,"case_id":cid,
                  "checkpoint_sha256":checkpoint_sha,"input_identity":plan["inputs"][cid],
                  "cache_receipt_sha256":index.receipts[cid]}
        if (folder/"complete.json").exists():
            receipt=c.read_json(folder/"complete.json");c.check_digest(receipt,"receipt_sha256")
            if receipt["identity"]!=identity or set(receipt["artifacts"])!={"prediction.nii.gz","metrics.json"}:
                raise ValueError("Validation receipt identity changed.")
            for name,digest in receipt["artifacts"].items():
                if c.sha256_file(folder/name)!=digest:raise ValueError("Validation artifact changed.")
            report=c.read_json(folder/"metrics.json")
            if report["identity"]!=identity or report["case_id"]!=cid:raise ValueError("Validation metrics identity changed.")
        else:
            print(f"Validation epoch {epoch:03d} [{number+1:03d}/{len(ids):03d}] {cid}",flush=True)
            started=time.perf_counter()
            image,meta=index.load("validation",cid)
            pred=full_volume_prediction(model,image,meta,plan["settings"],device)
            _,gt_path=c.case_paths(root,plan,"validation",cid)
            reference=nib.load(gt_path)
            if list(reference.shape)!=meta["original_shape"] or not np.allclose(reference.affine,meta["original_affine"],rtol=0,atol=1e-5):
                raise ValueError("Original-grid validation geometry changed.")
            metrics=voxel_metrics(np.asarray(reference.dataobj),pred,meta["original_affine"])
            folder.mkdir(parents=True,exist_ok=True)
            nib.save(nib.Nifti1Image(pred,np.asarray(meta["original_affine"])),folder/"prediction.nii.gz")
            report={"case_id":cid,"identity":identity,"metrics":metrics,"seconds":time.perf_counter()-started}
            c.atomic_json(folder/"metrics.json",report)
            c.verify_input(root,plan,"validation",cid)
            receipt={"identity":identity,"artifacts":{n:c.sha256_file(folder/n) for n in ("prediction.nii.gz","metrics.json")}}
            receipt["receipt_sha256"]=c.canonical_sha256(receipt);c.atomic_json(folder/"complete.json",receipt)
        reports.append(report)
        c.atomic_json(output/"progress.json",{"completed_case_ids":[r["case_id"] for r in reports],"requested_case_ids":ids})
    result={"epoch":epoch,"checkpoint_sha256":checkpoint_sha,"experiment_sha256":plan["experiment_sha256"],
            "status":"complete",**summarize(reports,ids)}
    result["summary_sha256"]=c.canonical_sha256(result)
    c.frozen_json(output/"summary.json",result)
    return result
