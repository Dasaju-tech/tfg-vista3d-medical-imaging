"""A separately frozen continuation of an unchanged Block 65 parent."""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path

from vista3d_tfg.training import contract65 as c
from vista3d_tfg.training.engine65 import epoch_receipt

CONFIG = "configs/experiments/kits23_supervised_block65b.json"
WORKER = "scripts/kits23_supervised65b_worker.py"
SOURCES = (WORKER, "scripts/run_kits23_supervised65b.py",
           "src/vista3d_tfg/training/contract65b.py", "src/vista3d_tfg/training/amp65b.py",
           "src/vista3d_tfg/training/engine65b.py", "src/vista3d_tfg/experiments/supervised65b.py")
REVISION = "65b-amp-same-patch-retry-v1"
POLICY = {"max_retries": 8, "retry": "same augmented tensors; dynamic GradScaler backoff",
          "nonfinite_loss": "fail", "persistent_nonfinite_gradients": "fail",
          "count": "successful optimizer updates only", "fp32_fallback": False}


def code_identity(root):
    return {**c.code_identity(root), **{name: c.sha256_file(root/name) for name in SOURCES}}


def parent_plan(root, protocol):
    parent = c.read_json(root/c.CONFIG)
    c.check_plan(root, parent, protocol)
    if c.read_json(c.output_root(root, parent)/"protocol.json") != parent:
        raise ValueError("Parent protocol copy changed.")
    return parent


def history(root, parent):
    output = c.output_root(root, parent)
    records = []
    for epoch in range(parent["settings"]["epochs"] + 1):
        receipt = epoch_receipt(output, parent, epoch)
        if receipt is None:
            if any((output/"epochs"/f"epoch_{later:03d}"/"complete.json").exists()
                   for later in range(epoch + 1, parent["settings"]["epochs"] + 1)):
                raise ValueError("Gap in parent checkpoint history.")
            break
        records.append(receipt)
    if len(records) < 2 or records[-1]["epoch"] >= parent["settings"]["epochs"]:
        raise ValueError("Continuation requires completed training epochs and remaining work.")
    return records


def summaries(root, parent, records):
    result = []
    for record in records:
        epoch = record["epoch"]
        if epoch not in parent["settings"]["validation_epochs"]:
            continue
        path = c.output_root(root, parent)/"validation"/f"epoch_{epoch:03d}"/"summary.json"
        s = c.read_json(path); c.check_digest(s, "summary_sha256")
        if (s["epoch"] != epoch or s["experiment_sha256"] != parent["experiment_sha256"]
                or s["checkpoint_sha256"] != record["sha256"] or s["status"] != "complete"
                or s["n_cases"] != len(c.partition_ids(parent, "validation"))
                or not math.isfinite(s["selection_score"]) or not 0 <= s["selection_score"] <= 1):
            raise ValueError("Parent validation summary changed or incomplete.")
        result.append(s)
    return result


def freeze(root: Path, protocol: dict):
    if (root/CONFIG).exists():
        plan = c.read_json(root/CONFIG); check_plan(root, plan, protocol)
        return plan
    parent = parent_plan(root, protocol)
    records = history(root, parent)
    previous = summaries(root, parent, records)
    output = c.output_root(root, parent)
    artifacts = ["protocol.json", "training_identity.json"]
    for r in records:
        folder = f"epochs/epoch_{r['epoch']:03d}/"
        artifacts += [folder+"complete.json", folder+r["file"]]
    artifacts += [f"validation/epoch_{s['epoch']:03d}/summary.json" for s in previous]
    plan = deepcopy(parent)
    plan.pop("experiment_sha256")
    plan.update(revision=REVISION, amp_recovery=deepcopy(POLICY), implementation=code_identity(root),
                parent_experiment_sha256=parent["experiment_sha256"],
                parent_last_epoch=records[-1]["epoch"],
                parent_artifacts={name: c.sha256_file(output/name) for name in artifacts})
    plan["experiment_sha256"] = c.canonical_sha256(plan)
    check_plan(root, plan, protocol)
    c.frozen_json(root/CONFIG, plan)
    c.frozen_json(c.output_root(root, plan)/"protocol.json", plan)
    return plan


def check_plan(root: Path, plan: dict, protocol: dict):
    c.check_digest(plan, "experiment_sha256")
    parent = parent_plan(root, protocol)
    inherited = {k: v for k, v in parent.items() if k not in ("experiment_sha256", "implementation")}
    extra = {"experiment_sha256", "implementation", "revision", "amp_recovery",
             "parent_experiment_sha256", "parent_last_epoch", "parent_artifacts"}
    if ({k: v for k, v in plan.items() if k not in extra} != inherited
            or plan["revision"] != REVISION or plan["amp_recovery"] != POLICY
            or plan["implementation"] != code_identity(root)
            or plan["parent_experiment_sha256"] != parent["experiment_sha256"]):
        raise ValueError("Changed continuation code, parent or frozen experiment settings.")
    records = history(root, parent)
    if records[-1]["epoch"] != plan["parent_last_epoch"]:
        raise ValueError("Parent advanced after continuation freeze; use only the 65b runner.")
    expected = {"protocol.json", "training_identity.json"}
    for r in records:
        expected.update({f"epochs/epoch_{r['epoch']:03d}/complete.json",
                         f"epochs/epoch_{r['epoch']:03d}/{r['file']}"})
    expected.update(f"validation/epoch_{s['epoch']:03d}/summary.json" for s in summaries(root, parent, records))
    if set(plan["parent_artifacts"]) != expected:
        raise ValueError("Incomplete parent lineage manifest.")
    output = c.output_root(root, parent)
    for name, digest in plan["parent_artifacts"].items():
        if c.sha256_file(output/name) != digest:
            raise ValueError(f"Parent artifact changed: {name}")
    return parent
