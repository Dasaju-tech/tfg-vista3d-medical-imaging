"""Build an offline VISTA3D case viewer and an audited Ollama research report."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from html import escape
from http.client import HTTPConnection
import json
from pathlib import Path
from string import Template
import sys
from time import perf_counter
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
TEMPLATE = Path(__file__).with_name("block77_local_demo.html")
HISTORICAL_POINT = (68, 372, 151)
STRUCTURES = (
    (14, "Riñón izquierdo", "#4fd1c5"),
    (5, "Riñón derecho", "#60a5fa"),
    (116, "Clase quiste izquierda, automática", "#fbbf24"),
    (117, "Clase quiste derecha, automática", "#fb923c"),
    (None, "Máscara de quiste por punto", "#e879f9"),
)


def local_json_request(method, url, payload, timeout):
    from vista3d_tfg.reports.ollama_local import DEFAULT_OLLAMA_URL

    if not url.startswith(DEFAULT_OLLAMA_URL + "/"):
        raise ValueError("La demo usa el servidor de Ollama en 127.0.0.1:11434.")
    connection = HTTPConnection("127.0.0.1", 11434, timeout=timeout)
    try:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        connection.request(method, url.removeprefix(DEFAULT_OLLAMA_URL), body,
                           {"Content-Type": "application/json"})
        response = connection.getresponse()
        if response.status != 200:
            raise RuntimeError(f"Ollama devolvió HTTP {response.status} en {method} {url}.")
        return json.loads(response.read())
    except OSError as exc:
        raise RuntimeError(f"Abre Ollama y comprueba 127.0.0.1:11434 ({exc}).") from exc
    finally:
        connection.close()


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def preflight(image_path, point, environment):
    import nibabel as nib
    import numpy as np
    from vista3d_tfg.integration.nv_inference import load_upstream_label_dict, validate_renal_label_contract
    from vista3d_tfg.integration.nv_segment import probe_runtime, run_checked, runtime_paths, validate_runtime_probe
    from vista3d_tfg.reports.ollama_local import DEFAULT_MODEL, probe_ollama

    if not image_path.is_file():
        raise FileNotFoundError(f"Falta el TAC local en {image_path}.")
    image = nib.load(str(image_path))
    if len(image.shape) != 3 or any(not 0 <= i < n for i, n in zip(point, image.shape)):
        raise ValueError(f"El punto {point} debe estar dentro del TAC 3D {image.shape}.")
    if (not np.isfinite(image.affine).all() or abs(np.linalg.det(image.affine[:3, :3])) < 1e-8
            or image.header.get_xyzt_units()[0] not in ("mm", "unknown")):
        raise ValueError("El TAC debe tener una rejilla KiTS23 válida en mm.")
    if not TEMPLATE.is_file():
        raise FileNotFoundError(f"Falta la plantilla de la demo en {TEMPLATE}.")

    ollama = probe_ollama(request_json=local_json_request)
    if not ollama.requested_model_present:
        raise RuntimeError(f"Instala el modelo local con ollama pull {DEFAULT_MODEL}.")
    paths = runtime_paths(ROOT)
    config = paths.ct_bundle_root / "configs"
    if not (paths.checkout_root / ".git").exists() or not (config / "inference.json").is_file():
        raise FileNotFoundError("Prepara el runtime con python scripts/prepare_nv_segment_ct_runtime.py --skip-install.")
    validate_renal_label_contract(load_upstream_label_dict(config / "label_dict.json"))
    commit = run_checked(["git", "rev-parse", "HEAD"], cwd=paths.checkout_root).stdout.strip()
    gpu = probe_runtime(environment_name=environment)
    validate_runtime_probe(gpu)
    print(f"TAC {image.shape} · {gpu.cuda_device_name} · Ollama {ollama.version} · {DEFAULT_MODEL}", flush=True)
    return paths, commit, gpu, ollama


def load_aligned_masks(image_path, automatic_path, interactive_path):
    import nibabel as nib
    import numpy as np

    image = nib.load(str(image_path))
    masks = [nib.load(str(path)) for path in (automatic_path, interactive_path)]
    for mask in masks:
        if (mask.shape != image.shape or not np.allclose(mask.affine, image.affine, rtol=0, atol=1e-4)
                or mask.header.get_xyzt_units()[0] not in ("mm", "unknown")):
            raise ValueError("Las dos máscaras deben estar alineadas con el TAC original.")
    return (image, *masks)


def render_case(image, automatic, interactive, point, target):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import nibabel as nib
    import numpy as np
    from scipy import ndimage
    from vista3d_tfg.experiments.cyst_oracle_prompt import _interactive_foreground
    from vista3d_tfg.metrics.morphology import summarize_binary_mask

    canonical, automatic, interactive = map(nib.as_closest_canonical, (image, automatic, interactive))
    click_mm = nib.affines.apply_affine(image.affine, point)
    click = np.rint(nib.affines.apply_affine(np.linalg.inv(canonical.affine), click_mm)).astype(int)
    ct = np.asanyarray(canonical.dataobj)
    automatic_data = np.asanyarray(automatic.dataobj)
    interactive_mask, _, _ = _interactive_foreground(np.asanyarray(interactive.dataobj))
    figure, panels = plt.subplots(1, 3, figsize=(13.5, 4.6), facecolor="#0b1522", layout="constrained")
    for axis, (panel, title) in enumerate(zip(panels, ("Sagital", "Coronal", "Axial"), strict=True)):
        h, v = (i for i in range(3) if i != axis)
        spacing = canonical.header.get_zooms()[:3]
        panel.imshow(np.take(ct, click[axis], axis=axis).T, cmap="gray", vmin=-160, vmax=240,
                     origin="lower", aspect=spacing[v] / spacing[h])
        panel.plot(click[h], click[v], marker="+", color="white", markersize=11)
        panel.set_title(title, color="white", fontsize=11)
        panel.set_axis_off()

    geometry = []
    point_cloud = []
    for label, name, color in STRUCTURES:
        mask = interactive_mask if label is None else automatic_data == label
        slices = ndimage.find_objects(mask, max_label=1)[0]
        cropped = mask[slices] if slices else np.zeros((1, 1, 1), dtype=bool)
        source = interactive if label is None else automatic
        morphology = summarize_binary_mask(cropped, source.header.get_zooms()[:3])
        geometry.append({"name": name, "label": label, "color": color, **asdict(morphology)})
        if slices:
            surface = cropped & ~ndimage.binary_erosion(cropped)
            points = np.argwhere(surface) + np.array([item.start for item in slices])
            points = points[np.linspace(0, len(points) - 1, min(1600, len(points)), dtype=int)]
            world = nib.affines.apply_affine(canonical.affine, points) - click_mm
            point_cloud.append({"color": color, "points": np.round(world, 1).tolist()})
        for axis, panel in enumerate(panels):
            section = np.take(mask, click[axis], axis=axis).T
            if section.any() and not section.all():
                panel.contour(section, levels=[0.5], colors=[color], linewidths=0.75)

    figure.savefig(target, dpi=140, facecolor=figure.get_facecolor())
    plt.close(figure)
    return geometry, point_cloud


def generate_report(findings, runtime, output):
    from vista3d_tfg.reports.llm_contract import build_llm_input_packet
    from vista3d_tfg.reports.llm_prompt_v2 import render_llm_prompt_v2
    from vista3d_tfg.reports.ollama_local import (
        OllamaGenerationConfig, candidate_json_schema, evaluate_ollama_content,
        generate_ollama_candidate, render_ollama_prompt, write_local_llm_run,
    )

    packet = build_llm_input_packet(findings.to_dict())
    schema = candidate_json_schema()
    prompt = render_ollama_prompt(base_prompt=render_llm_prompt_v2(packet), schema=schema)
    config = OllamaGenerationConfig(keep_alive="0s")
    start = perf_counter()
    response = generate_ollama_candidate(prompt=prompt, config=config, schema=schema, request_json=local_json_request)
    wall_time = perf_counter() - start
    candidate, audit, reason = evaluate_ollama_content(packet=packet, content=response.content)
    if not response.done or response.done_reason == "length" or response.model != config.model:
        reason = "Respuesta incompleta o modelo devuelto distinto al solicitado."
    run = write_local_llm_run(output_root=output / "llm", packet=packet, prompt=prompt,
                              runtime=runtime, config=config, response=response, candidate=candidate,
                              audit=audit, parse_error=reason, wall_time_seconds=wall_time)
    if run.accepted:
        report = candidate.narrative + "\n\n" + "\n".join(candidate.limitations)
        return "Informe de Qwen3 aprobado por el auditor", report, True
    issues = ", ".join(issue.code for issue in audit.issues) if audit is not None else ""
    return f"Resumen determinista · candidato rechazado ({reason or issues})", None, False


def write_demo_page(output, case_id, point, point_source, morphology, cloud, status, report, accepted):
    rows = "".join(
        f'<tr><th scope="row"><span class="swatch" style="--label:{item["color"]}"></span>{escape(item["name"])}</th>'
        f'<td>{item["volume_ml"]:.2f}</td><td>{item["component_count"]}</td>'
        f'<td>{item["bounding_box_diagonal_mm"]:.2f}</td></tr>' for item in morphology
    )
    page = Template(TEMPLATE.read_text(encoding="utf-8")).substitute(
        case=escape(case_id), point=escape(f"{point} · {point_source}"), rows=rows,
        cloud=json.dumps(cloud, separators=(",", ":"), allow_nan=False), status=escape(status),
        state="accepted" if accepted else "fallback", report=escape(report),
        audit_link='<a href="llm/run.json">Ejecución y auditoría</a>' if (output / "llm/run.json").is_file() else "",
        automatic_mask=f"automatic/{case_id}/prediction.nii.gz",
    )
    (output / "demo.html").write_text(page, encoding="utf-8")


def run_demo(case_number, image_path, point, point_source, environment, prepared):
    from vista3d_tfg.integration.nv_inference import run_renal_inference, write_inference_result
    from vista3d_tfg.reports.structured_findings import build_structured_case_findings, write_structured_report
    from vista3d_tfg.workflows.user_guided_cyst import run_user_guided_cyst_segmentation

    paths, commit, gpu, ollama = prepared
    case_id = f"case_{case_number:05d}"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = ROOT / "outputs/demo/block77" / case_id / f"{timestamp}-{uuid4().hex[:8]}"
    output.mkdir(parents=True, exist_ok=False)
    print(f"VISTA3D · segmentación automática · {output}", flush=True)
    automatic = run_renal_inference(case_id=case_id, image_path=image_path, paths=paths,
                                   repository_commit=commit, project_output_root=output / "automatic",
                                   environment_name=environment)
    automatic_path = Path(automatic.prediction_path)
    write_inference_result(automatic, automatic_path.parent / "inference.json")

    guided_root = output / "guided"
    print(f"VISTA3D · segmentación con punto {point}", flush=True)
    guided = run_user_guided_cyst_segmentation(
        project_root=ROOT, case_number=case_number, point_xyz=point, nv_paths=paths,
        environment_name=environment, automatic_prediction_path=automatic_path, output_directory=guided_root,
    )
    guided_path = Path(guided.prediction_path)
    images = load_aligned_masks(image_path, automatic_path, guided_path)
    findings = build_structured_case_findings(case_id=case_id, image_path=image_path,
                                             automatic_prediction_path=automatic_path,
                                             workflow_path=guided_root / "workflow.json",
                                             interactive_prediction_path=guided_path)
    _, deterministic = write_structured_report(findings, output)
    morphology, cloud = render_case(*images, point, output / "tac.png")
    write_json(output / "morphology.json", {"case_id": case_id, "structures": morphology})

    print("Ollama · generación y auditoría del informe", flush=True)
    try:
        status, report, accepted = generate_report(findings, ollama, output)
    except (OSError, RuntimeError, ValueError) as exc:
        status, report, accepted = "Resumen determinista · error de Ollama", None, False
        write_json(output / "llm_error.json", {"error": str(exc)})
        print(f"Ollama: {exc}", file=sys.stderr)
    published_report = report if accepted else deterministic.read_text(encoding="utf-8")
    (output / "published_report.txt").write_text(published_report, encoding="utf-8")
    write_json(output / "demo.json", {"case_id": case_id, "segmenter": "NV-Segment-CT/VISTA3D",
                                     "repository_commit": commit, "runtime": asdict(gpu),
                                     "point_xyz": point, "point_source": point_source,
                                     "reference_read": False, "llm_accepted": accepted, "status": status})
    write_demo_page(output, case_id, point, point_source, morphology, cloud, status, published_report, accepted)
    return output, status


def main():
    from vista3d_tfg.integration.nv_segment import DEFAULT_RUNTIME_ENV

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("check", "run"), required=True)
    parser.add_argument("--case", type=int, default=2)
    parser.add_argument("--point", type=int, nargs=3, metavar=("X", "Y", "Z"),
                        help="Coordenadas de vóxel del NIfTI original.")
    parser.add_argument("--environment", default=DEFAULT_RUNTIME_ENV)
    args = parser.parse_args()
    if not 0 <= args.case <= 99999 or (args.case != 2 and args.point is None):
        parser.error("Indica un número de caso KiTS23 válido y --point X Y Z para casos distintos de 2.")
    point = tuple(args.point) if args.point is not None else HISTORICAL_POINT
    historical = args.case == 2 and point == HISTORICAL_POINT
    point_source = "punto histórico derivado de la referencia" if historical else "coordenadas aportadas por el operador"
    image_path = ROOT / "data/raw/kits23" / f"case_{args.case:05d}" / "imaging.nii.gz"
    prepared = preflight(image_path, point, args.environment)
    if args.stage == "check":
        print(f"Bloque 77 check PASSED · {point_source} · {point}")
        return
    output, status = run_demo(args.case, image_path, point, point_source, args.environment, prepared)
    print(status)
    print(f"Abre en el navegador: {output / 'demo.html'}")


if __name__ == "__main__":
    try:
        main()
    except (ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"Bloque 77 FAILED · {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
