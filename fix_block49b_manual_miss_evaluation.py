from pathlib import Path

changes = []

def replace_exact(path_str: str, old: str, new: str) -> None:
    path = Path(path_str)
    if not path.is_file():
        raise SystemExit(f"ERROR: no existe {path}")

    text = path.read_text(encoding="utf-8")

    if new in text:
        print(f"ALREADY FIXED: {path}")
        return

    if old not in text:
        raise SystemExit(
            f"ERROR: no encuentro el bloque esperado en {path}. "
            "No se ha modificado ese archivo."
        )

    path.write_text(
        text.replace(old, new, 1),
        encoding="utf-8",
    )
    changes.append(str(path))
    print(f"OK: {path}")


replace_exact(
    "src/vista3d_tfg/experiments/tumor_oracle_smoke.py",
    '''def _target_component(
    ground_truth: np.ndarray,
    prompt: TumorOraclePrompt,
) -> np.ndarray:
''',
    '''def _target_component(
    ground_truth: np.ndarray,
    prompt: TumorOraclePrompt,
    *,
    require_prompt_inside_target: bool = True,
) -> np.ndarray:
''',
)

replace_exact(
    "src/vista3d_tfg/experiments/tumor_oracle_smoke.py",
    '''    if not target[prompt.point_xyz]:
        raise RuntimeError(
            f"{prompt.case_id}: frozen oracle point is outside target."
        )
''',
    '''    if (
        require_prompt_inside_target
        and not target[prompt.point_xyz]
    ):
        raise RuntimeError(
            f"{prompt.case_id}: frozen oracle point is outside target."
        )
''',
)

replace_exact(
    "src/vista3d_tfg/experiments/tumor_oracle_smoke.py",
    '''def evaluate_tumor_oracle_smoke(
    *,
    prompt: TumorOraclePrompt,
    ground_truth_path: Path,
    automatic_prediction_path: Path,
    interactive_prediction_path: Path,
    wall_time_seconds: float,
) -> TumorOracleSmokeResult:
''',
    '''def evaluate_tumor_oracle_smoke(
    *,
    prompt: TumorOraclePrompt,
    ground_truth_path: Path,
    automatic_prediction_path: Path,
    interactive_prediction_path: Path,
    wall_time_seconds: float,
    require_prompt_inside_target: bool = True,
) -> TumorOracleSmokeResult:
''',
)

replace_exact(
    "src/vista3d_tfg/experiments/tumor_oracle_smoke.py",
    '''    target = _target_component(
        ground_truth,
        prompt,
    )
''',
    '''    target = _target_component(
        ground_truth,
        prompt,
        require_prompt_inside_target=(
            require_prompt_inside_target
        ),
    )
''',
)

replace_exact(
    "src/vista3d_tfg/experiments/tumor_click_robustness.py",
    '''def run_click_variant(
    *,
    base_prompt: TumorOraclePrompt,
    variant: ClickVariant,
    project_root: Path,
    nv_paths: NVSegmentPaths,
    environment_name: str = DEFAULT_RUNTIME_ENV,
    force: bool = False,
) -> TumorOracleSmokeResult:
''',
    '''def run_click_variant(
    *,
    base_prompt: TumorOraclePrompt,
    variant: ClickVariant,
    project_root: Path,
    nv_paths: NVSegmentPaths,
    environment_name: str = DEFAULT_RUNTIME_ENV,
    force: bool = False,
    require_prompt_inside_target: bool = True,
) -> TumorOracleSmokeResult:
''',
)

replace_exact(
    "src/vista3d_tfg/experiments/tumor_click_robustness.py",
    '''    result = evaluate_tumor_oracle_smoke(
        prompt=prompt,
        ground_truth_path=ground_truth_path,
        automatic_prediction_path=automatic_prediction_path,
        interactive_prediction_path=stable_prediction,
        wall_time_seconds=elapsed,
    )
''',
    '''    result = evaluate_tumor_oracle_smoke(
        prompt=prompt,
        ground_truth_path=ground_truth_path,
        automatic_prediction_path=automatic_prediction_path,
        interactive_prediction_path=stable_prediction,
        wall_time_seconds=elapsed,
        require_prompt_inside_target=(
            require_prompt_inside_target
        ),
    )
''',
)

replace_exact(
    "src/vista3d_tfg/experiments/manual_tumor_click_inference.py",
    '''    manual = run_click_variant(
        base_prompt=manual_prompt,
        variant=ClickVariant(
            case_id=case.case_id,
            variant="manual_ct_only",
            point_xyz=click.point_xyz,
            interior_radius_mm=0.0,
        ),
        project_root=project_root,
        nv_paths=nv_paths,
        environment_name=environment_name,
        force=force,
    )
''',
    '''    manual = run_click_variant(
        base_prompt=manual_prompt,
        variant=ClickVariant(
            case_id=case.case_id,
            variant="manual_ct_only",
            point_xyz=click.point_xyz,
            interior_radius_mm=0.0,
        ),
        project_root=project_root,
        nv_paths=nv_paths,
        environment_name=environment_name,
        force=force,
        require_prompt_inside_target=False,
    )
''',
)

test_path = Path("tests/test_manual_tumor_click_inference.py")
if not test_path.is_file():
    raise SystemExit(f"ERROR: no existe {test_path}")

test_text = test_path.read_text(encoding="utf-8")
marker = "def test_manual_miss_can_be_evaluated_without_oracle_guard"
if marker not in test_text:
    addition = '''

def test_manual_miss_can_be_evaluated_without_oracle_guard(
    tmp_path: Path,
) -> None:
    from vista3d_tfg.experiments.tumor_oracle_protocol import (
        TumorOraclePrompt,
    )
    from vista3d_tfg.experiments.tumor_oracle_smoke import (
        evaluate_tumor_oracle_smoke,
    )

    gt = np.zeros((20, 20, 20), dtype=np.uint8)
    automatic = np.zeros_like(gt)
    interactive = np.zeros_like(gt)

    gt[8:12, 8:12, 8:12] = 2
    interactive[2:5, 2:5, 2:5] = 1

    gt_path = tmp_path / "gt.nii.gz"
    auto_path = tmp_path / "auto.nii.gz"
    interactive_path = tmp_path / "interactive.nii.gz"

    _save(gt_path, gt)
    _save(auto_path, automatic)
    _save(interactive_path, interactive)

    prompt = TumorOraclePrompt(
        case_id="fixture",
        component_id=1,
        point_xyz=(3, 3, 3),
        target_voxels=4 ** 3,
        target_volume_ml=0.064,
        total_tumor_volume_ml=0.064,
        interior_radius_mm=0.0,
    )

    result = evaluate_tumor_oracle_smoke(
        prompt=prompt,
        ground_truth_path=gt_path,
        automatic_prediction_path=auto_path,
        interactive_prediction_path=interactive_path,
        wall_time_seconds=1.0,
        require_prompt_inside_target=False,
    )

    assert result.point_xyz == (3, 3, 3)
    assert result.interactive_target_dice == pytest.approx(0.0)
'''
    test_path.write_text(
        test_text + addition,
        encoding="utf-8",
    )
    changes.append(str(test_path))
    print(f"OK: {test_path}")
else:
    print(f"ALREADY FIXED: {test_path}")

print()
print("Fix completed.")
print(f"Files changed: {len(changes)}")
for item in changes:
    print(f"  {item}")
