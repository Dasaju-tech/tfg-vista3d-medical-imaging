from pathlib import Path

path = Path(
    "scripts/analyze_nv_segment_ct_expanded100_cyst_failures.py"
)
if not path.is_file():
    raise SystemExit(f"ERROR: no existe {path}")

text = path.read_text(encoding="utf-8")

old = '''def _load_expanded100_case_ids(
    manifest_path: Path,
) -> tuple[str, ...]:
    payload = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    candidates = (
        payload.get("case_ids"),
        payload.get("selected_case_ids"),
        payload.get("cases"),
    )

    for candidate in candidates:
        if isinstance(candidate, list):
            values = []
            for item in candidate:
                if isinstance(item, str):
                    values.append(item)
                elif isinstance(item, dict) and "case_id" in item:
                    values.append(str(item["case_id"]))
                else:
                    break
            else:
                return tuple(values)

    raise ValueError(
        f"Could not read case IDs from {manifest_path}."
    )
'''

new = '''def _load_expanded100_case_ids(
    manifest_path: Path,
) -> tuple[str, ...]:
    payload = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    discovered: list[str] = []
    seen: set[str] = set()

    def add(value: object) -> None:
        if not isinstance(value, str):
            return

        candidate = value.strip()
        if (
            candidate.startswith("case_")
            and len(candidate) == 10
            and candidate[5:].isdigit()
            and candidate not in seen
        ):
            seen.add(candidate)
            discovered.append(candidate)

    def visit(value: object) -> None:
        if isinstance(value, dict):
            for key in (
                "case_id",
                "case",
                "id",
            ):
                if key in value:
                    add(value[key])

            for nested in value.values():
                visit(nested)
            return

        if isinstance(value, (list, tuple)):
            for nested in value:
                visit(nested)
            return

        add(value)

    visit(payload)

    if len(discovered) != 100:
        raise ValueError(
            f"Expected 100 unique expanded100 case IDs in "
            f"{manifest_path}, found {len(discovered)}: "
            f"{discovered[:10]}"
        )

    return tuple(discovered)
'''

if new in text:
    print("ALREADY FIXED")
elif old not in text:
    raise SystemExit(
        "ERROR: no encuentro la función original esperada. "
        "No he modificado el archivo."
    )
else:
    path.write_text(
        text.replace(old, new, 1),
        encoding="utf-8",
    )
    print(f"OK: {path}")

test_path = Path(
    "tests/test_cyst_failure_stratification.py"
)
if test_path.is_file():
    test_text = test_path.read_text(encoding="utf-8")
    marker = "test_expanded100_manifest_reader_handles_nested_case_ids"

    if marker not in test_text:
        addition = '''

def test_expanded100_manifest_reader_handles_nested_case_ids(
    tmp_path: Path,
) -> None:
    import importlib.util
    import json

    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "analyze_nv_segment_ct_expanded100_cyst_failures.py"
    )
    spec = importlib.util.spec_from_file_location(
        "cyst_failure_script",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    manifest = {
        "selection": {
            "frozen_cases": [
                {
                    "metadata": {
                        "case_id": f"case_{index:05d}"
                    }
                }
                for index in range(100)
            ]
        }
    }

    path = tmp_path / "selection.json"
    path.write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    case_ids = module._load_expanded100_case_ids(path)

    assert len(case_ids) == 100
    assert case_ids[0] == "case_00000"
    assert case_ids[-1] == "case_00099"
'''
        test_path.write_text(
            test_text + addition,
            encoding="utf-8",
        )
        print(f"OK: {test_path}")
    else:
        print(f"ALREADY FIXED: {test_path}")

print()
print("Block 50 manifest-reader fix completed.")
