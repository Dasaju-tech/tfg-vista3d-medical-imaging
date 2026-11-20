from pathlib import Path

source = Path(
    "src/vista3d_tfg/analysis/final_experimental_synthesis.py"
)
test = Path(
    "tests/test_final_experimental_synthesis.py"
)

if not source.is_file():
    raise SystemExit(f"ERROR: no existe {source}")
if not test.is_file():
    raise SystemExit(f"ERROR: no existe {test}")

text = source.read_text(encoding="utf-8")

old = '''            "dice_ge_050_volume_auroc": float(
                cyst_key[
                    "dice_ge_050_volume_auroc"
                ]
            ),
            "branch_status": (
                cyst_final[
                    "cyst_branch_status"
                ]
            ),
'''

new = '''            "dice_ge_050_volume_auroc": float(
                cyst_key[
                    "dice_ge_050_volume_auroc"
                ]
            ),
            "volume_response_bins": (
                cyst_key[
                    "volume_response_bins"
                ]
            ),
            "branch_status": (
                cyst_final[
                    "cyst_branch_status"
                ]
            ),
'''

if new in text:
    print(f"ALREADY FIXED: {source}")
elif old not in text:
    raise SystemExit(
        "ERROR: no encuentro el bloque esperado en "
        f"{source}. No se ha modificado."
    )
else:
    source.write_text(
        text.replace(old, new, 1),
        encoding="utf-8",
    )
    print(f"OK: {source}")

test_text = test.read_text(encoding="utf-8")
marker = "test_final_synthesis_propagates_cyst_volume_response_bins"

if marker not in test_text:
    addition = '''

def test_final_synthesis_propagates_cyst_volume_response_bins() -> None:
    artifacts = _artifacts()
    expected = [
        {
            "bin": 1,
            "count": 6,
            "volume_median_ml": 0.17,
            "any_overlap_fraction": 0.0,
            "dice_ge_050_fraction": 0.0,
            "dice_mean": 0.0,
            "recall_mean": 0.0,
        }
    ]
    artifacts["cyst"]["key_results"][
        "volume_response_bins"
    ] = expected

    payload = synthesize_final_experimental_results(
        artifacts
    )

    assert payload["cyst_branch"][
        "volume_response_bins"
    ] == expected
'''
    test.write_text(
        test_text + addition,
        encoding="utf-8",
    )
    print(f"OK: {test}")
else:
    print(f"ALREADY FIXED: {test}")

print()
print("Block 53 -> Block 54 cyst-bin propagation fix completed.")
