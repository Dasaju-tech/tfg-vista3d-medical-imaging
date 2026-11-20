from pathlib import Path

path = Path(
    "src/vista3d_tfg/evaluation/tumor_scope_audit.py"
)

if not path.is_file():
    raise SystemExit(f"Missing file: {path}")

text = path.read_text(encoding="utf-8")

old = '''        "tumor_status": {
            "positive": {
                "count": len(tumor_positive),
                "hd95_reduction_mm": _summary(
                    row.hd95_reduction_mm
                    for row in tumor_positive
                ),
            },
            "negative": {
                "count": len(tumor_negative),
                "hd95_reduction_mm": _summary(
                    row.hd95_reduction_mm
                    for row in tumor_negative
                ),
            },
        },
'''

new = '''        "tumor_status": {
            "positive": {
                "count": len(tumor_positive),
                "hd95_reduction_mm": (
                    _summary(
                        row.hd95_reduction_mm
                        for row in tumor_positive
                    )
                    if tumor_positive
                    else None
                ),
            },
            "negative": {
                "count": len(tumor_negative),
                "hd95_reduction_mm": (
                    _summary(
                        row.hd95_reduction_mm
                        for row in tumor_negative
                    )
                    if tumor_negative
                    else None
                ),
            },
        },
'''

count = text.count(old)

if count == 0:
    if new in text:
        print("Block 34A fix is already applied.")
        raise SystemExit(0)

    raise SystemExit(
        "Expected Block 34 tumor_status section was not found. "
        "No file was modified."
    )

if count != 1:
    raise SystemExit(
        f"Expected exactly one matching block, found {count}. "
        "No file was modified."
    )

path.write_text(text.replace(old, new, 1), encoding="utf-8")

print("Block 34A fix applied successfully.")
print(f"Modified: {path}")
