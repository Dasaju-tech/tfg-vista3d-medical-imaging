from pathlib import Path

path = Path("src/vista3d_tfg/experiments/tumor_click_depth_audit.py")

if not path.is_file():
    raise SystemExit(f"ERROR: no existe {path}")

text = path.read_text(encoding="utf-8")

old = '''        robustness[name] = {
            "dice_delta_vs_deep_mean": float(
                (values - deep).mean()
            ),
            "cases_within_0_05_dice_of_deep": int(
                np.count_nonzero(
                    np.abs(values - deep) <= 0.05
                )
            ),
            "cases_within_0_10_dice_of_deep": int(
                np.count_nonzero(
                    np.abs(values - deep) <= 0.10
                )
            ),
        }
'''

new = '''        difference = np.abs(values - deep)
        robustness[name] = {
            "dice_delta_vs_deep_mean": float(
                (values - deep).mean()
            ),
            "cases_within_0_05_dice_of_deep": int(
                np.count_nonzero(
                    difference <= (0.05 + 1e-12)
                )
            ),
            "cases_within_0_10_dice_of_deep": int(
                np.count_nonzero(
                    difference <= (0.10 + 1e-12)
                )
            ),
        }
'''

if old not in text:
    raise SystemExit(
        "ERROR: no encuentro exactamente el bloque esperado. "
        "No se ha modificado ningún archivo."
    )

updated = text.replace(old, new, 1)

if updated == text:
    raise SystemExit("ERROR: no se produjo ningún cambio.")

path.write_text(updated, encoding="utf-8")

print(f"OK: corregido {path}")
print("Cambio aplicado: tolerancia numerica 1e-12 para limites ±0.05 y ±0.10.")
