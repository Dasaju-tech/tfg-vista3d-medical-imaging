from pathlib import Path

path = Path("tests/test_final_manuscript.py")
if not path.is_file():
    raise SystemExit(f"ERROR: no existe {path}")

text = path.read_text(encoding="utf-8")

old1 = '''def test_results_preserve_reporting_boundaries() -> None:
    text = _results_section(_payload())
    assert "ground-truth-derived" in text
    assert "simulated-user" in text
    assert "not as clinician" in text
    assert "descriptive" in text
'''

new1 = '''def test_results_preserve_reporting_boundaries() -> None:
    text = _results_section(_payload())
    normalized = " ".join(text.split())

    assert "ground-truth-derived" in normalized
    assert "simulated-user" in normalized
    assert "not as clinician" in normalized
    assert "descriptive" in normalized
'''

old2 = '''def test_limitations_reject_overclaims() -> None:
    text = _limitations_section(_payload()).lower()
    assert "official kits23 challenge submission" in text
    assert "oracle experiments" in text
    assert "neither a clinician study" in text
    assert "clinically validated cyst-size cutoff" in text
'''

new2 = '''def test_limitations_reject_overclaims() -> None:
    text = _limitations_section(_payload()).lower()
    normalized = " ".join(text.split())

    assert "official kits23 challenge submission" in normalized
    assert "oracle experiments" in normalized
    assert "neither a clinician study" in normalized
    assert "clinically validated cyst-size cutoff" in normalized
'''

changed = False

if new1 in text:
    print("ALREADY FIXED: results-boundary test")
elif old1 in text:
    text = text.replace(old1, new1, 1)
    changed = True
    print("OK: results-boundary test")
else:
    raise SystemExit(
        "ERROR: no encuentro test_results_preserve_reporting_boundaries esperado."
    )

if new2 in text:
    print("ALREADY FIXED: limitations test")
elif old2 in text:
    text = text.replace(old2, new2, 1)
    changed = True
    print("OK: limitations test")
else:
    raise SystemExit(
        "ERROR: no encuentro test_limitations_reject_overclaims esperado."
    )

if changed:
    path.write_text(text, encoding="utf-8")
    print(f"UPDATED: {path}")

print()
print("Block 55 whitespace-robust test fix completed.")
