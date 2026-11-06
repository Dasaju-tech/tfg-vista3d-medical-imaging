from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable


FORBIDDEN_CLAIM_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bdiagnos(?:is|e|ed|tic)\b", "diagnostic_claim"),
    (r"\bmalignan(?:t|cy)\b", "malignancy_claim"),
    (r"\bbenign\b", "benignity_claim"),
    (r"\bcancer\b", "cancer_claim"),
    (r"\bcarcinoma\b", "carcinoma_claim"),
    (r"\bstage(?:d|s|ing)?\b", "staging_claim"),
    (r"\bmetasta(?:sis|ses|tic)\b", "metastatic_claim"),
    (r"\btreat(?:ment|ed|ing)?\b", "treatment_claim"),
    (r"\btherapy\b", "treatment_claim"),
    (r"\brecommend(?:ation|ed|s|ing)?\b", "recommendation_claim"),
    (r"\bbiopsy\b", "biopsy_claim"),
    (r"\bsuspicious\b", "suspicion_claim"),
    (r"\bpatholog(?:y|ic|ical)\b", "pathology_claim"),
)

NUMBER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])[-+]?(?:\d+(?:\.\d+)?|\.\d+)"
)


@dataclass(frozen=True, slots=True)
class AllowedFact:
    key: str
    value: str
    unit: str | None
    source_path: str


@dataclass(frozen=True, slots=True)
class LLMInputPacket:
    schema_version: str
    case_id: str
    task: str
    authoritative_findings: dict[str, Any]
    allowed_facts: tuple[AllowedFact, ...]
    required_limitations: tuple[str, ...]
    forbidden_claim_categories: tuple[str, ...]
    output_contract: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LLMCandidate:
    case_id: str
    narrative: str
    limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AuditIssue:
    code: str
    message: str
    evidence: str | None = None


@dataclass(frozen=True, slots=True)
class LLMAuditResult:
    passed: bool
    issue_count: int
    issues: tuple[AuditIssue, ...]
    observed_numeric_tokens: tuple[str, ...]
    allowed_numeric_tokens: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _format_number_variants(value: float | int) -> set[str]:
    variants: set[str] = set()

    numeric = float(value)

    if math.isfinite(numeric):
        variants.add(str(int(round(numeric))))

        for decimals in (1, 2, 3, 4):
            formatted = f"{numeric:.{decimals}f}"
            variants.add(formatted)
            variants.add(formatted.rstrip("0").rstrip("."))

    return {variant for variant in variants if variant}


def _iter_authoritative_numbers(value: Any) -> Iterable[float | int]:
    if isinstance(value, bool):
        return

    if isinstance(value, (int, float)):
        yield value
        return

    if isinstance(value, dict):
        for nested in value.values():
            yield from _iter_authoritative_numbers(nested)
        return

    if isinstance(value, (list, tuple)):
        for nested in value:
            yield from _iter_authoritative_numbers(nested)


def _is_meta_limitation_clause(clause: str) -> bool:
    normalized = " ".join(clause.lower().split())

    meta_subject = any(
        token in normalized
        for token in (
            "claim",
            "claims",
            "interpretation",
            "interpretations",
            "conclusion",
            "conclusions",
            "assessment",
            "assessments",
            "recommendation",
            "recommendations",
        )
    )
    negative_scope = any(
        token in normalized
        for token in (
            "no ",
            "not ",
            "does not ",
            "do not ",
            "without ",
            "neither ",
        )
    )
    reporting_scope = any(
        token in normalized
        for token in (
            "generate",
            "generated",
            "provided",
            "made",
            "included",
            "offered",
            "reported",
            "performed",
        )
    )

    return meta_subject and negative_scope and reporting_scope


def _append_fact(
    facts: list[AllowedFact],
    *,
    key: str,
    value: float | int | str,
    unit: str | None,
    source_path: str,
) -> None:
    if isinstance(value, float):
        rendered = f"{value:.2f}"
    else:
        rendered = str(value)

    facts.append(
        AllowedFact(
            key=key,
            value=rendered,
            unit=unit,
            source_path=source_path,
        )
    )


def extract_allowed_facts(
    findings: dict[str, Any],
) -> tuple[AllowedFact, ...]:
    auto = findings["automatic_measurements"]
    cyst = findings["interactive_cyst"]

    facts: list[AllowedFact] = []

    _append_fact(
        facts,
        key="left_kidney_core_volume_ml",
        value=float(auto["left_kidney_core_volume_ml"]),
        unit="mL",
        source_path="automatic_measurements.left_kidney_core_volume_ml",
    )
    _append_fact(
        facts,
        key="right_kidney_core_volume_ml",
        value=float(auto["right_kidney_core_volume_ml"]),
        unit="mL",
        source_path="automatic_measurements.right_kidney_core_volume_ml",
    )
    _append_fact(
        facts,
        key="automatic_left_cyst_volume_ml",
        value=float(auto["automatic_left_cyst_volume_ml"]),
        unit="mL",
        source_path="automatic_measurements.automatic_left_cyst_volume_ml",
    )
    _append_fact(
        facts,
        key="automatic_right_cyst_volume_ml",
        value=float(auto["automatic_right_cyst_volume_ml"]),
        unit="mL",
        source_path="automatic_measurements.automatic_right_cyst_volume_ml",
    )
    _append_fact(
        facts,
        key="interactive_cyst_predicted_volume_ml",
        value=float(cyst["predicted_volume_ml"]),
        unit="mL",
        source_path="interactive_cyst.predicted_volume_ml",
    )
    _append_fact(
        facts,
        key="interactive_cyst_selected_kidney_distance_mm",
        value=float(cyst["selected_kidney_distance_mm"]),
        unit="mm",
        source_path="interactive_cyst.selected_kidney_distance_mm",
    )
    _append_fact(
        facts,
        key="interactive_cyst_laterality_margin_mm",
        value=float(cyst["laterality_margin_mm"]),
        unit="mm",
        source_path="interactive_cyst.laterality_margin_mm",
    )
    _append_fact(
        facts,
        key="interactive_cyst_class_label",
        value=int(cyst["class_label"]),
        unit=None,
        source_path="interactive_cyst.class_label",
    )

    for index, coordinate in enumerate(cyst["user_point_xyz"]):
        _append_fact(
            facts,
            key=f"user_point_{'xyz'[index]}",
            value=int(coordinate),
            unit="voxel",
            source_path=f"interactive_cyst.user_point_xyz[{index}]",
        )

    return tuple(facts)


def build_llm_input_packet(
    findings: dict[str, Any],
) -> LLMInputPacket:
    if findings.get("ground_truth_used") is not False:
        raise ValueError(
            "LLM reporting contract requires ground_truth_used=false."
        )

    if findings.get("clinical_validation") is not False:
        raise ValueError(
            "LLM reporting contract requires clinical_validation=false."
        )

    limitations = tuple(
        str(value)
        for value in findings.get("limitations", ())
    )

    if not limitations:
        raise ValueError(
            "Structured findings must contain explicit limitations."
        )

    forbidden_categories = tuple(
        sorted({category for _, category in FORBIDDEN_CLAIM_PATTERNS})
    )

    return LLMInputPacket(
        schema_version="1.0",
        case_id=str(findings["case_id"]),
        task=(
            "Rewrite the authoritative segmentation-derived findings into "
            "concise research prose. Do not infer, diagnose, interpret "
            "malignancy, stage disease, recommend treatment, or introduce "
            "facts that are absent from the packet."
        ),
        authoritative_findings=findings,
        allowed_facts=extract_allowed_facts(findings),
        required_limitations=limitations,
        forbidden_claim_categories=forbidden_categories,
        output_contract={
            "type": "object",
            "required": [
                "case_id",
                "narrative",
                "limitations",
            ],
            "additional_properties": False,
            "fields": {
                "case_id": (
                    "Must exactly equal the packet case_id."
                ),
                "narrative": (
                    "Plain factual prose based only on authoritative_findings."
                ),
                "limitations": (
                    "Must reproduce every required limitation exactly once."
                ),
            },
        },
    )


def render_llm_prompt(packet: LLMInputPacket) -> str:
    facts = "\n".join(
        (
            f"- {fact.key}: {fact.value}"
            + (f" {fact.unit}" if fact.unit else "")
        )
        for fact in packet.allowed_facts
    )

    limitations = "\n".join(
        f"- {limitation}"
        for limitation in packet.required_limitations
    )

    return (
        "You are a constrained medical-imaging research report verbalizer.\n"
        "Your only authoritative source is the structured packet below.\n\n"
        "RULES\n"
        "1. Do not add clinical interpretation or diagnosis.\n"
        "2. Do not change any numerical value.\n"
        "3. Do not change laterality or class identity.\n"
        "4. Do not infer malignancy, benignity, stage, treatment, or prognosis.\n"
        "5. Return JSON only with keys: case_id, narrative, limitations.\n"
        "6. Reproduce every required limitation exactly once.\n"
        "7. The selected-kidney distance is the distance from the USER CLICK "
        "to the selected automatic kidney; never describe it as a cyst-to-"
        "kidney distance.\n"
        "8. Automatic left/right cyst-class measurements do not define the "
        "laterality of the interactive cyst.\n\n"
        f"CASE_ID\n{packet.case_id}\n\n"
        f"ALLOWED_NUMERIC_FACTS\n{facts}\n\n"
        "REQUIRED_LIMITATIONS\n"
        f"{limitations}\n\n"
        "AUTHORITATIVE_FINDINGS_JSON\n"
        + json.dumps(
            packet.authoritative_findings,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def parse_candidate_json(text: str) -> LLMCandidate:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"LLM output is not valid JSON: {error}"
        ) from error

    if not isinstance(payload, dict):
        raise ValueError("LLM output JSON must be an object.")

    allowed_keys = {"case_id", "narrative", "limitations"}

    if set(payload) != allowed_keys:
        raise ValueError(
            "LLM output must contain exactly case_id, narrative, limitations."
        )

    limitations = payload["limitations"]

    if not isinstance(limitations, list):
        raise ValueError("LLM limitations must be a JSON array.")

    return LLMCandidate(
        case_id=str(payload["case_id"]),
        narrative=str(payload["narrative"]),
        limitations=tuple(str(value) for value in limitations),
    )


def _allowed_numeric_tokens(
    packet: LLMInputPacket,
) -> set[str]:
    allowed: set[str] = set()

    for fact in packet.allowed_facts:
        try:
            numeric = float(fact.value)
        except ValueError:
            continue

        allowed.update(_format_number_variants(numeric))

    # Block 26 exposed the complete authoritative findings JSON to the model.
    # Therefore a number copied exactly from that JSON is factual even if it
    # was not part of the preferred narrative whitelist.
    for numeric in _iter_authoritative_numbers(
        packet.authoritative_findings
    ):
        if isinstance(numeric, bool):
            continue

        numeric_float = float(numeric)

        if not math.isfinite(numeric_float):
            continue

        allowed.update(_format_number_variants(numeric_float))
        allowed.add(str(numeric))
        allowed.add(repr(numeric))

    # Case IDs can contain digits but are validated separately and should not
    # be interpreted as medical measurements.
    case_suffix = packet.case_id.split("_")[-1]
    allowed.add(case_suffix)
    allowed.add(str(int(case_suffix)))

    return allowed


def audit_llm_candidate(
    packet: LLMInputPacket,
    candidate: LLMCandidate,
) -> LLMAuditResult:
    issues: list[AuditIssue] = []

    if candidate.case_id != packet.case_id:
        issues.append(
            AuditIssue(
                code="case_id_mismatch",
                message=(
                    f"Expected case_id={packet.case_id!r}, "
                    f"received {candidate.case_id!r}."
                ),
                evidence=candidate.case_id,
            )
        )

    if tuple(candidate.limitations) != tuple(packet.required_limitations):
        issues.append(
            AuditIssue(
                code="limitations_mismatch",
                message=(
                    "Candidate limitations must reproduce the authoritative "
                    "limitations exactly and in the same order."
                ),
            )
        )

    lower_narrative = candidate.narrative.lower()

    claim_clauses = tuple(
        clause.strip()
        for clause in re.split(
            r"(?<=[.!?])\s+|;\s*",
            candidate.narrative,
        )
        if clause.strip()
    )

    for clause in claim_clauses:
        lower_clause = clause.lower()

        if _is_meta_limitation_clause(clause):
            continue

        for pattern, category in FORBIDDEN_CLAIM_PATTERNS:
            match = re.search(pattern, lower_clause)

            if match:
                issues.append(
                    AuditIssue(
                        code=category,
                        message=(
                            "Candidate contains a prohibited unsupported "
                            f"claim category: {category}."
                        ),
                        evidence=match.group(0),
                    )
                )

    expected_laterality = str(
        packet.authoritative_findings["interactive_cyst"]["laterality"]
    ).lower()
    expected_class = int(
        packet.authoritative_findings["interactive_cyst"]["class_label"]
    )

    # Laterality and class checks must distinguish the automatic bilateral
    # cyst-class measurements from the single interactive target. A sentence
    # that only describes automatic outputs may legitimately mention both
    # left and right cysts/classes.
    semantic_clauses = tuple(
        clause.strip()
        for clause in re.split(
            r"(?<=[.!?])\s+|;\s*",
            candidate.narrative,
        )
        if clause.strip()
    )

    cyst_context_pattern = re.compile(
        r"\b(left|right)\b[^.]{0,80}\bcyst\b"
        r"|\bcyst\b[^.]{0,80}\b(left|right)\b",
        flags=re.IGNORECASE,
    )
    explicit_laterality_pattern = re.compile(
        r"\blaterality\s*(?:of|is|=|:)?\s*(left|right)\b",
        flags=re.IGNORECASE,
    )
    class_pattern = re.compile(
        r"\b(?:class|label)(?:\s+label)?\s*(?:of|is|=|:)?\s*(116|117)\b",
        flags=re.IGNORECASE,
    )

    for clause in semantic_clauses:
        lower_clause = clause.lower()
        automatic_only = (
            "automatic" in lower_clause
            and "interactive" not in lower_clause
            and "user-guided" not in lower_clause
            and "user guided" not in lower_clause
        )

        if automatic_only:
            continue

        for match in cyst_context_pattern.finditer(clause):
            mentioned = next(
                (
                    group.lower()
                    for group in match.groups()
                    if group is not None
                ),
                None,
            )

            # Automatic bilateral cyst-class measurements are legitimate
            # even when the interactive target is on the opposite side.
            local_prefix = lower_clause[
                max(0, match.start() - 32):match.start()
            ]
            automatic_mention = "automatic" in local_prefix

            if automatic_mention:
                continue

            if mentioned is not None and mentioned != expected_laterality:
                issues.append(
                    AuditIssue(
                        code="cyst_laterality_mismatch",
                        message=(
                            f"Interactive cyst is {expected_laterality}, "
                            f"but candidate describes it as {mentioned}."
                        ),
                        evidence=match.group(0),
                    )
                )

        for match in explicit_laterality_pattern.finditer(clause):
            mentioned = match.group(1).lower()

            if mentioned != expected_laterality:
                issues.append(
                    AuditIssue(
                        code="cyst_laterality_mismatch",
                        message=(
                            f"Interactive cyst laterality is "
                            f"{expected_laterality}, but candidate states "
                            f"{mentioned}."
                        ),
                        evidence=match.group(0),
                    )
                )

        for match in class_pattern.finditer(clause):
            mention = int(match.group(1))

            if mention != expected_class:
                issues.append(
                    AuditIssue(
                        code="cyst_class_mismatch",
                        message=(
                            f"Interactive cyst class is {expected_class}, "
                            f"but candidate mentions {mention}."
                        ),
                        evidence=match.group(0),
                    )
                )

    # Numeric equality is not enough: the model must preserve the semantic
    # relation attached to selected_kidney_distance_mm.
    selected_distance = float(
        packet.authoritative_findings["interactive_cyst"][
            "selected_kidney_distance_mm"
        ]
    )
    selected_distance_variants = sorted(
        _format_number_variants(selected_distance),
        key=len,
        reverse=True,
    )
    selected_distance_pattern = re.compile(
        r"(?<![\d.])(?:"
        + "|".join(re.escape(value) for value in selected_distance_variants)
        + r")(?![\d.])"
    )

    for clause in semantic_clauses:
        if not selected_distance_pattern.search(clause):
            continue

        lower_clause = clause.lower()

        if "mm" not in lower_clause:
            continue

        explicit_click_relation = (
            "click" in lower_clause
            and "kidney" in lower_clause
        )
        canonical_field_alias = bool(
            re.search(
                r"\bselected(?:\s+automatic)?\s+kidney\s+distance\b",
                lower_clause,
            )
        )
        wrong_endpoint = bool(
            re.search(
                r"\b(?:cyst|lesion|mask)\b",
                lower_clause,
            )
        ) and "click" not in lower_clause

        if wrong_endpoint or not (
            explicit_click_relation
            or canonical_field_alias
        ):
            issues.append(
                AuditIssue(
                    code="distance_relation_mismatch",
                    message=(
                        "selected_kidney_distance_mm is the physical "
                        "distance from the user click to the selected "
                        "automatic kidney; the candidate changed that "
                        "relationship."
                    ),
                    evidence=clause,
                )
            )

    allowed_numbers = _allowed_numeric_tokens(packet)
    observed_numbers = tuple(
        NUMBER_PATTERN.findall(candidate.narrative)
    )

    for token in observed_numbers:
        normalized = token.lstrip("+")

        if normalized not in allowed_numbers:
            issues.append(
                AuditIssue(
                    code="unsupported_numeric_value",
                    message=(
                        "Candidate contains a numeric value not present "
                        "in the authoritative structured findings."
                    ),
                    evidence=token,
                )
            )

    return LLMAuditResult(
        passed=not issues,
        issue_count=len(issues),
        issues=tuple(issues),
        observed_numeric_tokens=observed_numbers,
        allowed_numeric_tokens=tuple(sorted(allowed_numbers)),
    )


def load_findings(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(payload, dict):
        raise ValueError("findings.json must contain a JSON object.")

    return payload


def write_llm_packet(
    packet: LLMInputPacket,
    output_directory: Path,
) -> tuple[Path, Path]:
    output_directory.mkdir(parents=True, exist_ok=True)

    packet_path = output_directory / "llm_input_packet.json"
    prompt_path = output_directory / "llm_prompt.txt"

    packet_path.write_text(
        json.dumps(
            packet.to_dict(),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    prompt_path.write_text(
        render_llm_prompt(packet),
        encoding="utf-8",
    )

    return packet_path, prompt_path


def write_audit_result(
    result: LLMAuditResult,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(
            result.to_dict(),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
