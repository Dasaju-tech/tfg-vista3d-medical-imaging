from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from vista3d_tfg.reports.llm_contract import LLMInputPacket


PROMPT_VERSION = "minimal-facts-v2"


@dataclass(frozen=True, slots=True)
class MinimalVerbalizationFacts:
    case_id: str

    left_kidney_core_volume_ml: str
    right_kidney_core_volume_ml: str
    automatic_left_cyst_volume_ml: str
    automatic_right_cyst_volume_ml: str

    interactive_cyst_predicted_volume_ml: str
    interactive_cyst_laterality: str
    interactive_cyst_class_label: int

    selected_kidney_distance_mm: str
    laterality_margin_mm: str

    user_point_x: int
    user_point_y: int
    user_point_z: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _render_float(value: Any) -> str:
    return f"{float(value):.2f}"


def build_minimal_verbalization_facts(
    packet: LLMInputPacket,
) -> MinimalVerbalizationFacts:
    findings = packet.authoritative_findings
    auto = findings["automatic_measurements"]
    cyst = findings["interactive_cyst"]

    laterality = str(cyst["laterality"]).lower()

    if laterality not in {"left", "right"}:
        raise ValueError(
            "Interactive cyst laterality must be left or right."
        )

    class_label = int(cyst["class_label"])
    expected_class = 116 if laterality == "left" else 117

    if class_label != expected_class:
        raise ValueError(
            "Interactive cyst laterality and class label are inconsistent."
        )

    point = tuple(int(value) for value in cyst["user_point_xyz"])

    if len(point) != 3:
        raise ValueError(
            "Interactive user_point_xyz must contain exactly three values."
        )

    return MinimalVerbalizationFacts(
        case_id=packet.case_id,

        left_kidney_core_volume_ml=_render_float(
            auto["left_kidney_core_volume_ml"]
        ),
        right_kidney_core_volume_ml=_render_float(
            auto["right_kidney_core_volume_ml"]
        ),
        automatic_left_cyst_volume_ml=_render_float(
            auto["automatic_left_cyst_volume_ml"]
        ),
        automatic_right_cyst_volume_ml=_render_float(
            auto["automatic_right_cyst_volume_ml"]
        ),

        interactive_cyst_predicted_volume_ml=_render_float(
            cyst["predicted_volume_ml"]
        ),
        interactive_cyst_laterality=laterality,
        interactive_cyst_class_label=class_label,

        selected_kidney_distance_mm=_render_float(
            cyst["selected_kidney_distance_mm"]
        ),
        laterality_margin_mm=_render_float(
            cyst["laterality_margin_mm"]
        ),

        user_point_x=point[0],
        user_point_y=point[1],
        user_point_z=point[2],
    )


def render_llm_prompt_v2(
    packet: LLMInputPacket,
) -> str:
    facts = build_minimal_verbalization_facts(packet)

    limitations = "\n".join(
        f"- {value}"
        for value in packet.required_limitations
    )

    factual_lines = (
        f"- left_kidney_core_volume_ml: "
        f"{facts.left_kidney_core_volume_ml} mL",
        f"- right_kidney_core_volume_ml: "
        f"{facts.right_kidney_core_volume_ml} mL",
        f"- automatic_left_cyst_volume_ml: "
        f"{facts.automatic_left_cyst_volume_ml} mL",
        f"- automatic_right_cyst_volume_ml: "
        f"{facts.automatic_right_cyst_volume_ml} mL",
        f"- interactive_cyst_predicted_volume_ml: "
        f"{facts.interactive_cyst_predicted_volume_ml} mL",
        f"- interactive_cyst_laterality: "
        f"{facts.interactive_cyst_laterality}",
        f"- interactive_cyst_class_label: "
        f"{facts.interactive_cyst_class_label}",
        f"- selected_kidney_distance_mm: "
        f"{facts.selected_kidney_distance_mm} mm",
        f"- laterality_margin_mm: "
        f"{facts.laterality_margin_mm} mm",
        f"- user_point_xyz: "
        f"({facts.user_point_x}, {facts.user_point_y}, "
        f"{facts.user_point_z})",
    )

    return (
        "You are a constrained medical-imaging research report verbalizer.\n"
        f"Prompt protocol: {PROMPT_VERSION}\n\n"

        "TASK\n"
        "Convert only the supplied facts into concise factual research prose.\n"
        "Do not calculate, infer, explain, diagnose, or add medical meaning.\n\n"

        "SEMANTIC DEFINITIONS\n"
        "- automatic_left_cyst_volume_ml and "
        "automatic_right_cyst_volume_ml are bilateral automatic model "
        "measurements; they do not determine the side of the interactive "
        "target.\n"
        "- interactive_cyst_laterality and "
        "interactive_cyst_class_label describe the single interactive "
        "target.\n"
        "- selected_kidney_distance_mm means ONLY the physical distance "
        "from the USER-PROVIDED CLICK to the selected automatic kidney.\n"
        "- selected_kidney_distance_mm does NOT describe the distance, "
        "alignment, overlap, attachment, location, or spatial relationship "
        "between the cyst and the kidney.\n"
        "- laterality_margin_mm is only the routing margin used to select "
        "left versus right; do not give it additional clinical meaning.\n\n"

        "STRICT RULES\n"
        "1. Use only the values in FACTS below.\n"
        "2. Preserve every reported numerical value exactly as written.\n"
        "3. Do not compute ratios, differences, percentages, or new values.\n"
        "4. Preserve interactive laterality and class identity exactly.\n"
        "5. Never infer a cyst-to-kidney spatial relationship from the "
        "click-to-kidney distance.\n"
        "6. Do not make diagnostic, malignancy, staging, treatment, "
        "prognostic, pathological, or recommendation claims.\n"
        "7. Do not mention hidden metadata, segmentation labels other than "
        "the supplied interactive class label, file paths, hashes, or "
        "implementation details.\n"
        "8. Return JSON only with exactly the keys case_id, narrative, "
        "limitations.\n"
        "9. case_id must exactly match CASE_ID.\n"
        "10. Reproduce every REQUIRED_LIMITATION exactly once in the "
        "limitations array.\n"
        "11. Keep the narrative factual and concise; do not restate the "
        "limitations inside the narrative unless necessary for grammatical "
        "clarity.\n\n"

        f"CASE_ID\n{facts.case_id}\n\n"

        "FACTS\n"
        + "\n".join(factual_lines)
        + "\n\n"

        "REQUIRED_LIMITATIONS\n"
        f"{limitations}\n"
    )
