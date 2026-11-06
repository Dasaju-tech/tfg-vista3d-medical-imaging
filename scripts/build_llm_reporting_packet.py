from __future__ import annotations

import argparse

from vista3d_tfg.config import discover_project_root
from vista3d_tfg.reports.llm_contract import (
    build_llm_input_packet,
    load_findings,
    write_llm_packet,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a constrained LLM reporting packet from authoritative "
            "GT-free structured findings. No model is called."
        )
    )
    parser.add_argument("--case", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = discover_project_root()
    case_id = f"case_{args.case:05d}"

    findings_path = (
        project_root
        / "outputs"
        / "reports"
        / "structured"
        / case_id
        / "findings.json"
    )

    packet = build_llm_input_packet(
        load_findings(findings_path)
    )

    output_root = (
        project_root
        / "outputs"
        / "reports"
        / "llm_contract"
        / case_id
    )

    packet_path, prompt_path = write_llm_packet(
        packet,
        output_root,
    )

    print("Constrained LLM reporting packet")
    print(f"Case: {packet.case_id}")
    print(f"Allowed facts: {len(packet.allowed_facts)}")
    print(
        f"Required limitations: "
        f"{len(packet.required_limitations)}"
    )
    print(
        f"Forbidden claim categories: "
        f"{len(packet.forbidden_claim_categories)}"
    )
    print(f"Packet: {packet_path}")
    print(f"Prompt: {prompt_path}")
    print("LLM called: NO")
    print("LLM reporting packet PASSED.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
