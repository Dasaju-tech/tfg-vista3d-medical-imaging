from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from vista3d_tfg.config import discover_project_root
from vista3d_tfg.reports.ollama_local import (
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_URL,
    probe_ollama,
    pull_ollama_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe the local Ollama server and optionally pull the frozen "
            "Block 24 verbalization model."
        )
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get(
            "VISTA3D_OLLAMA_URL",
            DEFAULT_OLLAMA_URL,
        ),
    )
    parser.add_argument(
        "--pull",
        action="store_true",
        help="Pull the requested model if it is not already installed.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = discover_project_root()

    print("Local LLM runtime preparation")
    print(f"Ollama endpoint: {args.base_url}")
    print(f"Model:           {args.model}")

    runtime = probe_ollama(
        base_url=args.base_url,
        requested_model=args.model,
    )

    print(f"Ollama version:  {runtime.version}")
    print(
        f"Model installed: "
        f"{runtime.requested_model_present}"
    )

    if not runtime.requested_model_present:
        if not args.pull:
            raise RuntimeError(
                f"Model {args.model!r} is not installed. "
                "Re-run this command with --pull or execute "
                f"'ollama pull {args.model}'."
            )

        print("Pulling model. This can take several minutes ...")
        pull_ollama_model(
            model=args.model,
            base_url=args.base_url,
        )

        runtime = probe_ollama(
            base_url=args.base_url,
            requested_model=args.model,
        )

        if not runtime.requested_model_present:
            raise RuntimeError(
                "Ollama pull completed but the requested model is "
                "still absent from /api/tags."
            )

        print("Model pull:      PASSED")

    output_path = (
        project_root
        / "outputs"
        / "runtime"
        / "local_llm"
        / "ollama_runtime.json"
    )
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        json.dumps(
            runtime.to_dict(),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print(f"Runtime manifest: {output_path}")
    print("Local LLM runtime PREPARATION PASSED.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
