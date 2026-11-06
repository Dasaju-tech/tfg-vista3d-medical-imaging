from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from vista3d_tfg.reports.llm_contract import (
    LLMAuditResult,
    LLMCandidate,
    LLMInputPacket,
    audit_llm_candidate,
    parse_candidate_json,
)


DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen3:4b-instruct"
DEFAULT_SEED = 20260828
DEFAULT_TEMPERATURE = 0.0
DEFAULT_NUM_PREDICT = 700
DEFAULT_KEEP_ALIVE = "5m"

JsonRequest = Callable[
    [str, str, dict[str, Any] | None, float],
    dict[str, Any],
]


@dataclass(frozen=True, slots=True)
class OllamaRuntimeInfo:
    base_url: str
    version: str
    models: tuple[str, ...]
    requested_model: str
    requested_model_present: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class OllamaGenerationConfig:
    base_url: str = DEFAULT_OLLAMA_URL
    model: str = DEFAULT_MODEL
    seed: int = DEFAULT_SEED
    temperature: float = DEFAULT_TEMPERATURE
    num_predict: int = DEFAULT_NUM_PREDICT
    keep_alive: str = DEFAULT_KEEP_ALIVE
    timeout_seconds: float = 180.0
    think: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class OllamaResponse:
    model: str
    created_at: str | None
    content: str
    done: bool
    done_reason: str | None
    total_duration_ns: int | None
    load_duration_ns: int | None
    prompt_eval_count: int | None
    prompt_eval_duration_ns: int | None
    eval_count: int | None
    eval_duration_ns: int | None
    raw_response: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LocalLLMRun:
    case_id: str
    provider: str
    endpoint: str
    model: str
    ollama_version: str
    generation_config: dict[str, Any]
    prompt_sha256: str
    response_sha256: str
    wall_time_seconds: float
    parsed: bool
    parse_error: str | None
    candidate: dict[str, Any] | None
    audit: dict[str, Any] | None
    accepted: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def _normalize_base_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")

    if not normalized.startswith(("http://", "https://")):
        raise ValueError(
            "Ollama base URL must start with http:// or https://."
        )

    return normalized


def _stdlib_json_request(
    method: str,
    url: str,
    payload: dict[str, Any] | None,
    timeout_seconds: float,
) -> dict[str, Any]:
    body = (
        json.dumps(payload).encode("utf-8")
        if payload is not None
        else None
    )

    request = Request(
        url,
        data=body,
        method=method,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )

    try:
        with urlopen(
            request,
            timeout=timeout_seconds,
        ) as response:
            raw = response.read().decode("utf-8")
    except HTTPError as error:
        details = error.read().decode(
            "utf-8",
            errors="replace",
        )
        raise RuntimeError(
            f"Ollama HTTP {error.code} for {url}: {details}"
        ) from error
    except URLError as error:
        raise RuntimeError(
            "Could not connect to the local Ollama server at "
            f"{url}. Install/start Ollama before running the local "
            "LLM experiment."
        ) from error

    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"Ollama returned non-JSON content from {url}."
        ) from error

    if not isinstance(decoded, dict):
        raise RuntimeError(
            f"Ollama returned an unexpected JSON value from {url}."
        )

    return decoded


def probe_ollama(
    *,
    base_url: str = DEFAULT_OLLAMA_URL,
    requested_model: str = DEFAULT_MODEL,
    timeout_seconds: float = 10.0,
    request_json: JsonRequest = _stdlib_json_request,
) -> OllamaRuntimeInfo:
    base_url = _normalize_base_url(base_url)

    version_payload = request_json(
        "GET",
        f"{base_url}/api/version",
        None,
        timeout_seconds,
    )
    tags_payload = request_json(
        "GET",
        f"{base_url}/api/tags",
        None,
        timeout_seconds,
    )

    version = str(version_payload.get("version", "unknown"))

    raw_models = tags_payload.get("models", [])

    if not isinstance(raw_models, list):
        raise RuntimeError(
            "Ollama /api/tags response does not contain a model list."
        )

    model_names: set[str] = set()

    for item in raw_models:
        if not isinstance(item, dict):
            continue

        for field in ("name", "model"):
            value = item.get(field)

            if isinstance(value, str) and value:
                model_names.add(value)

    models = tuple(sorted(model_names))

    return OllamaRuntimeInfo(
        base_url=base_url,
        version=version,
        models=models,
        requested_model=requested_model,
        requested_model_present=requested_model in model_names,
    )


def pull_ollama_model(
    *,
    model: str,
    base_url: str = DEFAULT_OLLAMA_URL,
    timeout_seconds: float = 1800.0,
    request_json: JsonRequest = _stdlib_json_request,
) -> dict[str, Any]:
    base_url = _normalize_base_url(base_url)

    response = request_json(
        "POST",
        f"{base_url}/api/pull",
        {
            "model": model,
            "stream": False,
        },
        timeout_seconds,
    )

    status = response.get("status")

    if status != "success":
        raise RuntimeError(
            f"Ollama model pull did not report success: {response}"
        )

    return response


def candidate_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "required": [
            "case_id",
            "narrative",
            "limitations",
        ],
        "properties": {
            "case_id": {
                "type": "string",
                "pattern": "^case_[0-9]{5}$",
            },
            "narrative": {
                "type": "string",
                "minLength": 1,
            },
            "limitations": {
                "type": "array",
                "items": {
                    "type": "string",
                },
                "minItems": 1,
            },
        },
        "additionalProperties": False,
    }


def render_ollama_prompt(
    *,
    base_prompt: str,
    schema: dict[str, Any],
) -> str:
    return (
        base_prompt.rstrip()
        + "\n\nOUTPUT_JSON_SCHEMA\n"
        + json.dumps(
            schema,
            indent=2,
            sort_keys=True,
        )
        + "\n\n"
        + (
            "Return exactly one JSON object matching OUTPUT_JSON_SCHEMA. "
            "Do not use Markdown fences and do not include commentary "
            "outside the JSON object.\n"
        )
    )


def build_chat_payload(
    *,
    prompt: str,
    config: OllamaGenerationConfig,
    schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if schema is None:
        schema = candidate_json_schema()

    if config.temperature != 0.0:
        raise ValueError(
            "Block 24 baseline requires temperature=0."
        )

    return {
        "model": config.model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You verbalize structured research findings. "
                    "Follow the supplied JSON output contract exactly. "
                    "Never invent facts or clinical interpretation."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "format": schema,
        "stream": False,
        "think": config.think,
        "keep_alive": config.keep_alive,
        "options": {
            "temperature": config.temperature,
            "seed": config.seed,
            "num_predict": config.num_predict,
        },
    }


def parse_ollama_response(
    payload: dict[str, Any],
) -> OllamaResponse:
    message = payload.get("message")

    if not isinstance(message, dict):
        raise ValueError(
            "Ollama chat response does not contain a message object."
        )

    content = message.get("content")

    if not isinstance(content, str) or not content.strip():
        raise ValueError(
            "Ollama chat response contains no textual content."
        )

    return OllamaResponse(
        model=str(payload.get("model", "unknown")),
        created_at=(
            str(payload["created_at"])
            if payload.get("created_at") is not None
            else None
        ),
        content=content,
        done=bool(payload.get("done", False)),
        done_reason=(
            str(payload["done_reason"])
            if payload.get("done_reason") is not None
            else None
        ),
        total_duration_ns=(
            int(payload["total_duration"])
            if payload.get("total_duration") is not None
            else None
        ),
        load_duration_ns=(
            int(payload["load_duration"])
            if payload.get("load_duration") is not None
            else None
        ),
        prompt_eval_count=(
            int(payload["prompt_eval_count"])
            if payload.get("prompt_eval_count") is not None
            else None
        ),
        prompt_eval_duration_ns=(
            int(payload["prompt_eval_duration"])
            if payload.get("prompt_eval_duration") is not None
            else None
        ),
        eval_count=(
            int(payload["eval_count"])
            if payload.get("eval_count") is not None
            else None
        ),
        eval_duration_ns=(
            int(payload["eval_duration"])
            if payload.get("eval_duration") is not None
            else None
        ),
        raw_response=payload,
    )


def generate_ollama_candidate(
    *,
    prompt: str,
    config: OllamaGenerationConfig,
    schema: dict[str, Any] | None = None,
    request_json: JsonRequest = _stdlib_json_request,
) -> OllamaResponse:
    base_url = _normalize_base_url(config.base_url)

    payload = build_chat_payload(
        prompt=prompt,
        config=config,
        schema=schema,
    )

    response = request_json(
        "POST",
        f"{base_url}/api/chat",
        payload,
        config.timeout_seconds,
    )

    return parse_ollama_response(response)


def evaluate_ollama_content(
    *,
    packet: LLMInputPacket,
    content: str,
) -> tuple[
    LLMCandidate | None,
    LLMAuditResult | None,
    str | None,
]:
    try:
        candidate = parse_candidate_json(content)
    except ValueError as error:
        return None, None, str(error)

    audit = audit_llm_candidate(
        packet,
        candidate,
    )

    return candidate, audit, None


def write_local_llm_run(
    *,
    output_root: Path,
    packet: LLMInputPacket,
    prompt: str,
    runtime: OllamaRuntimeInfo,
    config: OllamaGenerationConfig,
    response: OllamaResponse,
    candidate: LLMCandidate | None,
    audit: LLMAuditResult | None,
    parse_error: str | None,
    wall_time_seconds: float,
) -> LocalLLMRun:
    output_root.mkdir(parents=True, exist_ok=True)

    (output_root / "raw_response.json").write_text(
        json.dumps(
            response.raw_response,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    (output_root / "raw_content.txt").write_text(
        response.content,
        encoding="utf-8",
    )

    if candidate is not None:
        (output_root / "candidate.json").write_text(
            json.dumps(
                candidate.to_dict(),
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    if audit is not None:
        (output_root / "audit.json").write_text(
            json.dumps(
                audit.to_dict(),
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    accepted = bool(
        candidate is not None
        and audit is not None
        and audit.passed
        and parse_error is None
    )

    run = LocalLLMRun(
        case_id=packet.case_id,
        provider="ollama-local",
        endpoint=runtime.base_url,
        model=response.model,
        ollama_version=runtime.version,
        generation_config=config.to_dict(),
        prompt_sha256=_sha256_text(prompt),
        response_sha256=_sha256_text(response.content),
        wall_time_seconds=float(wall_time_seconds),
        parsed=candidate is not None,
        parse_error=parse_error,
        candidate=(
            candidate.to_dict()
            if candidate is not None
            else None
        ),
        audit=(
            audit.to_dict()
            if audit is not None
            else None
        ),
        accepted=accepted,
    )

    (output_root / "run.json").write_text(
        json.dumps(
            run.to_dict(),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return run
