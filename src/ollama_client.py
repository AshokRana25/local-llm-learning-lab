"""Talk to the Ollama service over its local HTTP API."""

from __future__ import annotations

import json
from collections.abc import Iterator

import requests

BASE_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "llama3.2:3b"


class OllamaError(RuntimeError):
    """A service, model, or streaming error with a readable message."""


def _session() -> requests.Session:
    session = requests.Session()
    # A corporate HTTP proxy should never receive local prompts.
    session.trust_env = False
    return session


def _check_response(response: requests.Response) -> None:
    if response.ok:
        return
    try:
        detail = response.json().get("error", response.text)
    except ValueError:
        detail = response.text
    raise OllamaError(f"Ollama HTTP {response.status_code}: {str(detail)[:400]}")


def list_models() -> list[str]:
    """List installed models without loading one into memory."""
    try:
        with _session() as session:
            response = session.get(f"{BASE_URL}/api/tags", timeout=(3, 10))
            _check_response(response)
            return sorted(model["name"] for model in response.json()["models"])
    except requests.RequestException as exc:
        raise OllamaError(
            "Cannot reach Ollama. Open Ollama from the Start menu, then check again. "
            "If needed, run 'ollama serve' in a separate terminal."
        ) from exc
    except (ValueError, KeyError, TypeError) as exc:
        raise OllamaError("Ollama returned an unexpected model list.") from exc


def stream_chat(
    messages: list[dict[str, str]],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    num_ctx: int = 4096,
) -> Iterator[str]:
    """Yield text pieces; never treat a broken stream as a complete answer."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "keep_alive": "2m",
        "options": {
            "temperature": temperature,
            "num_ctx": num_ctx,
            "num_predict": 512,
        },
    }
    try:
        with _session() as session:
            with session.post(
                f"{BASE_URL}/api/chat", json=payload, stream=True, timeout=(5, 300)
            ) as response:
                _check_response(response)
                for line in response.iter_lines():
                    if not line:
                        continue
                    event = json.loads(line)
                    if event.get("error"):
                        raise OllamaError(str(event["error"]))
                    content = event.get("message", {}).get("content", "")
                    if content:
                        yield content
                    if event.get("done"):
                        return
        raise OllamaError("Ollama stopped before completing the answer. Please retry.")
    except requests.Timeout as exc:
        raise OllamaError("Ollama timed out. Close other heavy apps and retry.") from exc
    except requests.RequestException as exc:
        raise OllamaError("Ollama connection failed. Check that Ollama is running.") from exc
    except (ValueError, AttributeError, TypeError) as exc:
        raise OllamaError("Ollama returned an invalid streaming response.") from exc
