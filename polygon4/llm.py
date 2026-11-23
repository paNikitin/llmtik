"""Транспорт модели для Полигона 4.

ScriptedTransport читает только буквальные stated_claims и не раскрывает truth.
LiveTransport ходит в Messages API шлюза и на 429 ждёт Retry-After, не сбрасывая прогон.
Токен не попадает ни в repr, ни в текст исключений.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit
from typing import Any, Callable

from .spec import DEFAULT_LLM_CALLS_PER_EPISODE, DocumentSpec, ScenarioSpec

_ACTIVE: Any = None
_DEFAULT_URL = "https://lectures.ai.dscs.pro"


class BudgetExceededError(RuntimeError):
    """Лимит вызовов модели на эпизод исчерпан."""


class TransportError(RuntimeError):
    """Сбой транспорта. В тексте нет секретов и тела ответа."""


class CassetteMissError(LookupError):
    """В кассете нет ответа на этот запрос."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def set_active_transport(transport: Any) -> None:
    global _ACTIVE
    _ACTIVE = transport


def get_active_transport() -> Any:
    return _ACTIVE


def ask_model(
    prompt: str,
    *,
    system: str = "",
    max_tokens: int = 500,
    temperature: float = 0.0,
) -> str:
    transport = get_active_transport()
    if transport is None:
        raise RuntimeError("активный транспорт не выбран")
    complete = getattr(transport, "complete", None)
    if complete is None:
        return transport(prompt, system=system, max_tokens=max_tokens, temperature=temperature)
    return complete(prompt, system=system, max_tokens=max_tokens, temperature=temperature)


def request_sha256(
    prompt: str,
    *,
    system: str = "",
    max_tokens: int = 500,
    temperature: float = 0.0,
) -> str:
    payload = json.dumps(
        {
            "max_tokens": max_tokens,
            "prompt": prompt,
            "system": system,
            "temperature": temperature,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def clamp_retry_after(value: Any) -> float:
    """Локальное и стендовое ожидание: не короче секунды и не длиннее пятнадцати."""
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        seconds = 1.0
    if seconds < 1:
        seconds = 1.0
    if seconds > 15:
        seconds = 15.0
    return seconds


class ScriptedTransport:
    """Учебный чтец писем: возвращает stated_claims упомянутых документов и ничего больше."""

    def __init__(self, scenario: Any, max_calls: int | None = None) -> None:
        self._docs = tuple(_coerce_documents(scenario))
        self.max_calls = DEFAULT_LLM_CALLS_PER_EPISODE if max_calls is None else int(max_calls)
        if self.max_calls < 0:
            raise ValueError("лимит вызовов не может быть отрицательным")
        self.calls = 0

    def complete(
        self,
        prompt: str,
        *,
        system: str = "",
        max_tokens: int = 500,
        temperature: float = 0.0,
    ) -> str:
        del system, max_tokens, temperature
        self._spend()
        matched = _mentioned(self._docs, prompt or "")
        payload = [{"doc_id": doc.doc_id, "stated_claims": doc.stated_claims} for doc in matched]
        encoded = json.dumps(payload, ensure_ascii=False)
        if _leaks_hidden(encoded):
            raise RuntimeError("ответ учебного транспорта содержит скрытые поля")
        return encoded

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        return self.complete(prompt, **kwargs)

    def _spend(self) -> None:
        if self.calls >= self.max_calls:
            raise BudgetExceededError(
                f"исчерпан расход вызовов модели на эпизод ({self.max_calls})"
            )
        self.calls += 1


class CassetteTransport:
    """Запись и воспроизведение ответов по SHA-256 полного запроса."""

    def __init__(self, path: str | Path, source: Any = None) -> None:
        self.path = Path(path)
        self.source = source
        self.records = _load_cassette(self.path)

    def complete(
        self,
        prompt: str,
        *,
        system: str = "",
        max_tokens: int = 500,
        temperature: float = 0.0,
    ) -> str:
        key = request_sha256(prompt, system=system, max_tokens=max_tokens, temperature=temperature)
        cached = self.records.get(key)
        if isinstance(cached, str):
            return cached
        if self.source is None:
            raise CassetteMissError("в кассете нет ответа на этот запрос")
        text = _invoke(self.source, prompt, system=system, max_tokens=max_tokens, temperature=temperature)
        if not isinstance(text, str):
            raise TypeError("источник кассеты должен вернуть строку")
        self.records[key] = text
        _save_cassette(self.path, self.records)
        return text

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        return self.complete(prompt, **kwargs)


class LiveTransport:
    """POST {base}/v1/messages. На 429 ждёт и повторяет запрос ещё до двух раз."""

    def __init__(
        self,
        base_url: str | None = None,
        token: str | None = None,
        max_calls: int | None = None,
        timeout: float = 30.0,
    ) -> None:
        raw_url = (base_url or "").strip() or _env("POLYGON_URL") or _env("LLM_BASE_URL") or _DEFAULT_URL
        self.base_url = raw_url.rstrip("/")
        _validate_base_url(self.base_url)
        if token is None:
            self._token = _env("POLYGON_TOKEN") or _env("LLM_API_KEY")
        else:
            self._token = token.strip() if isinstance(token, str) else ""
        self.max_calls = DEFAULT_LLM_CALLS_PER_EPISODE if max_calls is None else int(max_calls)
        if self.max_calls < 0:
            raise ValueError("лимит вызовов не может быть отрицательным")
        self.timeout = float(timeout)
        self.calls = 0
        self.infrastructure_failed = False
        self.last_status: int | None = None
        self._opener = urllib.request.build_opener(_NoRedirect)

    def __repr__(self) -> str:
        return (
            f"LiveTransport(base_url={self.base_url!r}, max_calls={self.max_calls!r}, "
            f"timeout={self.timeout!r})"
        )

    def complete(
        self,
        prompt: str,
        *,
        system: str = "",
        max_tokens: int = 500,
        temperature: float = 0.0,
    ) -> str:
        if self.calls >= self.max_calls:
            raise BudgetExceededError(
                f"исчерпан расход вызовов модели на эпизод ({self.max_calls})"
            )
        if not self._token:
            raise TransportError("не задан токен доступа к модели")
        self.calls += 1
        payload: dict[str, Any] = {
            "model": "deepseek-flash",
            "max_tokens": int(max_tokens),
            "temperature": temperature,
            "thinking": {"type": "disabled"},
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            payload["system"] = system
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        url = self.base_url + "/v1/messages"
        status = 0
        raw = b""
        for attempt in range(1, 4):
            request = urllib.request.Request(
                url,
                data=data,
                headers={
                    "content-type": "application/json",
                    "x-api-key": self._token,
                    "anthropic-version": "2023-06-01",
                },
                method="POST",
            )
            status, headers, raw = self._exchange(request)
            self.last_status = status
            if status == 429 and attempt <= 2:
                time.sleep(clamp_retry_after(_header(headers, "Retry-After")))
                continue
            break
        if status == 200:
            return _text_from_message(raw)
        if status == 429 or status in (408, 425) or status >= 500:
            self.infrastructure_failed = True
        raise TransportError(f"поставщик модели ответил статусом {status}")

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        return self.complete(prompt, **kwargs)

    def _http_open(self, request: urllib.request.Request) -> Any:
        return self._opener.open(request, timeout=self.timeout)

    def _exchange(self, request: urllib.request.Request) -> tuple[int, Any, bytes]:
        try:
            with self._http_open(request) as response:
                raw = response.read()
                status = getattr(response, "status", 200)
                headers = getattr(response, "headers", {})
                return int(status), headers, raw
        except urllib.error.HTTPError as exc:
            try:
                raw = exc.read()
            except Exception:
                raw = b""
            return int(exc.code), exc.headers, raw
        except (urllib.error.URLError, TimeoutError, OSError):
            self.infrastructure_failed = True
            raise TransportError("не удалось связаться с поставщиком модели") from None


def _env(name: str) -> str:
    return (os.environ.get(name) or "").strip()


def _validate_base_url(base_url: str) -> None:
    parts = urlsplit(base_url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError("неподходящий адрес модели")
    if parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError("адрес модели не должен содержать секрет или лишние части")


def _header(headers: Any, name: str) -> Any:
    if headers is None:
        return None
    getter = getattr(headers, "get", None)
    if getter is None:
        return None
    try:
        return getter(name)
    except Exception:
        return None


def _text_from_message(raw: bytes) -> str:
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError, TypeError):
        raise TransportError("неожиданный ответ поставщика модели") from None
    if isinstance(parsed, dict) and isinstance(parsed.get("content"), str):
        return parsed["content"]
    blocks = parsed.get("content") if isinstance(parsed, dict) else None
    if not isinstance(blocks, list):
        raise TransportError("неожиданный ответ поставщика модели")
    parts: list[str] = []
    for block in blocks:
        if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"])
    if not parts:
        raise TransportError("неожиданный ответ поставщика модели")
    return "".join(parts)


def _coerce_documents(source: Any) -> list[DocumentSpec]:
    if isinstance(source, ScenarioSpec):
        return list(source.documents)
    if isinstance(source, DocumentSpec):
        return [source]
    if isinstance(source, (list, tuple)):
        documents: list[DocumentSpec] = []
        for item in source:
            documents.extend(_coerce_documents(item))
        return documents
    raise TypeError("нужен сценарий или список документов")


def _mentioned(documents: tuple[DocumentSpec, ...] | list[DocumentSpec], prompt: str) -> list[DocumentSpec]:
    bodies = [doc.body for doc in documents]
    found: list[DocumentSpec] = []
    for doc in documents:
        if doc.doc_id and doc.doc_id in prompt:
            found.append(doc)
            continue
        if doc.body and doc.body in prompt:
            found.append(doc)
            continue
        if _unique_fragment(doc.body, prompt, bodies):
            found.append(doc)
    return found


def _unique_fragment(body: str, prompt: str, bodies: list[str]) -> bool:
    if not body or len(body) < 48:
        return False
    windows = [body[start:start + 48] for start in range(0, len(body) - 47, 24)]
    windows.append(body[-48:])
    for fragment in windows:
        if fragment in prompt and sum(fragment in other for other in bodies) == 1:
            return True
    return False


def _leaks_hidden(encoded: str) -> bool:
    try:
        parsed = json.loads(encoded)
    except json.JSONDecodeError:
        return True
    if not isinstance(parsed, list):
        return True
    for item in parsed:
        if not isinstance(item, dict):
            return True
        if set(item) - {"doc_id", "stated_claims"}:
            return True
        claims = item.get("stated_claims")
        if not isinstance(claims, dict):
            return True
        if "oracle_marker" in claims or "verdict" in claims or "truth" in claims:
            return True
    return False


def _invoke(source: Any, prompt: str, **kwargs: Any) -> Any:
    complete = getattr(source, "complete", None)
    if complete is not None:
        return complete(prompt, **kwargs)
    return source(prompt, **kwargs)


def _load_cassette(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not all(isinstance(key, str) and isinstance(value, str) for key, value in raw.items()):
        raise ValueError("кассета должна быть объектом строк")
    return dict(raw)


def _save_cassette(path: Path, records: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(records, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
