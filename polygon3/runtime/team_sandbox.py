"""Переносимая программа, которая запускает роль C8 в отдельном процессе, и общие
функции кодирования JSON-кадров с префиксом длины. Используется только стандартная библиотека.

Формат кадра: длина в байтах как беззнаковое 32-битное число со старшим байтом
впереди, затем ровно столько байтов JSON в UTF-8. В каждый момент
ждать ответа может не больше одного запроса. print() студента идёт в stderr.
Прямая запись в дескриптор 1 попадает в протокол и считается недоверенными данными.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import struct
import sys
from pathlib import Path


class FrameError(ValueError):
    pass


def _object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise FrameError("duplicate JSON key")
        obj[key] = value
    return obj


def _validate_json(value):
    stack = [(value, 0)]
    while stack:
        item, depth = stack.pop()
        if depth > 64:
            raise FrameError("JSON nesting limit")
        if item is None or type(item) in (bool, int):
            continue
        if type(item) is str:
            item.encode("utf-8")  # отклоняет и одиночные суррогаты, пришедшие экранированными при декодировании
            continue
        if type(item) is float:
            if not math.isfinite(item):
                raise FrameError("non-finite JSON number")
        elif type(item) is list:
            stack.extend((child, depth + 1) for child in item)
        elif type(item) is dict:
            if any(type(key) is not str for key in item):
                raise FrameError("non-string JSON key")
            for key in item:
                key.encode("utf-8")
            stack.extend((child, depth + 1) for child in item.values())
        else:
            raise FrameError("non-JSON value")


def json_bytes(value) -> bytes:
    try:
        _validate_json(value)
        return json.dumps(value, ensure_ascii=False, allow_nan=False,
                          separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError, RecursionError, UnicodeError) as exc:
        raise FrameError("not finite UTF-8 JSON") from exc


def decode_json(data: bytes):
    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=_object,
                           parse_constant=lambda _: (_ for _ in ()).throw(FrameError("non-finite JSON")))
        _validate_json(value)  # Значение 1e999 в JSON становится бесконечностью в обход parse_constant.
        return value
    except (ValueError, RecursionError, UnicodeError) as exc:
        raise FrameError("invalid UTF-8 JSON") from exc


def encode_frame(value, limit: int) -> bytes:
    payload = json_bytes(value)
    if not payload or len(payload) > limit:
        raise FrameError("frame size exceeded")
    return struct.pack("!I", len(payload)) + payload


def _read_exact(src, size):
    buf = bytearray()
    while len(buf) < size:
        chunk = src.read(size - len(buf))
        if not chunk:
            if not buf:
                return None
            raise FrameError("truncated frame")
        buf.extend(chunk)
    return bytes(buf)


def read_frame(src, limit: int):
    header = _read_exact(src, 4)
    if header is None:
        return None
    size, = struct.unpack("!I", header)
    if not 0 < size <= limit:
        raise FrameError("invalid frame length")
    payload = _read_exact(src, size)
    if payload is None:
        raise FrameError("truncated frame")
    return decode_json(payload)


def load_sibling(name: str):
    """Работает при импорте как app.*, как polygon3.runtime.* и при запуске по абсолютному пути."""
    qualified = "_c8_" + name
    if qualified in sys.modules:
        return sys.modules[qualified]
    spec = importlib.util.spec_from_file_location(qualified, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified] = module
    spec.loader.exec_module(module)
    return module


def _limits(config):
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CPU, (math.ceil(config["timeout_s"]) + 1,) * 2)
    resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20,) * 2)
    # RSS измеряет доверенный родительский процесс. RLIMIT_AS в Linux добавляет
    # ещё одну границу: на ней процесс аварийно завершится.
    # Darwin отклоняет малые значения RLIMIT_AS даже для только что запущенного
    # интерпретатора. Режим разработки на macOS явно небезопасен, но в нём
    # всё равно действуют ограничения RSS, CPU и сроков.
    if sys.platform.startswith("linux"):
        as_bytes = int(max(64, config["per_role_rss_mb"] * 2) * (1 << 20))
        resource.setrlimit(resource.RLIMIT_AS, (as_bytes,) * 2)
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))


def _load_act(path):
    spec = importlib.util.spec_from_file_location("student_solution", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["student_solution"] = module
    spec.loader.exec_module(module)
    act = getattr(module, "act", None)
    if not callable(act):
        raise TypeError("missing act")
    return act


def main() -> int:
    # Забираем потоки протокола до того, как подменить обычный stdout Python.
    source, wire = sys.stdin.buffer, sys.stdout.buffer
    sys.stdout = sys.stderr
    config = decode_json(sys.argv[2].encode("utf-8"))
    limit = config["max_frame_bytes"]

    def send(value):
        wire.write(encode_frame(value, limit))
        wire.flush()

    try:
        security = load_sibling("team_security")  # до любых ограничений доступа к файлам
        _limits(config)
        mode = sys.argv[3]
        if mode == "required":
            # Инициализация OpenSSL вызывает futex. Фильтр роли запрещает этот вызов,
            # и glibc аварийно завершает процесс. Без модуля _hashlib модуль hashlib
            # берёт встроенные в интерпретатор md5/sha1/sha2/sha3/blake2, хеши те же.
            sys.modules["_hashlib"] = None
            isolation = security.confine_role(str(Path(sys.argv[1]).parent))
        elif mode == "development":
            isolation = {"warning": "DEVELOPMENT: no hostile-code isolation", "seccomp": False,
                         "landlock_abi": 0, "nondumpable": False}
        else:
            raise ValueError("invalid isolation mode")
        send({"kind": "ready", "pid": os.getpid(), "security": isolation})
        act = None
        while True:
            message = read_frame(source, limit)
            if message is None:
                return 0
            if type(message) is not dict or message.get("kind") not in ("act", "end"):
                raise FrameError("unexpected request")
            if message["kind"] == "end":
                return 0
            if act is None:
                act = _load_act(sys.argv[1])
            result = act(message["observation"], message["memory"])
            if type(result) not in (tuple, list) or len(result) != 2 or type(result[0]) is not dict:
                raise FrameError("act must return (action, memory)")
            action, memory = result
            if len(json_bytes(memory)) > config["max_memory_bytes"]:
                raise FrameError("memory size exceeded")
            send({"kind": "action", "action": action, "memory": memory})
    except BaseException as exc:
        # Текст исключений и трассировку стека не пересылаем: в них могут быть
        # недоверенные байты или пути.
        try:
            send({"kind": "error", "code": "isolation_error" if isinstance(
                exc, security.SecurityUnavailable) else "role_error", "exception": type(exc).__name__})
        except BaseException:
            pass  # Если не ушёл даже короткий кадр ошибки, родитель увидит EOF или ненулевой код выхода.
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
