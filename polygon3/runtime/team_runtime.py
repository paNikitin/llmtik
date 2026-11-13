"""Общий диспетчер C8: три новых процесса, у каждой роли своё состояние, модель
вызывает только родительский процесс.

Публичный интерфейс: run_team_episode(solution_path, episode, model_transport=None,
limits=None, isolation="required") -> dict. Здесь нет импортов кода предметной
области и кода студента. Транспорт обязан делать одну попытку обращения к
провайдеру без скрытых повторов внутри SDK, возвращать явный флаг consumed типа
bool и сам соблюдать срок сетевого запроса. Среда выполнения никогда сама не
повторяет вызовы с неясным исходом. После истечения срока эпизода родительский
процесс перестаёт ждать зависший вызов транспорта. Может остаться один фоновый
поток, который Python не может принудительно остановить.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import math
import os
import queue
import selectors
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import threading
import time
import warnings
from pathlib import Path


def _sibling(name):
    key = "_c8_" + name
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, Path(__file__).with_name(name + ".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


_codec = _sibling("team_sandbox")
_security = _sibling("team_security")


@dataclasses.dataclass(frozen=True)
class RuntimeLimits:
    max_actions: int = 120
    max_model_calls: int = 9
    max_messages: int = 24
    max_message_bytes: int = 4096
    max_total_message_bytes: int = 32768
    max_memory_bytes: int = 131072
    max_frame_bytes: int = 1 << 20
    timeout_s: float = 120.0
    max_rss_mb: float = 384.0
    per_role_rss_mb: float = 128.0
    caller_work_root: str | None = None

    def __post_init__(self):
        for name in ("max_actions", "max_model_calls", "max_messages", "max_message_bytes",
                     "max_total_message_bytes", "max_memory_bytes", "max_frame_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name in ("max_model_calls", "max_messages") else 1):
                raise ValueError("invalid runtime limit: " + name)
        if not 256 <= self.max_frame_bytes <= (16 << 20):
            raise ValueError("max_frame_bytes must be between 256 bytes and 16 MiB")
        for name in ("timeout_s", "max_rss_mb", "per_role_rss_mb"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError("invalid runtime limit: " + name)


class RuntimeFailure(RuntimeError):
    def __init__(self, code):
        super().__init__(code)
        self.code = code


def _copy(value):
    return _codec.decode_json(_codec.json_bytes(value))


def _error(kind, code, **extra):
    return {"status": "error", "type": kind, "code": code, **extra}


def _model_payload(action, frame_limit):
    if set(action) != {"type", "system", "messages", "max_tokens"}:
        raise ValueError("model_fields")
    system, messages, tokens = action["system"], action["messages"], action["max_tokens"]
    if type(system) is not str or not system.strip() or type(tokens) is not int or not 1 <= tokens <= 1024:
        raise ValueError("model_fields")
    if type(messages) is not list or not 1 <= len(messages) <= 32:
        raise ValueError("model_messages")
    for i, message in enumerate(messages):
        if type(message) is not dict or set(message) != {"role", "content"}:
            raise ValueError("model_messages")
        if message["role"] not in ("user", "assistant") or (i == 0 and message["role"] != "user"):
            raise ValueError("model_messages")
        content = message["content"]
        if type(content) is list:
            if not content or len(content) > 64 or any(type(block) is not dict or set(block) != {"type", "text"}
                    or block["type"] != "text" or type(block["text"]) is not str or not block["text"].strip()
                    for block in content):
                raise ValueError("model_messages")
        elif type(content) is not str or not content.strip():
            raise ValueError("model_messages")
    payload = {"system": system, "messages": messages, "max_tokens": tokens}
    # Размер считаем вместе с постоянными полями транспорта и с форматированием
    # json.dumps по умолчанию, которое длиннее компактного.
    body = dict(payload, model="deepseek-flash", thinking={"type": "disabled"})
    body_size = len(json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8"))
    if body_size > min(frame_limit // 2, 65536):
        raise ValueError("model_payload_size")
    return _copy(payload)


_USAGE_KEYS = ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")


def _usage(value):
    if type(value) is not dict:
        return {}
    return {key: value[key] for key in _USAGE_KEYS if type(value.get(key)) is int and 0 <= value[key] <= 10**12}


def _public_model(raw, frame_limit):
    if type(raw) is not dict or type(raw.get("ok")) is not bool or type(raw.get("consumed")) is not bool:
        raise RuntimeFailure("transport_contract_error")
    if raw["ok"] and not raw["consumed"]:
        raise RuntimeFailure("transport_contract_error")
    usage = _usage(raw.get("usage"))
    if raw["ok"]:
        response = raw.get("response")
        if type(response) is not dict or type(response.get("content")) is not list:
            raise RuntimeFailure("transport_contract_error")
        content = []
        for block in response["content"]:
            if type(block) is not dict or block.get("type") != "text" or type(block.get("text")) is not str:
                raise RuntimeFailure("transport_contract_error")
            content.append({"type": "text", "text": block["text"]})
        usage = usage or _usage(response.get("usage"))
        result = {"status": "ok", "type": "model", "response": {"content": content, "usage": usage},
                  "text": "\n".join(block["text"] for block in content)}
    else:
        # Текст исключений провайдера, заголовки, URL, запрос и данные авторизации не копируем.
        error = str(raw.get("error", "model_error"))
        known = {"429": "rate_limited", "rate_limited": "rate_limited", "rate-limited": "rate_limited",
                 "503": "model_unavailable", "model_unavailable": "model_unavailable",
                 "budget_exhausted": "model_budget_exhausted", "model_budget_exhausted": "model_budget_exhausted"}
        code = known.get(error, "model_error")
        result = _error("model", code, consumed=raw["consumed"])
        delay = raw.get("retry_after")
        if type(delay) in (float, int) and math.isfinite(delay) and delay >= 0:
            result["retry_after"] = min(float(delay), 3600.0)
    # Признать сбой инфраструктурным может только явный bool от доверенного транспорта.
    infrastructure = {"infrastructure_error": True} if raw.get("infrastructure_error") is True else {}
    result.update(infrastructure)
    if len(_codec.json_bytes(result)) > frame_limit // 2:
        return _error("model", "model_response_size", consumed=raw["consumed"], **infrastructure), usage
    return result, usage


class _Role:
    def __init__(self, actor, process):
        self.actor, self.process = actor, process
        self.out = bytearray()
        self.stderr_bytes = 0
        self.stdout_bytes = 0
        self.expecting = True  # ждём кадр ready с подтверждением готовности
        self.memory = {}
        self.last_result = None
        self.peak = 0.0
        self.security = {}


class _Supervisor:
    def __init__(self, limits, deadline, required):
        self.limits, self.deadline, self.required = limits, deadline, required
        self.selector = selectors.DefaultSelector()
        self.roles = []
        self.simultaneous_peak = 0.0
        self.samples = 0
        self.next_sample = 0.0

    def add(self, role):
        self.roles.append(role)
        for stream, kind in ((role.process.stdout, "stdout"), (role.process.stderr, "stderr")):
            os.set_blocking(stream.fileno(), False)
            self.selector.register(stream, selectors.EVENT_READ, (role, kind))
        os.set_blocking(role.process.stdin.fileno(), False)

    def check_alive(self):
        # Доверенная программа роли team_sandbox завершается, только когда её
        # закрывает этот _Supervisor. Если роль умерла между своими ходами, это
        # нельзя скрыть действием submit другой роли.
        if any(role.process.poll() is not None for role in self.roles):
            raise RuntimeFailure("role_exited")

    def sample(self):
        now = time.monotonic()
        if now < self.next_sample:
            return
        self.next_sample = now + 0.02
        current = {}
        if sys.platform.startswith("linux"):
            for role in self.roles:
                try:
                    with open(f"/proc/{role.process.pid}/statm", "rb") as stream:
                        rss = int(stream.read(256).split()[1]) * os.sysconf("SC_PAGE_SIZE") / (1 << 20)
                except (OSError, IndexError, ValueError):
                    if role.process.poll() is not None:
                        rss = 0.0
                    else:
                        raise RuntimeFailure("rss_unavailable") from None
                current[role.actor] = rss
        elif self.roles:
            try:
                output = subprocess.run(["ps", "-o", "pid=,rss=", "-p", ",".join(str(r.process.pid) for r in self.roles)],
                                        capture_output=True, timeout=0.5, check=False).stdout.decode("ascii")
                values = {int(line.split()[0]): float(line.split()[1]) / 1024 for line in output.splitlines()}
                current = {r.actor: values.get(r.process.pid, 0.0) for r in self.roles}
            except (OSError, ValueError, subprocess.TimeoutExpired):
                raise RuntimeFailure("rss_unavailable") from None
        for role in self.roles:
            value = current.get(role.actor, 0.0)
            role.peak = max(role.peak, value)
            if value > self.limits.per_role_rss_mb:
                raise RuntimeFailure("role_rss_limit")
        total = sum(current.values())
        self.simultaneous_peak = max(self.simultaneous_peak, total)
        self.samples += 1
        if total > self.limits.max_rss_mb:
            raise RuntimeFailure("team_rss_limit")

    def pump(self, wait=0.02):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeFailure("timeout")
        self.sample()
        for key, _ in self.selector.select(min(wait, remaining, 0.02)):
            role, kind = key.data
            try:
                data = os.read(key.fileobj.fileno(), 65536)
            except BlockingIOError:
                continue
            if not data:
                self.selector.unregister(key.fileobj)
                if kind == "stdout" and not role.expecting:
                    raise RuntimeFailure("role_pipe_closed")
                continue
            if kind == "stderr":
                role.stderr_bytes += len(data)
                # Недоверенный stderr вычитываем, но не храним и никуда не пересылаем.
                if role.stderr_bytes > min(self.limits.max_frame_bytes, 131072):
                    raise RuntimeFailure("stderr_limit")
            else:
                role.stdout_bytes += len(data)
                if not role.expecting:
                    raise RuntimeFailure("unsolicited_frame")
                role.out.extend(data)
                if len(role.out) >= 4:
                    size, = struct.unpack("!I", role.out[:4])
                    if not 0 < size <= self.limits.max_frame_bytes:
                        raise RuntimeFailure("frame_size")
                    if len(role.out) > size + 4:
                        raise RuntimeFailure("extra_frame_bytes")
        self.sample()

    def receive(self, role):
        while True:
            if time.monotonic() >= self.deadline:
                raise RuntimeFailure("timeout")
            if len(role.out) >= 4:
                size, = struct.unpack("!I", role.out[:4])
                if len(role.out) == size + 4:
                    try:
                        message = _codec.decode_json(bytes(role.out[4:]))
                    except _codec.FrameError:
                        raise RuntimeFailure("malformed_frame") from None
                    role.out.clear()
                    role.expecting = False
                    if type(message) is not dict:
                        raise RuntimeFailure("malformed_frame")
                    if message.get("kind") == "error":
                        raise RuntimeFailure("isolation_error" if message.get("code") == "isolation_error" else "role_error")
                    return message
            if role.process.poll() is not None:
                # Прежде чем разбирать EOF, дочитываем последний кадр, если он есть.
                # Его размер ограничен.
                self.pump(0)
                if len(role.out) >= 4 and len(role.out) == struct.unpack("!I", role.out[:4])[0] + 4:
                    continue
                raise RuntimeFailure("role_exited")
            self.pump()

    def request(self, role, value):
        self.check_alive()
        if role.out or role.expecting:
            raise RuntimeFailure("protocol_state")
        try:
            data = _codec.encode_frame(value, self.limits.max_frame_bytes)
        except _codec.FrameError:
            raise RuntimeFailure("observation_frame_size") from None
        role.expecting = True
        offset = 0
        while offset < len(data):
            try:
                written = os.write(role.process.stdin.fileno(), data[offset:offset + 65536])
                if written <= 0:
                    raise RuntimeFailure("role_pipe_closed")
                offset += written
            except BlockingIOError:
                self.pump()
            except BrokenPipeError:
                raise RuntimeFailure("role_exited") from None
            if time.monotonic() >= self.deadline:
                raise RuntimeFailure("timeout")
        message = self.receive(role)
        self.check_alive()
        return message

    def call_transport(self, transport, actor, payload):
        # За весь прогон одновременно идёт не больше одной попытки. Скрытых
        # повторов нет. Если попытку бросили, отчёт учитывает её как
        # израсходованную, с неизвестным исходом.
        box = queue.Queue(maxsize=1)

        def invoke():
            try:
                box.put((True, transport(actor, payload)))
            except BaseException:
                box.put((False, None))

        thread = threading.Thread(target=invoke, name="c8-model-attempt", daemon=True)
        thread.start()
        while True:
            if time.monotonic() >= self.deadline:
                raise RuntimeFailure("timeout")
            try:
                ok, result = box.get_nowait()
                if not ok:
                    raise RuntimeFailure("transport_outcome_unknown")
                return result
            except queue.Empty:
                self.pump()

    def close(self):
        failures = []
        # Linux: убиваем каждую группу процессов, даже если её лидер уже завершился.
        # Режим разработки на macOS не защищает от враждебного кода. Графические
        # приложения в песочнице могут запрещать killpg и при этом разрешать
        # завершить наш собственный Popen.
        for role in self.roles:
            try:
                if not self.required and sys.platform == "darwin":
                    role.process.kill()
                else:
                    os.killpg(role.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError:
                failures.append("killpg")
        for role in self.roles:
            try:
                role.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                failures.append("wait")
            for pipe in (role.process.stdin, role.process.stdout, role.process.stderr):
                pipe.close()
        self.selector.close()
        return failures


def _public_verdict(verdict):
    if type(verdict) is not dict:
        return {}
    out = {}
    for key in ("status", "code"):
        if type(verdict.get(key)) is str and len(verdict[key]) <= 128:
            out[key] = verdict[key]
    if type(verdict.get("correct")) is bool or (type(verdict.get("correct")) is int and verdict["correct"] in (0, 1)):
        out["correct"] = bool(verdict["correct"])
    if type(verdict.get("cost")) is int:
        out["cost"] = verdict["cost"]
    gap = verdict.get("cost_gap")
    if type(gap) is int or (type(gap) is float and math.isfinite(gap)):
        out["cost_gap"] = gap
    # Предметная область возвращает словари. Оставляем только коды; старый
    # формат, где нарушение записано строкой, тоже принимаем. Подробности,
    # идентификаторы и объекты world/oracle/expected/debug никогда не сериализуем.
    if type(verdict.get("violations")) is list:
        codes = []
        for violation in verdict["violations"][:64]:
            code = violation.get("code") if type(violation) is dict else violation
            if type(code) is str and len(code) <= 256:
                codes.append(code)
        out["violations"] = codes
    return out


def _verified_security(value):
    return (type(value) is dict and type(value.get("landlock_abi")) is int
            and value["landlock_abi"] >= 4 and value.get("seccomp") is True
            and value.get("nondumpable") is True and value.get("no_new_privs") is True)


def run_team_episode(solution_path, episode, model_transport=None, limits=None, isolation="required") -> dict:
    limits = limits if isinstance(limits, RuntimeLimits) else RuntimeLimits(**(limits or {}))
    if isolation not in ("required", "development"):
        raise ValueError("isolation must be required or development")
    roles = tuple(episode.roles)
    if len(roles) != 3 or len(set(roles)) != 3 or any(type(r) is not str for r in roles):
        raise ValueError("episode must define exactly three distinct role names")
    started = time.monotonic()
    supervisor = _Supervisor(limits, started + limits.timeout_s, isolation == "required")
    trace, plan = [], None
    stats = {"actions": 0, "logical_model_calls": 0, "model_calls": 0, "model_attempts": 0,
             "model_retries": 0, "transport_unknown_calls": 0, "usage": {}, "messages": 0,
             "message_bytes": 0, "per_role": {r: {"actions": 0, "logical_model_calls": 0,
             "model_calls": 0, "model_attempts": 0, "usage": {}, "messages": 0, "message_bytes": 0} for r in roles}}
    warning = None
    if isolation == "development":
        warning = "DEVELOPMENT: separate processes only; hostile-code isolation is NOT enforced"
        warnings.warn(warning, RuntimeWarning, stacklevel=2)
    status, failure = "incomplete", None
    work = None
    # Канонические идентификаторы корпуса знает только родительский процесс:
    # по порядковому номеру нельзя угадывать скрытую выполнимость задачи.
    # Все роли получают один общий новый псевдоним прогона, по которому
    # ничего нельзя предсказать.
    observation_case_id = "episode-" + os.urandom(16).hex()
    cooldown = 0.0
    attempt_cap = max(1, limits.max_model_calls * 3)
    reserved_attempt = None
    try:
        if isolation == "required":
            _security.require_unprivileged()
            _security.require_nondumpable()  # и для родительского процесса тоже; прежнее состояние намеренно не восстанавливаем
        source = Path(solution_path)
        if not source.is_file() or source.stat().st_size > limits.max_frame_bytes:
            raise RuntimeFailure("solution_size")
        code = source.read_bytes()
        root = Path(limits.caller_work_root) if limits.caller_work_root else Path(__file__).resolve().parent.parent / ".agent-work/polygon3/runtime/work"
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        work = tempfile.TemporaryDirectory(prefix="c8-", dir=root)
        os.chmod(work.name, 0o700)
        config = {k: v for k, v in dataclasses.asdict(limits).items() if k != "caller_work_root"}
        driver = str(Path(__file__).with_name("team_sandbox.py").resolve())
        for index, actor in enumerate(roles):
            directory = Path(work.name) / ("role-" + str(index))
            directory.mkdir(mode=0o700)
            submitted = directory / "agent.py"
            submitted.write_bytes(code)
            submitted.chmod(0o400)
            env = {"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "HOME": str(directory), "TMPDIR": str(directory)}
            process = subprocess.Popen([sys.executable, "-I", "-S", "-B", "-u", driver, str(submitted),
                                        _codec.json_bytes(config).decode("utf-8"), isolation],
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       cwd=directory, env=env, close_fds=True, start_new_session=True, bufsize=0)
            supervisor.add(_Role(actor, process))
        for role in supervisor.roles:
            ready = supervisor.receive(role)
            if ready.get("kind") != "ready" or ready.get("pid") != role.process.pid:
                raise RuntimeFailure("invalid_handshake")
            role.security = ready.get("security", {})
            if isolation == "required" and not _verified_security(role.security):
                raise RuntimeFailure("isolation_error")
        turn = 0
        while not episode.finished:
            if stats["actions"] >= limits.max_actions:
                raise RuntimeFailure("action_limit")
            role = supervisor.roles[turn % 3]
            turn += 1
            budget = {"max_model_calls": limits.max_model_calls, "model_calls": stats["model_calls"],
                      "model_calls_remaining": max(0, limits.max_model_calls - stats["model_calls"]),
                      "role_model_calls": stats["per_role"][role.actor]["model_calls"],
                      "actions_remaining": limits.max_actions - stats["actions"],
                      "messages_remaining": max(0, limits.max_messages - stats["messages"]),
                      "message_bytes_remaining": max(0, limits.max_total_message_bytes - stats["message_bytes"])}
            observation = _copy(episode.observation(role.actor))
            if type(observation) is not dict:
                raise RuntimeFailure("episode_contract_error")
            observation["case_id"] = observation_case_id
            observation["role"] = role.actor
            observation["last_result"] = _copy(role.last_result)
            observation["budget"] = budget
            recorder = getattr(episode, "record_observation", None)
            if recorder is not None:
                recorder(role.actor, _copy(observation))
            returned = supervisor.request(role, {"kind": "act", "observation": observation, "memory": role.memory})
            if returned.get("kind") != "action" or set(returned) != {"kind", "action", "memory"} or type(returned["action"]) is not dict:
                raise RuntimeFailure("malformed_action")
            if len(_codec.json_bytes(returned["memory"])) > limits.max_memory_bytes:
                raise RuntimeFailure("memory_limit")
            role.memory = returned["memory"]
            action = returned["action"]
            kind = action.get("type")
            stats["actions"] += 1
            own = stats["per_role"][role.actor]
            own["actions"] += 1
            epoch = episode.epoch
            if kind == "model":
                stats["logical_model_calls"] += 1
                own["logical_model_calls"] += 1
                try:
                    payload = _model_payload(action, limits.max_frame_bytes)
                except ValueError:
                    payload = None
                if payload is None:
                    result = _error("model", "invalid_model_request", consumed=False)
                elif stats["model_calls"] >= limits.max_model_calls:
                    result = _error("model", "model_budget_exhausted", consumed=False)
                elif stats["model_attempts"] >= attempt_cap:
                    result = _error("model", "model_attempt_limit", consumed=False)
                elif model_transport is None:
                    result = _error("model", "model_transport_unavailable", consumed=False)
                elif time.monotonic() < cooldown:
                    result = _error("model", "rate_limited", consumed=False,
                                    retry_after=round(cooldown - time.monotonic(), 3))
                else:
                    # Резервируем вызов до обращения к произвольному транспорту.
                    # Исключения и истечение срока с неясным исходом никогда
                    # не снимают общий резерв.
                    stats["model_attempts"] += 1
                    own["model_attempts"] += 1
                    stats["model_calls"] += 1
                    own["model_calls"] += 1
                    reserved_attempt = role.actor
                    raw = supervisor.call_transport(model_transport, role.actor, payload)
                    result, usage = _public_model(raw, limits.max_frame_bytes)
                    if not raw["consumed"]:
                        stats["model_calls"] -= 1
                        own["model_calls"] -= 1
                    reserved_attempt = None
                    for key, value in usage.items():
                        stats["usage"][key] = stats["usage"].get(key, 0) + value
                        own["usage"][key] = own["usage"].get(key, 0) + value
                    if "retry_after" in result:
                        cooldown = max(cooldown, time.monotonic() + result["retry_after"])
            elif kind == "send":
                body = action.get("body")
                size = len(_codec.json_bytes(body))
                if type(body) not in (dict, str):
                    result = _error("send", "invalid_message")
                elif size > limits.max_message_bytes:
                    result = _error("send", "message_size")
                elif stats["messages"] >= limits.max_messages or stats["message_bytes"] + size > limits.max_total_message_bytes:
                    result = _error("send", "message_budget_exhausted")
                else:
                    result = _copy(episode.apply(role.actor, _copy(action)))
                    if result.get("status") == "ok":
                        stats["messages"] += 1
                        stats["message_bytes"] += size
                        own["messages"] += 1
                        own["message_bytes"] += size
            else:
                result = _copy(episode.apply(role.actor, _copy(action)))
            if type(result) is not dict:
                raise RuntimeFailure("episode_contract_error")
            # Результат действия получает только роль, которая его сделала.
            # Общей стенограммы для всех ролей нет.
            role.last_result = result
            episode.advance()  # и после вызова модели, и после отклонённого действия; ход часов определяет предметная область
            trace.append({"step": stats["actions"], "actor": role.actor, "action": _copy(action),
                          "result": _copy(result), "epoch": epoch})
            if episode.finished and role.actor == "coordinator" and kind == "submit" and result.get("status") == "ok":
                plan = _copy(action.get("plan"))
        if time.monotonic() >= supervisor.deadline:
            raise RuntimeFailure("timeout")
        status = "finished"
    except RuntimeFailure as exc:
        failure = exc.code
        status = "isolation_error" if failure == "isolation_error" else "runtime_error"
        if reserved_attempt is not None:
            stats["transport_unknown_calls"] += 1
    except _security.SecurityUnavailable:
        failure, status = "isolation_unavailable", "isolation_error"
    except (OSError, _codec.FrameError, ValueError, TypeError, AttributeError, KeyError):
        failure, status = "runtime_contract_error", "runtime_error"
    finally:
        cleanup = supervisor.close()
        if work is not None:
            try:
                work.cleanup()
            except OSError:
                cleanup.append("workdir")
        if cleanup:
            failure, status = "cleanup_failed", "runtime_error"
    verdict = _public_verdict(episode.verdict) if status == "finished" else {}
    stats["seconds"] = round(time.monotonic() - started, 6)
    stats["simultaneous_rss_mb"] = round(supervisor.simultaneous_peak, 3)
    stats["rss_per_role_mb"] = {r.actor: round(r.peak, 3) for r in supervisor.roles}
    stats["rss_samples"] = supervisor.samples
    stats["rss_measurement"] = "sampled-current-rss-sum; not sum of historical per-role peaks"
    for role in supervisor.roles:
        stats["per_role"][role.actor].update({"pid": role.process.pid, "rss_mb": round(role.peak, 3),
            "stdout_bytes": role.stdout_bytes, "stderr_bytes": role.stderr_bytes,
            "exit_code": role.process.returncode})
    report = {"case_id": str(episode.case_id), "status": status, "correct": verdict.get("correct", False),
              "verdict": verdict, "plan": plan, "trace": trace, "stats": stats,
              "isolation": {"mode": isolation, "verified": isolation == "required" and len(supervisor.roles) == 3
                  and all(_verified_security(r.security) for r in supervisor.roles),
                  "roles": {r.actor: r.security for r in supervisor.roles}},
              "cleanup": {"ok": not cleanup, "errors": cleanup}}
    if failure:
        report["error"] = failure
    if warning:
        report["isolation"]["warning"] = warning
    return report
