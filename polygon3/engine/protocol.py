"""Диспетчер предметной среды в доверенном родительском процессе.

Модуль не запускает подпроцессы, не обращается к сети и к модели и не
выполняет код студентов.

Среда выполнения берёт роль actor из канала дочернего процесса,
подлинность которого проверена. Для действий задачи она вызывает apply, а затем
ровно один раз вызывает advance после *каждого* действия, включая
перехваченные вызовы модели и отклонённые команды. Сам apply не вызывает
advance. Каждая роль видит только свой список документов, свои входящие
сообщения и свои уведомления. Память, last_result и бюджет хранит среда
выполнения, и в наблюдения, которые строит этот модуль, они не попадают. При
отправке плана с устаревшим значением epoch и при итоговой отправке
скрытые сведения о нарушенных ограничениях не возвращаются.
"""
from __future__ import annotations
from copy import deepcopy
from typing import Any

from .schema import (Case, CaseError, ROLES, MAX_ACTION_BYTES, fields, integer,
                     json_size, strict_json_loads)
from .checks import evaluate_plan


class TeamEpisode:
    roles = ROLES

    def __init__(self, case: Case, limits: Any = None):
        if not isinstance(case, Case):
            raise TypeError("TeamEpisode requires a trusted Case")
        self.case_id = case.case_id
        self.epoch = 1
        self.finished = False
        self.verdict: dict = {}
        self.events: list[dict] = []
        self.plan: Any = None
        self._world = deepcopy(case.world)
        self._documents = {role: {doc["id"]: deepcopy(doc) for doc in case.documents[role]} for role in ROLES}
        self._updates = deepcopy(list(case.events))
        self._inboxes = {role: [] for role in ROLES}
        self._notices = {role: [] for role in ROLES}
        self._step = 0
        self._next_message_id = 1
        self._message_count = 0
        self._message_bytes = 0
        defaults = {"max_messages": 64, "max_message_bytes": 4096, "max_total_message_bytes": 65536}
        self._limits = {}
        for name, default in defaults.items():
            value = limits.get(name, default) if isinstance(limits, dict) else getattr(limits, name, default)
            self._limits[name] = integer(value, maximum=10_000_000, where=name)

    def debug_case(self) -> Case:
        """Текущий снимок задачи только для доверенного родительского процесса.

        Снимок нужен для локальных опытов по заранее написанному сценарию.
        Метод возвращает защитные копии и не возвращает ожидающие события,
        поэтому загрузка снимка не может повторно применить обновления.
        Транспорт scripted использует снимок для подстановки готовых фактов.
        """
        documents = {role: list(deepcopy(self._documents[role]).values()) for role in ROLES}
        return Case(self.case_id, deepcopy(self._world), documents, (), {})

    def observation(self, actor: str) -> dict:
        if type(actor) is not str or actor not in ROLES:
            raise ValueError("unknown actor")
        return {"case_id": self.case_id, "role": actor, "epoch": self.epoch,
                "documents": [{"id": doc["id"], "revision": doc["revision"]}
                              for doc in self._documents[actor].values()],
                "messages": deepcopy(self._inboxes[actor]),
                "notices": deepcopy(self._notices[actor])}

    def _result(self, actor: str, kind: str, *, code: str | None = None, **payload: Any) -> dict:
        result = {"status": "error" if code else "ok", "type": kind}
        if code:
            result["code"] = code
        result.update(payload)
        # Здесь только служебные данные для аудита. Сюда не попадают тексты
        # сообщений, тексты документов, скрытые факты, история обращений к
        # модели, учётные данные и оптимальные планы. Открытых ID достаточно,
        # чтобы сопоставить запись с журналом действий родительского процесса,
        # доступ к которому выдаётся отдельно.
        event = {"step": self._step, "actor": actor, "type": kind,
                 "status": result["status"], "epoch": self.epoch}
        if code:
            event["code"] = code
        if "message_id" in payload:
            event["message_id"] = payload["message_id"]
        self.events.append(event)
        return result

    def apply(self, actor: str, action: Any) -> dict:
        if type(actor) is not str or actor not in ROLES:
            return {"status": "error", "type": "unknown", "code": "unknown_actor"}
        if self.finished:
            return self._result(actor, "unknown", code="episode_finished")
        try:
            if type(action) in (str, bytes):
                action = strict_json_loads(action, limit=MAX_ACTION_BYTES)
            json_size(action, limit=MAX_ACTION_BYTES)
            if type(action) is not dict or type(action.get("type")) is not str:
                raise CaseError("action_object_required")
            kind = action["type"]
            allowed = {"read": {"type", "doc_id"}, "send": {"type", "to", "body"},
                       "wait": {"type"}, "submit": {"type", "epoch", "plan"},
                       "stop": {"type", "reason"}}
            if kind == "model":
                return self._result(actor, kind, code="runtime_only")
            if kind not in allowed:
                return self._result(actor, "unknown", code="unknown_action")
            fields(action, allowed[kind], where="action")
        except CaseError:
            # Не повторяем в ошибке присланные имена полей, тело сообщения и очень
            # большие значения.
            return self._result(actor, "unknown", code="invalid_action")
        if kind == "read":
            doc_id = action["doc_id"]
            if type(doc_id) is not str or doc_id not in self._documents[actor]:
                # Для несуществующего и для чужого ID код одинаковый, чтобы не
                # выдать, существует ли документ.
                return self._result(actor, kind, code="forbidden_document")
            return self._result(actor, kind, document=deepcopy(self._documents[actor][doc_id]))
        if kind == "send":
            recipient, body = action["to"], action["body"]
            if type(recipient) is not str or recipient not in ROLES:
                return self._result(actor, kind, code="invalid_recipient")
            if type(body) not in (dict, str):
                return self._result(actor, kind, code="invalid_message_body")
            try:
                size = json_size(body, limit=self._limits["max_message_bytes"])
            except CaseError:
                return self._result(actor, kind, code="message_too_large")
            if self._message_count >= self._limits["max_messages"]:
                return self._result(actor, kind, code="message_limit")
            if self._message_bytes + size > self._limits["max_total_message_bytes"]:
                return self._result(actor, kind, code="total_message_bytes")
            message_id = self._next_message_id
            self._next_message_id += 1
            self._message_count += 1
            self._message_bytes += size
            self._inboxes[recipient].append({"id": message_id, "from": actor, "to": recipient,
                                           "body": deepcopy(body), "epoch": self.epoch})
            return self._result(actor, kind, message_id=message_id)
        if kind == "wait":
            return self._result(actor, kind)
        if actor != "coordinator":
            return self._result(actor, kind, code="coordinator_only")
        if kind == "stop":
            reason = action["reason"]
            if type(reason) is not str or len(reason.encode("utf-8")) > 1024:
                return self._result(actor, kind, code="invalid_reason")
            self.finished = True
            self.verdict = {"correct": 0, "status": "incomplete", "violations": [],
                            "cost": None, "optimal_cost": None, "cost_gap": None, "is_optimal": False}
            return self._result(actor, kind)
        try:
            integer(action["epoch"], minimum=1, where="epoch")
        except CaseError:
            return self._result(actor, kind, code="invalid_epoch")
        if action["epoch"] != self.epoch:
            return self._result(actor, kind, code="stale_epoch")
        # Принятый план завершает эпизод, даже если он неправильный или ложный.
        # Повторно запросить проверку ограничений нельзя, и advance не меняет
        # итоговый результат.
        self.plan = deepcopy(action["plan"])
        self.verdict = evaluate_plan(self._world, self.plan)
        self.finished = True
        return self._result(actor, kind)

    def advance(self) -> None:
        if self.finished:
            return
        self._step += 1
        while self._updates and self._updates[0]["after_step"] <= self._step:
            event = self._updates.pop(0)
            owner, document = event["owner"], event["document"]
            self._world.update(deepcopy(event["world_patch"]))
            self._documents[owner][document["id"]] = deepcopy(document)
            self.epoch += 1
            notice = {"type": "document_updated", "doc_id": document["id"],
                      "revision": document["revision"], "epoch": self.epoch}
            self._notices[owner].append(notice)
            self.events.append({"step": self._step, "type": "document_updated", "actor": owner,
                                "doc_id": document["id"], "revision": document["revision"], "epoch": self.epoch})
