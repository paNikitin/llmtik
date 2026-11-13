"""Доверенная схема входных данных Полигона 3 с ограниченными размерами.

Время задаётся целым числом минут от 09:00 одного общего дня, деньги в
копейках, вес в граммах, количество в неделимых штуках. В заказе от 1 до 3
строк, предложений от 0 до 12, прямых рейсов от 0 до 12.

Загрузить Case может только доверенный родительский процесс. Наблюдения ролей
никогда не содержат мир, метаданные, изменения мира из событий, зёрна
случайного генератора и ответы.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any

ROLES = ("coordinator", "supply", "logistics")
MAX_CASE_BYTES = 1_048_576
MAX_PLAN_BYTES = 16_384
MAX_ACTION_BYTES = 65_536
MAX_DOCUMENT_BYTES = 32_768
MAX_ITEMS = 12
MAX_INT = 1_000_000_000
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")


class CaseError(ValueError):
    """Неправильные канонические данные."""


def json_size(value: Any, *, limit: int, max_depth: int = 32) -> int:
    """Проверяет типы JSON, глубину вложенности и конечность чисел до сериализации, включая ключи.

    Функция пропускает конечные числа float из JSON. Они допустимы только в
    телах сообщений и в метаданных. Типизированные поля задачи проверяются
    отдельно. Для них используется точная проверка ``type(x) is int`` вместо isinstance.
    """
    active: set[int] = set()

    def visit(item: Any, depth: int) -> None:
        if depth > max_depth:
            raise CaseError("json_too_deep")
        if item is None or type(item) in (bool, int):
            if type(item) is int and item.bit_length() > 64:
                raise CaseError("json_integer_too_large")
            return
        if type(item) is float:
            if not math.isfinite(item):
                raise CaseError("nonfinite_number")
            return
        if type(item) is str:
            if len(item) > limit:
                raise CaseError("json_too_large")
            return
        if type(item) not in (dict, list):
            raise CaseError("not_json")
        if len(item) > limit:
            raise CaseError("json_too_large")
        identity = id(item)
        if identity in active:
            raise CaseError("cyclic_json")
        active.add(identity)
        if type(item) is dict:
            for key, child in item.items():
                if type(key) is not str:
                    raise CaseError("nonstring_key")
                visit(key, depth + 1)
                visit(child, depth + 1)
        else:
            for child in item:
                visit(child, depth + 1)
        active.remove(identity)

    visit(value, 0)
    try:
        size = len(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8"))
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise CaseError("invalid_json") from exc
    if size > limit:
        raise CaseError("json_too_large")
    return size


def strict_json_loads(text: str | bytes, *, limit: int = MAX_CASE_BYTES) -> Any:
    """Отклоняет повторяющиеся ключи, NaN и Infinity, неправильный UTF-8 и слишком большой JSON.

    После того как JSON превращён в dict, повторяющиеся ключи уже не найти.
    Поэтому код, который получает JSON как текст из канала, должен вызывать
    эту функцию до такого преобразования.
    """
    if type(text) not in (str, bytes):
        raise CaseError("json_text_required")
    try:
        raw = text.encode("utf-8") if type(text) is str else text
        if len(raw) > limit:
            raise CaseError("json_too_large")
        decoded = raw.decode("utf-8")
    except UnicodeError as exc:
        raise CaseError("invalid_utf8") from exc

    def pairs(items: list[tuple[str, Any]]) -> dict:
        result: dict = {}
        for key, value in items:
            if key in result:
                raise CaseError("duplicate_key")
            result[key] = value
        return result

    def constant(_: str) -> Any:
        raise CaseError("nonfinite_number")

    try:
        value = json.loads(decoded, object_pairs_hook=pairs, parse_constant=constant)
    except (ValueError, RecursionError) as exc:
        if isinstance(exc, CaseError):
            raise
        raise CaseError("invalid_json") from exc
    json_size(value, limit=limit)
    return value


def fields(value: Any, required: set[str], optional: set[str] | None = None, *, where: str = "object") -> dict:
    if type(value) is not dict:
        raise CaseError(f"{where}:object_required")
    if set(value) - required - (optional or set()):
        raise CaseError(f"{where}:unknown_fields")
    if required - set(value):
        raise CaseError(f"{where}:missing_fields")
    return value


def integer(value: Any, *, minimum: int = 0, maximum: int = MAX_INT, where: str = "integer") -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise CaseError(f"{where}:integer_out_of_range")
    return value


def identifier(value: Any, *, where: str = "id") -> str:
    if type(value) is not str or not _ID.fullmatch(value):
        raise CaseError(f"{where}:invalid_id")
    return value


def string_list(value: Any, *, where: str) -> list[str]:
    if type(value) is not list or len(value) > 24:
        raise CaseError(f"{where}:list_required_or_too_long")
    for item in value:
        identifier(item, where=where)
    if len(set(value)) != len(value):
        raise CaseError(f"{where}:duplicates")
    return value


def object_list(value: Any, *, where: str, maximum: int = MAX_ITEMS) -> list[dict]:
    if type(value) is not list or len(value) > maximum:
        raise CaseError(f"{where}:list_required_or_too_long")
    if any(type(item) is not dict for item in value):
        raise CaseError(f"{where}:object_required")
    return value


def unique_ids(items: list[dict], *, where: str) -> None:
    ids = [identifier(item.get("id"), where=where) for item in items]
    if len(ids) != len(set(ids)):
        raise CaseError(f"{where}:duplicate_id")


def validate_world(data: Any) -> dict:
    """Проверяет структуру, а не выполнимость. Пустые списки предложений и рейсов допустимы.

    Категории груза из заявки и из предложения действуют вместе, то есть
    берётся их объединение. Обязательные свойства проверяются и для
    запрошенного SKU, а не только для замен. Список allowed_skus добавляет
    замены к исходному sku.
    """
    fields(data, {"request", "offers", "trips"}, where="world")
    request = fields(data["request"], {"destination", "deadline_min", "budget_kopecks", "lines"}, where="request")
    identifier(request["destination"], where="destination")
    integer(request["deadline_min"], where="deadline_min")
    integer(request["budget_kopecks"], where="budget_kopecks")
    lines = object_list(request["lines"], where="lines", maximum=3)
    if not lines:
        raise CaseError("lines:one_to_three_required")
    for line in lines:
        fields(line, {"id", "sku", "quantity", "allowed_skus", "required_properties", "cargo"}, where="line")
        identifier(line["sku"], where="sku")
        integer(line["quantity"], minimum=1, where="quantity")
        for key in ("allowed_skus", "required_properties", "cargo"):
            string_list(line[key], where=key)
    unique_ids(lines, where="lines")
    offers = object_list(data["offers"], where="offers")
    for offer in offers:
        fields(offer, {"id", "sku", "stock", "min_qty", "pack_qty", "unit_price_kopecks", "fee_kopecks", "source", "ready_min", "unit_weight_g", "properties", "cargo"}, where="offer")
        for key in ("sku", "source"):
            identifier(offer[key], where=key)
        for key in ("stock", "unit_price_kopecks", "fee_kopecks", "ready_min"):
            integer(offer[key], where=key)
        for key in ("min_qty", "pack_qty", "unit_weight_g"):
            integer(offer[key], minimum=1, where=key)
        string_list(offer["properties"], where="properties")
        string_list(offer["cargo"], where="cargo")
    unique_ids(offers, where="offers")
    trips = object_list(data["trips"], where="trips")
    for trip in trips:
        fields(trip, {"id", "source", "destination", "departure_min", "arrival_min", "capacity_g", "fee_kopecks", "cargo"}, where="trip")
        for key in ("source", "destination"):
            identifier(trip[key], where=key)
        for key in ("departure_min", "arrival_min", "capacity_g", "fee_kopecks"):
            integer(trip[key], where=key)
        if trip["arrival_min"] < trip["departure_min"]:
            raise CaseError("trip:arrival_before_departure")
        string_list(trip["cargo"], where="cargo")
    unique_ids(trips, where="trips")
    return deepcopy(data)


def validate_document(data: Any) -> dict:
    fields(data, {"id", "revision", "text"}, where="document")
    identifier(data["id"], where="document_id")
    integer(data["revision"], minimum=1, where="revision")
    if type(data["text"]) is not str or not data["text"].strip():
        raise CaseError("document:text_required")
    json_size(data, limit=MAX_DOCUMENT_BYTES)
    return deepcopy(data)


@dataclass(frozen=True)
class Case:
    case_id: str
    world: dict
    documents: dict[str, list[dict]]
    events: tuple[dict, ...] = ()
    metadata: dict | None = None

    def to_dict(self) -> dict:
        return {"schema_version": 1, "case_id": self.case_id, "world": deepcopy(self.world),
                "documents": deepcopy(self.documents), "events": deepcopy(list(self.events)),
                "metadata": deepcopy(self.metadata or {})}


def case_from_dict(data: Any) -> Case:
    json_size(data, limit=MAX_CASE_BYTES)
    fields(data, {"schema_version", "case_id", "world", "documents", "events", "metadata"}, where="case")
    integer(data["schema_version"], minimum=1, maximum=1, where="schema_version")
    case_id = identifier(data["case_id"], where="case_id")
    world = validate_world(data["world"])
    fields(data["documents"], set(ROLES), where="documents")
    documents = {}
    owners = {}
    for role in ROLES:
        docs = object_list(data["documents"][role], where="documents", maximum=8)
        if not docs:
            raise CaseError("documents:each_role_requires_document")
        documents[role] = [validate_document(doc) for doc in docs]
        for doc in documents[role]:
            if doc["id"] in owners:
                raise CaseError("documents:duplicate_id")
            owners[doc["id"]] = role
    if type(data["metadata"]) is not dict:
        raise CaseError("metadata:object_required")
    events = object_list(data["events"], where="events", maximum=16)
    revisions = {doc["id"]: doc["revision"] for docs in documents.values() for doc in docs}
    future_world = deepcopy(world)
    previous_step = 0
    for event in events:
        fields(event, {"after_step", "owner", "document", "world_patch"}, where="event")
        step = integer(event["after_step"], minimum=1, maximum=100_000, where="after_step")
        if step < previous_step:
            raise CaseError("events:must_be_sorted")
        previous_step = step
        owner = event["owner"]
        if type(owner) is not str or owner not in ROLES:
            raise CaseError("event:invalid_owner")
        doc = validate_document(event["document"])
        if owners.get(doc["id"]) != owner:
            raise CaseError("event:not_own_document")
        if doc["revision"] != revisions[doc["id"]] + 1:
            raise CaseError("event:revision_must_increment")
        revisions[doc["id"]] = doc["revision"]
        owned_key = {"coordinator": "request", "supply": "offers", "logistics": "trips"}[owner]
        fields(event["world_patch"], {owned_key}, where="world_patch")
        future_world.update(deepcopy(event["world_patch"]))
        validate_world(future_world)
    return Case(case_id, world, documents, tuple(deepcopy(events)), deepcopy(data["metadata"]))


def load_case(path: str | Path) -> Case:
    """Отсутствующий или повреждённый набор задач считается ошибкой. Запасных примеров нет."""
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_CASE_BYTES + 1)
    return case_from_dict(strict_json_loads(raw))


def world_of(case_or_world: Case | dict) -> dict:
    """Возвращает доверенный снимок мира.

    Мир в виде обычного dict от служебных инструментов родительского процесса
    сначала проверяется.
    """
    if isinstance(case_or_world, Case):
        return case_or_world.world
    json_size(case_or_world, limit=MAX_CASE_BYTES)
    return validate_world(case_or_world)
