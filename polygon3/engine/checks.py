"""Строгая проверка плана.

Основная оценка правильности равна 0 или 1. Единицу дают любой выполнимый
полный план или верное утверждение, что плана не существует. Оптимальная
стоимость и разница с ней считаются отдельно, как дополнительные показатели.
Остановка прогона, неправильно оформленный план или неудачный поиск плана
никогда не доказывают, что задача невыполнима.
"""
from __future__ import annotations
from typing import Any

from .schema import (Case, CaseError, MAX_PLAN_BYTES, fields, identifier,
                     integer, json_size, strict_json_loads, world_of)
from .constraints import inspect_assignment


def parse_plan(plan: Any) -> dict:
    """Принимает объект JSON или строгий текст JSON.

    В уже раскодированном dict повторяющиеся ключи JSON не сохраняются. Поэтому
    код, который получает JSON как текст из канала, должен вызывать
    strict_json_loads. Отклоняются неизвестные поля, в том числе стоимость,
    роль, мир и версия, присланные агентом. Также отклоняются количества типа
    bool или float, NaN и слишком большие данные.
    """
    if type(plan) in (str, bytes):
        plan = strict_json_loads(plan, limit=MAX_PLAN_BYTES)
    json_size(plan, limit=MAX_PLAN_BYTES)
    if type(plan) is not dict or type(plan.get("status")) is not str:
        raise CaseError("plan:status_required")
    if plan["status"] == "infeasible":
        fields(plan, {"status"}, where="plan")
        return {"status": "infeasible"}
    if plan["status"] != "feasible":
        raise CaseError("plan:invalid_status")
    fields(plan, {"status", "lines"}, where="plan")
    if type(plan["lines"]) is not list or not 1 <= len(plan["lines"]) <= 3:
        raise CaseError("plan:one_to_three_lines_required")
    seen = set()
    lines = []
    for selection in plan["lines"]:
        fields(selection, {"line_id", "offer_id", "trip_id", "quantity"}, where="selection")
        for key in ("line_id", "offer_id", "trip_id"):
            identifier(selection[key], where=key)
        integer(selection["quantity"], minimum=1, where="quantity")
        if selection["line_id"] in seen:
            raise CaseError("plan:duplicate_line")
        seen.add(selection["line_id"])
        lines.append(dict(selection))
    return {"status": "feasible", "lines": lines}


def evaluate_plan(case_or_world: Case | dict, plan: Any) -> dict:
    """Доверенная итоговая оценка. Её нельзя давать ролям как инструмент для повторных проверок.

    Первый аргумент во всех пространствах имён пакета по-прежнему принимает Case.
    Неправильный канонический мир вызывает CaseError и не засчитывается как
    невыполнимая задача.
    """
    world = world_of(case_or_world)
    # Импорт внутри функции не даёт возникнуть циклическому импорту.
    from .oracle import solve
    truth = solve(world)
    optimal_cost = truth["optimal_cost"]
    result = {"correct": 0, "status": "invalid_plan", "violations": [],
              "cost": None, "optimal_cost": optimal_cost, "cost_gap": None,
              "is_optimal": False}
    try:
        normalized = parse_plan(plan)
    except CaseError as exc:
        result["violations"] = [{"code": "invalid_plan", "detail": str(exc)}]
        return result
    if normalized["status"] == "infeasible":
        result.update(correct=int(not truth["feasible"]),
                      status="infeasible" if not truth["feasible"] else "false_infeasible")
        if truth["feasible"]:
            result["violations"] = [{"code": "infeasible_claim_false"}]
        return result
    lines = {item["id"]: item for item in world["request"]["lines"]}
    offer_ids = {item["id"] for item in world["offers"]}
    trip_ids = {item["id"] for item in world["trips"]}
    errors = []
    chosen = {selection["line_id"] for selection in normalized["lines"]}
    if chosen != set(lines):
        errors.append({"code": "line_set_mismatch"})
    for selection in normalized["lines"]:
        line_id = selection["line_id"]
        if line_id not in lines:
            errors.append({"code": "unknown_line_id", "line_id": line_id})
        elif selection["quantity"] != lines[line_id]["quantity"]:
            errors.append({"code": "quantity_mismatch", "line_id": line_id})
        if selection["offer_id"] not in offer_ids:
            errors.append({"code": "unknown_offer_id", "line_id": line_id})
        if selection["trip_id"] not in trip_ids:
            errors.append({"code": "unknown_trip_id", "line_id": line_id})
    if errors:
        result["violations"] = errors
        return result
    errors, cost = inspect_assignment(world, normalized["lines"])
    result.update(status="invalid" if errors else "feasible", correct=int(not errors),
                  violations=errors, cost=cost)
    if not errors:
        result["cost_gap"] = cost - optimal_cost
        result["is_optimal"] = cost == optimal_cost
    return result
