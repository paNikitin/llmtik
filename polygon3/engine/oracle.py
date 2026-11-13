"""Точный привилегированный решатель с конечным перебором.

Каждая строка целиком получает одно предложение из списка и один прямой рейс из
списка. Декартово произведение вариантов конечно: не больше 3 строк,
12 предложений и 12 рейсов. Можно отбрасывать варианты, которые нарушают
правила для отдельной строки, упираются в исчерпанные ресурсы или дают
частичную стоимость выше уже известной границы. Это допустимо, потому что
количества, веса и все платы неотрицательны. Проверки кратности упаковке и
минимального количества откладываются до полного назначения: две строки
вместе могут выполнить минимум или кратность, даже если ни одна из них не
выполняет их отдельно.
"""
from __future__ import annotations
from copy import deepcopy

from .schema import Case, world_of
from .constraints import local_violations, inspect_assignment


def solve(case_or_world: Case | dict) -> dict:
    world = world_of(case_or_world)
    request = world["request"]
    offers = {item["id"]: item for item in world["offers"]}
    trips = {item["id"]: item for item in world["trips"]}
    candidates = []
    for line in request["lines"]:
        options = []
        quantity = line["quantity"]
        for offer in world["offers"]:
            if quantity > offer["stock"]:
                continue
            for trip in world["trips"]:
                if quantity * offer["unit_weight_g"] > trip["capacity_g"]:
                    continue
                if not local_violations(request, line, offer, trip):
                    options.append({"line_id": line["id"], "offer_id": offer["id"],
                                    "trip_id": trip["id"], "quantity": quantity})
        candidates.append(options)
    best_cost: int | None = None
    best_plan: dict | None = None
    quantities: dict[str, int] = {}
    weights: dict[str, int] = {}
    selected: list[dict] = []

    def search(index: int, partial_cost: int) -> None:
        nonlocal best_cost, best_plan
        if partial_cost > request["budget_kopecks"]:
            return
        if best_cost is not None and partial_cost >= best_cost:
            return  # Чтобы вернуть один оптимум, другие равные оптимумы перебирать не нужно.
        if index == len(candidates):
            errors, cost = inspect_assignment(world, selected)
            if not errors:
                best_cost = cost
                best_plan = {"status": "feasible", "lines": deepcopy(selected)}
            return
        for selection in candidates[index]:
            offer_id, trip_id, quantity = selection["offer_id"], selection["trip_id"], selection["quantity"]
            offer, trip = offers[offer_id], trips[trip_id]
            previous_qty = quantities.get(offer_id, 0)
            previous_weight = weights.get(trip_id, 0)
            total_qty = previous_qty + quantity
            total_weight = previous_weight + quantity * offer["unit_weight_g"]
            if total_qty > offer["stock"] or total_weight > trip["capacity_g"]:
                continue
            extra = quantity * offer["unit_price_kopecks"]
            if offer_id not in quantities:
                extra += offer["fee_kopecks"]
            if trip_id not in weights:
                extra += trip["fee_kopecks"]
            quantities[offer_id], weights[trip_id] = total_qty, total_weight
            selected.append(selection)
            search(index + 1, partial_cost + extra)
            selected.pop()
            if previous_qty:
                quantities[offer_id] = previous_qty
            else:
                del quantities[offer_id]
            if previous_weight:
                weights[trip_id] = previous_weight
            else:
                del weights[trip_id]

    search(0, 0)
    return {"feasible": best_plan is not None, "optimal_cost": best_cost,
            "plan": best_plan if best_plan is not None else {"status": "infeasible"}}
