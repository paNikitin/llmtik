"""Опубликованные правила задачи. Их используют проверка плана и доверенный решатель с конечным перебором.

Строка заказа целиком использует ровно одно предложение и один *прямой* рейс.
Нужен исходный SKU или явно разрешённая замена. У предложения должны быть все
обязательные свойства. Действуют категории груза и из заявки, и из предложения,
даже если одна из сторон какую-то категорию не указала. Рейс должен принимать
объединение этих категорий, начинаться в пункте отправки предложения,
заканчиваться в пункте назначения заявки, отправляться не раньше готовности
товара и прибывать не позже срока. Обе границы включаются.

Остаток на складе, минимальное количество и кратность упаковке проверяются для
СУММЫ количеств по всем строкам с этим предложением, а не для каждой строки
отдельно. Фиксированная плата за предложение и фиксированная плата за рейс
берутся ОДИН раз. Вместимость рейса проверяется для СУММЫ произведений
фактического веса единицы товара в предложении на заказанное количество.
Покупки сверх нужного, деления строки, перегрузки с рейса на рейс, непрерывной
оптимизации и маршрута, который угадан или взят по умолчанию, здесь нет.
"""
from __future__ import annotations


def local_violations(request: dict, line: dict, offer: dict, trip: dict) -> list[str]:
    errors = []
    if offer["sku"] not in {line["sku"], *line["allowed_skus"]}:
        errors.append("sku_not_allowed")
    if not set(line["required_properties"]).issubset(offer["properties"]):
        errors.append("required_properties")
    if offer["source"] != trip["source"]:
        errors.append("source_mismatch")
    if trip["destination"] != request["destination"]:
        errors.append("destination_mismatch")
    if offer["ready_min"] > trip["departure_min"]:
        errors.append("not_ready_at_departure")
    if trip["arrival_min"] > request["deadline_min"]:
        errors.append("arrives_after_deadline")
    if not (set(line["cargo"]) | set(offer["cargo"])).issubset(trip["cargo"]):
        errors.append("cargo_not_accepted")
    return errors


def assignment_cost(world: dict, selections: list[dict]) -> int:
    """Вызывать только для известных ID предложений и рейсов и точных заказанных количеств."""
    offers = {item["id"]: item for item in world["offers"]}
    trips = {item["id"]: item for item in world["trips"]}
    return (sum(offers[item["offer_id"]]["unit_price_kopecks"] * item["quantity"] for item in selections)
            + sum(offers[key]["fee_kopecks"] for key in {item["offer_id"] for item in selections})
            + sum(trips[key]["fee_kopecks"] for key in {item["trip_id"] for item in selections}))


def inspect_assignment(world: dict, selections: list[dict]) -> tuple[list[dict], int]:
    """Проверяет полное назначение с известными ID, у которого структура уже проверена."""
    request = world["request"]
    lines = {item["id"]: item for item in request["lines"]}
    offers = {item["id"]: item for item in world["offers"]}
    trips = {item["id"]: item for item in world["trips"]}
    quantities: dict[str, int] = {}
    weights: dict[str, int] = {}
    errors = []
    for selection in selections:
        line = lines[selection["line_id"]]
        offer = offers[selection["offer_id"]]
        trip = trips[selection["trip_id"]]
        errors.extend({"code": code, "line_id": line["id"]} for code in local_violations(request, line, offer, trip))
        quantities[offer["id"]] = quantities.get(offer["id"], 0) + selection["quantity"]
        weights[trip["id"]] = weights.get(trip["id"], 0) + selection["quantity"] * offer["unit_weight_g"]
    for offer_id, quantity in quantities.items():
        offer = offers[offer_id]
        if quantity > offer["stock"]:
            errors.append({"code": "offer_stock", "offer_id": offer_id})
        if quantity < offer["min_qty"]:
            errors.append({"code": "offer_min_qty", "offer_id": offer_id})
        if quantity % offer["pack_qty"]:
            errors.append({"code": "offer_pack_qty", "offer_id": offer_id})
    for trip_id, weight in weights.items():
        if weight > trips[trip_id]["capacity_g"]:
            errors.append({"code": "trip_capacity", "trip_id": trip_id})
    cost = assignment_cost(world, selections)
    if cost > request["budget_kopecks"]:
        errors.append({"code": "budget_exceeded"})
    return errors, cost
