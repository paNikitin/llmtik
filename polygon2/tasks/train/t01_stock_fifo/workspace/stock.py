"""Склад партий скоропортящегося товара.

Партия задаётся словарём {"day": день поступления, "qty": количество,
"expires": последний день, когда товар ещё можно продать}.
Функции не меняют переданные списки и словари, а возвращают новые.
"""


def add_lot(lots, day, qty, shelf_life):
    """Добавить партию, поступившую в день day и годную shelf_life дней."""
    if qty <= 0:
        raise ValueError("количество должно быть положительным")
    if shelf_life <= 0:
        raise ValueError("срок годности должен быть положительным")
    return lots + [{"day": day, "qty": qty, "expires": day + shelf_life - 1}]


def expire(lots, today):
    """Убрать партии, которые нельзя продавать в день today; вернуть (остаток, списано)."""
    alive = [dict(lot) for lot in lots if lot["expires"] >= today]
    spoiled = sum(lot["qty"] for lot in lots if lot["expires"] < today)
    return alive, spoiled


def consume(lots, qty):
    """Продать до qty единиц, начиная с самых старых партий; вернуть (партии, продано)."""
    if qty < 0:
        raise ValueError("нельзя продать отрицательное количество")
    result = [dict(lot) for lot in lots]
    sold = 0
    for lot in sorted(result, key=lambda item: item["day"]):
        if sold == qty:
            break
        take = min(lot["qty"], qty - sold)
        lot["qty"] -= take
        sold += take
    return [lot for lot in result if lot["qty"] > 0], sold


def total(lots):
    """Сколько единиц на складе."""
    return sum(lot["qty"] for lot in lots)
