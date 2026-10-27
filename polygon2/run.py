from polygon2 import open_task

task = open_task("t02")
task.write(
    "offer.py",
    """from decimal import ROUND_HALF_UP, Decimal

KOPECK = Decimal("0.01")


def order_cost(qty, unit_price, threshold, discount):
    if type(qty) is not int:                     # bool тоже: type(True) is bool
        raise TypeError("количество должно быть целым")
    if qty < 0:
        raise ValueError("количество не может быть отрицательным")
    price = Decimal(str(unit_price))
    rate = Decimal(str(discount))
    if price < 0 or not Decimal(0) <= rate < Decimal(1):
        raise ValueError("недопустимые условия поставщика")
    total = price * qty
    if threshold is not None and qty >= threshold:   # порог включительно
        total = total * (Decimal(1) - rate)
    return total.quantize(KOPECK, rounding=ROUND_HALF_UP)""",
)
task.run()
task.submit()
