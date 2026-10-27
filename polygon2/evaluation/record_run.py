"""Локально записать команды агента. По умолчанию сеть и отправка на стенд не нужны.

Пример из каталога polygon2: python evaluation/record_run.py --demo --out demo.json
Для настоящей модели нужно явно выбрать --model live. В файл не попадают
промпты, ответы модели, память, содержимое файлов и текст тестовых ошибок.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stdout
import hashlib
import io
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent


def demo_act(observation, memory):
    """Демонстрация записи, а не решение задачи: прочитать модуль, проверить, сдать."""
    phase = memory.get("phase", 0)
    commands = [{"cmd": "read", "path": observation["target"]},
                {"cmd": "run"}, {"cmd": "submit"}]
    memory["phase"] = phase + 1
    return commands[min(phase, 2)]


def polygon_dir(requested):
    candidates = [Path(requested)] if requested else [Path.cwd(), HERE.parent]
    for directory in candidates:
        if (directory / "polygon2.py").is_file():
            return directory.resolve()
    raise ValueError("Запускайте из каталога polygon2 или укажите --polygon-dir")


def data_digest(root, selected):
    """Отпечаток выбранных учебных задач и записанных ответов, без их содержимого в отчёте."""
    digest = hashlib.sha256()
    directories = [p for p in (root / "tasks" / "train").iterdir()
                   if p.is_dir() and p.name.split("_")[0] in selected]
    found = {p.name.split("_")[0] for p in directories}
    if found != set(selected):
        raise ValueError("Не все --task найдены в учебном наборе")
    paths = [p for directory in directories for p in directory.rglob("*")
             if p.is_file() and not p.is_symlink() and "__pycache__" not in p.parts
             and p.name != ".DS_Store"]
    paths += [root / "replies" / f"{task}.json" for task in selected]
    for path in sorted(set(paths)):
        if not path.is_file():
            continue
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def load_agent(path):
    name = "practice3_user_agent"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError("Не удалось загрузить модуль агента")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if not callable(getattr(module, "act", None)):
        raise ValueError("В файле агента нет функции act")
    return module.act


@contextmanager
def offline_environment(polygon, offline):
    """В автономных режимах не читаем настоящий .env; поведение live не меняем."""
    original = polygon._load_env
    if offline:
        polygon._load_env = lambda: None
    try:
        yield
    finally:
        polygon._load_env = original


def safe_report(report):
    """Для учебной записи берём только численные итоги и безопасную трассу клиента."""
    rows = []
    for row in report["tasks"]:
        clean = {key: row[key] for key in
                 ("task_id", "kind", "score", "steps", "runs", "trace", "termination") if key in row}
        # Движок хранит solved до штрафов. В сравнении нужен полный итоговый балл.
        clean["solved"] = row["score"] == 1
        if type(row.get("solved")) is bool:
            clean["checks_solved"] = row["solved"]
        rows.append(clean)
    return {"total": report["total"], "tasks": rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--polygon-dir")
    parser.add_argument("--agent", help="Файл агента; по умолчанию agent.py в polygon2")
    parser.add_argument("--demo", action="store_true", help="Готовые три команды, без модели")
    parser.add_argument("--model", choices=("recorded", "live"), default="recorded")
    parser.add_argument("--task", action="append", dest="tasks", help="Например t01; можно повторить")
    parser.add_argument("--label", default="default", help="Метка неизменных условий опыта, без секретов")
    parser.add_argument("--out", required=True, help="Новый JSON-файл; существующий не перезаписывается")
    args = parser.parse_args(argv)
    if args.demo and (args.agent or args.model == "live"):
        parser.error("--demo не совмещается с --agent или --model live")
    out = Path(args.out)
    if out.exists():
        parser.error("Файл --out уже существует: выберите новое имя, чтобы сохранить прежний опыт")
    root = polygon_dir(args.polygon_dir)
    sys.path.insert(0, str(root))
    import polygon2 as polygon

    agent_path = Path(args.agent).resolve() if args.agent else root / "agent.py"
    source = Path(__file__).read_bytes() if args.demo else agent_path.read_bytes()
    code_hash = hashlib.sha256(source).hexdigest()
    selected = sorted(set(args.tasks or ["t01"]))
    config = {"agent_sha256": code_hash, "mode": "demo" if args.demo else args.model,
              "label": args.label, "calls_per_task": polygon.CALLS_PER_TASK,
              "data_sha256": data_digest(root, selected),
              "client_sha256": hashlib.sha256((root / "polygon2.py").read_bytes()).hexdigest(),
              "engine_sha256": hashlib.sha256((root / "engine" / "local.py").read_bytes()).hexdigest()}
    config_id = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    act = demo_act if args.demo else load_agent(agent_path)
    with offline_environment(polygon, args.demo or args.model != "live"), redirect_stdout(io.StringIO()):
        raw = polygon.self_check(act, tasks=selected,
                                 model="recorded" if args.demo else args.model, trace=True)
    report = safe_report(raw)
    report.update({"schema_version": 1, "config_id": config_id, "config": config,
                   "provenance": ("Локальная демонстрация записи команд без модели." if args.demo
                                  else "Локальный запуск: " + args.model)})
    # Разрешаем сериализовать только значения строгого JSON. Не стираем старый опыт.
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with out.open("x", encoding="utf-8") as handle:
        handle.write(payload)
    print(f"Сохранено задач: {len(report['tasks'])}; режим: {config['mode']}; файл: {out}")
    print("config_id проверяет код и записанные условия, но не фиксирует версию сервера модели.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as error:
        print(f"Не удалось записать прогон: {error}", file=sys.stderr)
        raise SystemExit(1)
