"""
Кодовый агент для «Полигона 2».

Агент решает задачи командами среды: read, write, append, edit, run, submit.
Готово: обращение к модели (`ask`) и цикл `act`, который хранит состояние
задачи в memory и по очереди: читает нужные файлы, спрашивает модель,
применяет правку, запускает видимые тесты, решает, сдавать ли задачу.

Вам нужно дописать шесть функций с пометкой TODO:
  1. choose_reads: выбрать файлы для чтения и не открывать лишние;
  2. parse_report: разобрать вывод команды run;
  3. extract_commands: превратить ответ модели в команды среды;
  4. should_submit: решить, когда сдавать задачу;
  5. drop_failing_tests: убрать тесты, падающие на правильном модуле;
  6. build_prompt: составить запрос к модели.

На стенде модуль перезагружается на каждом шаге: всё, что нужно помнить,
храните в memory (только JSON, не больше 64 КиБ).
"""

from __future__ import annotations

import ast
import os
import re
import sys
import time

PUBLIC_POLYGON_URL = "https://lectures.ai.dscs.pro"
MODEL = "deepseek-flash"
FILE_LIMIT = 12000
REPORT_LIMIT = 2500
PROMPT_LIMIT = 40000
MAX_ATTEMPTS = 4


# ------------------------------------------------------------- готово: модель


def ask(system: str, prompt: str, max_tokens: int = 1024) -> str | None:
    """Один запрос к модели курса; None, если нет токена, исчерпан бюджет или случилась ошибка."""
    token = (
        os.environ.get("POLYGON_TOKEN") or os.environ.get("LLM_API_KEY") or ""
    ).strip()
    url = (
        os.environ.get("POLYGON_URL")
        or os.environ.get("LLM_BASE_URL")
        or PUBLIC_POLYGON_URL
    ).rstrip("/")
    if not token:
        return None
    try:
        import anthropic
    except ImportError:
        return None
    client = anthropic.Anthropic(
        api_key=token, base_url=url, timeout=90.0, max_retries=0
    )
    for attempt in range(2):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}],
                thinking={"type": "disabled"},
                extra_body={"temperature": 0.0},
            )
            return "".join(
                b.text for b in response.content if getattr(b, "type", "") == "text"
            )
        except anthropic.RateLimitError:
            if attempt == 0:
                time.sleep(10)
                continue
            return None
        except (anthropic.APIError, OSError, ValueError):
            return None
    return None


# ---------------------------------------------------------- 1. чтение файлов


def _module_paths(module: str, level: int, importer: str) -> list[str]:
    """Пути-кандидаты локального модуля; level — число точек в относительном импорте."""
    parts = module.split(".") if module else []
    if level:
        base = os.path.dirname(importer)
        for _ in range(level - 1):
            base = os.path.dirname(base)
    else:
        base = ""
    stem = "/".join(part for part in [base, *parts] if part)
    if not stem:
        return []
    return [stem + ".py", stem + "/__init__.py"]


def _local_imports(source: str, importer: str, available: set[str]) -> list[str]:
    """Локальные файлы из списка задачи, которые импортирует прочитанный модуль."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            wanted = [(0, alias.name) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            wanted = [(node.level, module)]
            for alias in node.names:
                if alias.name != "*":
                    name = f"{module}.{alias.name}" if module else alias.name
                    wanted.append((node.level, name))
        else:
            continue
        for level, name in wanted:
            for path in _module_paths(name, level, importer):
                if path in available and path not in found:
                    found.append(path)
    return found


def choose_reads(observation: dict, state: dict) -> list[str]:
    """Пути файлов, которые нужно прочитать, в порядке чтения.

    Целевой модуль, видимые тесты, файл тестов (в задаче tests) и локальные
    модули, которые импортируют уже прочитанные файлы. Больше ничего: в каталоге
    бывают файлы, которые выдают себя за ответы или скрытые тесты, и их чтение
    обнуляет задачу.
    """
    available = {item["path"] for item in observation.get("files", [])}
    reads: list[str] = []

    def add(path) -> None:
        if path and path in available and path not in reads:
            reads.append(path)

    add(observation.get("target"))
    for path in observation.get("visible_tests") or []:
        add(path)
    if observation.get("kind") == "tests":
        add(observation.get("tests_path"))

    files = state.get("files") or {}
    pending = list(reads) + [path for path in files if path not in reads]
    seen: set[str] = set()
    while pending:
        path = pending.pop(0)
        if path in seen:
            continue
        seen.add(path)
        source = files.get(path)
        if not isinstance(source, str):
            continue
        for local in _local_imports(source, path, available):
            add(local)
            pending.append(local)
    return reads


# ------------------------------------------------------ 2. разбор вывода run


COUNT = re.compile(r"прошло\s+(\d+)\s+из\s+(\d+)")
FAILED = re.compile(r"^(?:FAIL|ERROR|TIMEOUT)\b[:\s]+([\w.\[\]-]+)", re.M)


def parse_report(text: str) -> dict:
    """Из вывода run: сколько прошло, сколько всего, какие тесты упали, были ли ошибки импорта."""
    text = text if isinstance(text, str) else ""
    match = COUNT.search(text)
    passed, ran = (int(match.group(1)), int(match.group(2))) if match else (0, 0)
    failing: list[str] = []
    for name in FAILED.findall(text):
        short = name.rsplit(".", 1)[-1].strip("()")
        if short and short not in failing:
            failing.append(short)
    return {
        "passed": passed,
        "ran": ran,
        "failing": failing,
        "import_errors": "ошибка импорта" in text.lower(),
    }


# ------------------------------------------------- 3. ответ модели → команды

BLOCK = re.compile(
    r"(?:FILE:\s*(?P<path>[^\n]+)\n)?<{5,9} ?SEARCH\n(?P<old>.*?)\n?={5,9}\n"
    r"(?P<new>.*?)\n?>{5,9} ?REPLACE",
    re.S,
)
CODE = re.compile(r"```(?:python|py)?\n(?P<code>.*?)```", re.S)


def extract_commands(reply: str, observation: dict, state: dict) -> list[dict]:
    """Команды среды из ответа модели.

    Блоки SEARCH/REPLACE → команды edit. Иначе самый длинный блок ```python```:
    в задаче tests это файл тестов (write, а если тесты уже проходят, то append),
    в остальных это весь целевой модуль (write). Если ответ не разобран, вернуть [].
    """
    if not isinstance(reply, str) or not reply.strip():
        return []
    kind = observation.get("kind", "fix")
    target = observation.get("target") or ""
    tests_path = observation.get("tests_path") or target
    protected = (
        set() if kind == "tests" else set(observation.get("visible_tests") or [])
    )

    edits = []
    for block in BLOCK.finditer(reply):
        path = (block.group("path") or target).strip()
        if not path or path in protected:
            continue
        edits.append(
            {
                "cmd": "edit",
                "path": path,
                "old": block.group("old"),
                "new": block.group("new"),
            }
        )
    if edits:
        return edits

    blocks = CODE.findall(reply)
    if not blocks:
        return []
    content = max(blocks, key=len)
    if not content.endswith("\n"):
        content += "\n"
    if kind == "tests":
        command = "append" if state.get("passing") else "write"
        return [{"cmd": command, "path": tests_path, "content": content}]
    if target in protected:
        return []
    return [{"cmd": "write", "path": target, "content": content}]


# ----------------------------------------------------- 4. когда сдавать задачу


def report_is_green(report: dict) -> bool:
    return (
        report.get("ran", 0) > 0
        and report.get("passed") == report.get("ran")
        and not report.get("failing")
        and not report.get("import_errors")
    )


def resources_exhausted(observation: dict, state: dict) -> bool:
    return (
        state.get("attempts", 0) >= MAX_ATTEMPTS
        or observation.get("calls_left", 1) <= 0
        or observation.get("steps_left", 99) <= 3
    )


def should_submit(observation: dict, state: dict, report: dict) -> bool:
    """Сдавать, если видимые тесты проходят и правка применилась, или если ресурсы кончаются.

    В задаче tests проходящие тесты сдаются после одного раунда дополнительных
    тестов. Ресурсы: попытки (MAX_ATTEMPTS), обращения к модели (calls_left),
    шаги (оставьте запас на применение правки и прогон).
    """
    ok = report_is_green(report) and not state.get("notes")
    if ok:
        if observation.get("kind") == "tests":
            return state.get("extra", 0) >= 1
        return True
    return resources_exhausted(observation, state)


# ------------------------------------------------ 5. убрать неверные тесты


def _drop_range(doomed: set[int], node) -> None:
    """Пометить строки определения вместе с его декораторами."""
    start = node.lineno
    decorators = getattr(node, "decorator_list", None)
    if decorators:
        start = min(item.lineno for item in decorators)
    doomed.update(range(start, (node.end_lineno or node.lineno) + 1))


def drop_failing_tests(source: str, names: list[str]) -> str | None:
    """Убрать из файла тестов функции и методы с именами names; None, если файл не разбирается.

    Модуль в задаче tests правильный, поэтому тест, который на нём падает,
    проверяет неверное ожидание: его надо убрать, а не чинить модуль.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return None
    wanted = set(names or [])
    lines = source.splitlines(keepends=True)
    doomed: set[int] = set()
    replace: dict[int, str] = {}

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in wanted:
                _drop_range(doomed, node)
        elif isinstance(node, ast.ClassDef):
            body = node.body
            removed = [
                item
                for item in body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                and item.name in wanted
            ]
            for item in removed:
                _drop_range(doomed, item)
            if removed and len(removed) == len(body):
                first = lines[body[0].lineno - 1]
                indent = first[: len(first) - len(first.lstrip())]
                replace[body[0].lineno] = indent + "pass\n"

    if not doomed:
        return source
    return "".join(
        replace.get(number, line)
        for number, line in enumerate(lines, 1)
        if number in replace or number not in doomed
    )


# ------------------------------------------------------- 6. запрос к модели

SYSTEM_EDIT = (
    "Ты правишь код Python в чужом проекте. Отвечай только блоками правки, без пояснений.\n"
    "Формат блока:\nFILE: путь\n<<<<<<< SEARCH\nточный фрагмент из файла\n=======\n"
    "новый фрагмент\n>>>>>>> REPLACE\n"
    "SEARCH копирует строки файла дословно и встречается в файле один раз. Не меняй тесты. "
    "Выполни все правила из условия, включая ошибки ввода и краевые случаи; не меняй поведение, "
    "о котором условие молчит."
)
SYSTEM_TESTS = (
    "Ты пишешь тесты Python (unittest или функции test_*) к правильному модулю. Каждый тест "
    "проверяет одно правило из документации и условия: значения, границы, ошибки ввода, "
    "неизменность аргументов, порядок. Тест обязан проходить на этом модуле. Не копируй код "
    "модуля. Ответь одним блоком ```python```: не больше 12 тестов и 70 строк."
)


def build_prompt(observation: dict, state: dict) -> tuple[str, str]:
    """(system, prompt): условие, прочитанные файлы, вывод тестов и ошибки прошлой попытки."""
    kind = observation.get("kind", "fix")
    system = SYSTEM_TESTS if kind == "tests" else SYSTEM_EDIT

    title = observation.get("title") or observation.get("task_id") or ""
    statement = observation.get("statement", "").strip()
    target = observation.get("target") or ""
    text = f"Задача: {title}\nУсловие:\n{statement}\n\nЦелевой модуль: {target}"
    if kind == "tests":
        text += f"\nФайл тестов: {observation.get('tests_path', '')}"
    else:
        visible = observation.get("visible_tests") or []
        if visible:
            text += "\nВидимые тесты: " + ", ".join(visible)

    files = state.get("files") or {}
    for path, source in files.items():
        block = f"=== Файл {path} ===\n{source}"
        if len(text) + len(block) + 2 > PROMPT_LIMIT:
            break
        text += "\n\n" + block
    if not files:
        text += "\n\nФайлы ещё не прочитаны."

    if state.get("report"):
        text += "\n\nВывод видимых тестов:\n" + state["report"]
    if state.get("notes"):
        text += "\n\nОшибки прошлой попытки:\n" + "\n".join(state["notes"])

    if kind == "tests":
        text += "\n\nНапиши тесты к модулю. Ответь одним блоком ```python```."
    else:
        text += "\n\nИсправь модуль. Ответь блоками SEARCH/REPLACE."
    return system, text


# ------------------------------------------------------------- готово: цикл


def _new_state(observation: dict) -> dict:
    return {
        "id": observation["task_id"],
        "files": {},
        "queue": [],
        "attempts": 0,
        "extra": 0,
        "report": "",
        "notes": [],
        "stage": "read",
        "passing": False,
        "backup": None,
        "dropped": 0,
        "revision": 0,
        "tested_revision": None,
    }


def _absorb(observation: dict, state: dict) -> None:
    """Учесть результат прошлой команды."""
    last = observation["last"]
    if last["cmd"] == "read" and last.get("ok"):
        state["files"][last["path"]] = last["output"][:FILE_LIMIT]
    elif last["cmd"] in ("edit", "write", "append") and last.get("ok"):
        state["revision"] += 1

    elif last["cmd"] in ("edit", "write", "append") and not last.get("ok"):
        state["notes"].append(f"{last['cmd']} {last.get('path')}: {last['output']}")
        state["files"].pop(
            last.get("path"), None
        )  # перечитать перед следующей попыткой
        state["queue"] = []
    elif last["cmd"] == "run":
        state["report"] = last["output"][:REPORT_LIMIT]
        state["stage"] = "judge"
        state["tested_revision"] = state["revision"]


def _issue(state: dict) -> dict:
    """Выдать следующую подготовленную команду и обновить свою копию файла."""
    command = state["queue"].pop(0)
    path, files = command.get("path"), state["files"]
    if command["cmd"] == "edit" and path in files:
        files[path] = files[path].replace(command["old"], command["new"], 1)
    elif command["cmd"] == "write":
        files[path] = command["content"][:FILE_LIMIT]
    elif command["cmd"] == "append":
        files[path] = (files.get(path, "") + command["content"])[:FILE_LIMIT]
    state["stage"] = "applied" if not state["queue"] else "apply"
    return command


def _decide(observation: dict, state: dict) -> dict:

    present = {item["path"] for item in observation["files"]}
    for path in choose_reads(observation, state):
        if path in present and path not in state["files"]:
            return {"cmd": "read", "path": path}
    if state["queue"]:
        return _issue(state)
    if state["stage"] == "applied":
        state["stage"] = "running"
        return {"cmd": "run"}
    if state["stage"] == "judge":
        if state["tested_revision"] != state["revision"]:
            state["stage"] = "running"
            return {"cmd": "run"}
        report = parse_report(state["report"])
        ok = report_is_green(report) and not state.get("notes")
        if observation["kind"] == "tests":
            tests_path = observation["tests_path"]
            if ok and not state["passing"]:
                state["passing"], state["backup"] = True, state["files"].get(tests_path)
            elif ok:
                state["backup"] = state["files"].get(tests_path)
            elif (
                report["failing"]
                and tests_path in state["files"]
                and state["dropped"] < 2
            ):
                trimmed = drop_failing_tests(
                    state["files"][tests_path], report["failing"]
                )
                if trimmed:
                    state["dropped"] += 1
                    state["queue"] = [
                        {"cmd": "write", "path": tests_path, "content": trimmed}
                    ]
                    return _issue(state)
            elif state.get("backup"):
                state["extra"] = 1
                state["queue"] = [
                    {"cmd": "write", "path": tests_path, "content": state.pop("backup")}
                ]
                return _issue(state)
        if should_submit(observation, state, report):
            return {"cmd": "submit"}
        if ok and observation["kind"] == "tests":
            state["extra"] += 1
        state["stage"] = "ask"
    if resources_exhausted(observation, state):
        return {"cmd": "submit"}
    state["attempts"] += 1
    system, prompt = build_prompt(observation, state)
    commands = extract_commands(ask(system, prompt), observation, state)
    state["notes"], state["report"] = [], ""
    if not commands:
        state["notes"].append("модель не вернула правку, которую можно разобрать")
        return {"cmd": "run"}
    state["queue"] = commands
    return _issue(state)


def act(observation, memory):
    state = memory.get("task")
    if not state or state.get("id") != observation["task_id"]:
        state = memory["task"] = _new_state(observation)
    _absorb(observation, state)
    try:
        return _decide(observation, state)
    except NotImplementedError as todo:
        # Пока функции не написаны, задача сдаётся как есть и получает ноль баллов.
        print(f"не готово: {todo}", file=sys.stderr)
        return {"cmd": "submit"}


if __name__ == "__main__":
    from polygon2 import self_check

    self_check(act, model="recorded")
