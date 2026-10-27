"""
Набор для работы над «Полигоном 2: кодовый агент».

Что здесь есть:
    open_task("t01"): решить учебную задачу руками теми же командами,
        что и агент: read, write, append, edit, run, submit;
    self_check(act): прогнать агента по учебным задачам и увидеть баллы;
    self_check(act, model="recorded"): то же без сети, на записанных ответах модели;
    check_tests("t03", source): посчитать мутационный балл своих тестов к задаче tests;
    ask_model(system, prompt): обратиться к модели курса (deepseek-flash);
    submit("agent.py"): отправить агента на боевой прогон (тур c7).

Учебные задачи лежат в tasks/train вместе со скрытыми проверками, мутантами и
эталонами: подсматривать в них можно, на боевые задачи это не переносится.
Боевые задачи другие и в набор не входят.
"""
from __future__ import annotations

import contextlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from engine.checks import evaluate, evaluate_tests  # noqa: E402
from engine.local import SubprocessHarness, print_report, run_tasks  # noqa: E402
from engine.workspace import Workspace, load_set, load_task  # noqa: E402

PUBLIC_POLYGON_URL = "https://lectures.ai.dscs.pro"
MODEL = "deepseek-flash"
TRAIN = HERE / "tasks" / "train"
REPLIES = HERE / "replies"
CALLS_PER_TASK = 8


def _load_env() -> None:
    """.env рядом с набором или в корне репозитория; окружение важнее файла."""
    for path in (HERE / ".env", HERE.parent / ".env"):
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                key, sep, value = line.partition("=")
                if sep and key.strip() and not key.strip().startswith("#"):
                    os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _harness() -> SubprocessHarness:
    return SubprocessHarness(env=dict(os.environ, POLYGON_SANDBOX_LANDLOCK="off"))


def _find_task(task_id: str):
    for path in sorted(TRAIN.iterdir()):
        if path.name.split("_", 1)[0] == task_id or path.name == task_id:
            return load_task(str(path))
    raise KeyError(f"нет учебной задачи {task_id}; есть: "
                   + ", ".join(p.name.split('_', 1)[0] for p in sorted(TRAIN.iterdir()) if p.is_dir()))


# ---------------------------------------------------------- ручной режим

class ManualTask:
    """Учебная задача, которую решает человек: каждая команда тратит шаг, как у агента."""

    def __init__(self, task_id: str):
        self.ws = Workspace(_find_task(task_id), _harness())
        task = self.ws.task
        print(f"{task.id} ({task.kind}): {task.title}\n\n{task.statement}\n")
        self.files()

    def files(self) -> None:
        for item in self.ws.observation()["files"]:
            print(f"  {item['path']}  ({item['bytes']} байт)")
        print(f"осталось шагов: {self.ws.steps_left}")

    def _do(self, action: dict) -> None:
        self.ws.apply(action)
        last = self.ws.last
        print(last["output"] if last["ok"] else f"ошибка: {last['output']}")
        print(f"осталось шагов: {self.ws.steps_left}")
        if self.ws.done:
            self._score()

    def read(self, path: str) -> None:
        self._do({"cmd": "read", "path": path})

    def write(self, path: str, content: str) -> None:
        self._do({"cmd": "write", "path": path, "content": content})

    def append(self, path: str, content: str) -> None:
        self._do({"cmd": "append", "path": path, "content": content})

    def edit(self, path: str, old: str, new: str) -> None:
        self._do({"cmd": "edit", "path": path, "old": old, "new": new})

    def run(self) -> None:
        self._do({"cmd": "run"})

    def submit(self) -> None:
        self._do({"cmd": "submit"})

    def _score(self) -> None:
        result = evaluate(self.ws)
        print(f"\nбалл {result['score']:.2f}: {result['detail']}")
        for flag in result["flags"]:
            print(f"отметка: {flag}")


def open_task(task_id: str) -> ManualTask:
    return ManualTask(task_id)


# ------------------------------------------------------------- модель

class _Budget:
    """Сколько обращений осталось на текущую задачу; так же устроен бюджет прогона на стенде."""
    left = CALLS_PER_TASK
    used = 0


def ask_model(system: str, prompt: str, max_tokens: int = 1024) -> str | None:
    """Один запрос к deepseek-flash через прокси Полигона; без токена, бюджета или при ошибке None."""
    _load_env()
    token = os.environ.get("POLYGON_TOKEN", "").strip()
    url = (os.environ.get("POLYGON_URL", "").strip() or PUBLIC_POLYGON_URL).rstrip("/")
    if not token or _Budget.left <= 0:
        return None
    import anthropic
    _Budget.left -= 1
    _Budget.used += 1
    client = anthropic.Anthropic(api_key=token, base_url=url, timeout=90.0, max_retries=0)
    try:
        response = client.messages.create(
            model=MODEL, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": prompt}],
            thinking={"type": "disabled"}, extra_body={"temperature": 0.0})
    except anthropic.RateLimitError:
        print("общий лимит модели исчерпан, повторите через минуту", file=sys.stderr)
        return None
    except anthropic.APIError as error:
        print(f"модель не ответила: {type(error).__name__}", file=sys.stderr)
        return None
    return "".join(b.text for b in response.content if getattr(b, "type", "") == "text")


@contextlib.contextmanager
def _patched_ask(module, mode: str, task_ids: list[str]):
    """Подменить ask в модуле агента: учитывать бюджет задачи, а в режиме recorded выдавать записанные ответы."""
    replies = {}
    for task_id in task_ids:
        path = REPLIES / f"{task_id}.json"
        replies[task_id] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    state = {"task": None, "index": 0}
    original = getattr(module, "ask", None)

    def ask(system, prompt, max_tokens=1024):
        if _Budget.left <= 0:
            return None
        _Budget.left -= 1
        _Budget.used += 1
        if mode == "recorded":
            queue = replies.get(state["task"], [])
            state["index"] += 1
            return queue[state["index"] - 1] if state["index"] <= len(queue) else None
        return original(system, prompt, max_tokens) if original else None

    if module is not None and original is not None:
        module.ask = ask
    try:
        yield state
    finally:
        if module is not None and original is not None:
            module.ask = original


def self_check(act, tasks: list[str] | None = None, model: str = "live", verbose: bool = False,
               *, trace: bool = False) -> dict:
    """Прогнать агента; model="recorded" работает на записанных ответах, без сети.

    trace=True добавляет tasks[].trace и termination, без содержимого файлов,
    запросов к модели и вывода команд. При исключении прогон, как прежде,
    прерывается без полного отчёта. По умолчанию формат отчёта не меняется.
    """
    _load_env()
    selected = [t for t in load_set(str(TRAIN)) if tasks is None or t.id in tasks]
    module = sys.modules.get(getattr(act, "__module__", ""), None)
    report = {"total": 0.0, "tasks": []}

    def budget():
        return {"calls_left": _Budget.left}

    results = []
    with _patched_ask(module, model, [t.id for t in selected]) as recording:
        for task in selected:
            _Budget.left = CALLS_PER_TASK
            recording["task"], recording["index"] = task.id, 0
            part = run_tasks(act, [task], _harness(), extra=budget, verbose=verbose, trace=trace)
            results += part["tasks"]
    report["tasks"] = results
    report["total"] = round(100 * sum(r["score"] for r in results) / len(results), 2) if results else 0.0
    print_report(report)
    return report


def check_tests(task_id: str, source: str) -> dict:
    """Мутационный балл файла тестов для учебной задачи вида tests."""
    task = _find_task(task_id)
    if task.kind != "tests":
        raise ValueError(f"{task_id}: задача вида {task.kind}, а не tests")
    files = dict(task.files)
    files[task.tests_path] = source
    result = evaluate_tests(task, files, _harness())
    print(f"балл {result['score']:.2f}: {result['detail']}")
    return result


# ------------------------------------------------------------ отправка

def submit(path: str = "agent.py", token: str = "", challenge: str = "c7") -> dict:
    """Отправить агента на боевой прогон. Две отправки в сутки."""
    import requests

    _load_env()
    token = (token or os.environ.get("POLYGON_TOKEN", "")).strip()
    if not token:
        raise ValueError("нужен POLYGON_TOKEN в .env или аргумент token")
    if not os.path.exists(path):
        raise FileNotFoundError(f"нет файла {path}")
    url = (os.environ.get("POLYGON_URL", "").strip() or PUBLIC_POLYGON_URL).rstrip("/")
    with open(path, "rb") as f:
        response = requests.post(f"{url}/submit", headers={"Authorization": f"Bearer {token}"},
                                 files={"solution": (os.path.basename(path), f, "text/x-python")},
                                 data={"challenge": challenge}, timeout=60)
    if response.status_code >= 400:
        detail = response.json().get("detail", response.text) if response.headers.get(
            "content-type", "").startswith("application/json") else response.text
        raise RuntimeError(f"стенд не принял отправку ({response.status_code}): {detail}")
    data = response.json()
    print(f"принято, номер прогона {data.get('job_id')}, осталось отправок сегодня: "
          f"{data.get('left_today')}. Результат: {url}/run/{data.get('job_id')}")
    return data
