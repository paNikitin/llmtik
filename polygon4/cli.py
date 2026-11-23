"""Командный интерфейс Полигона 4: прогон, проверка по train, пульт studio и отправка."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib import error, request
from urllib.parse import urlsplit

from .corpora import build_scenarios
from .spec import stand_score
from .studio.server import create_app, resolve_actor, run_episode, scenario_by_id, serve


def _setting(name: str) -> str:
    val = os.environ.get(name, "").strip()
    if val:
        return val
    for candidate in (Path.cwd() / ".env", Path(__file__).resolve().parent / ".env", Path(__file__).resolve().parents[1] / ".env"):
        if candidate.is_file():
            for line in candidate.read_text(encoding="utf-8").splitlines():
                key, sep, raw = line.partition("=")
                if sep and key.strip() == name:
                    return raw.strip().strip('"').strip("'")
    return ""


def _cmd_run(args: argparse.Namespace) -> int:
    scenario = scenario_by_id(args.scenario)
    name, actor = resolve_actor(args.agent, scenario)
    report = run_episode(scenario, actor, args.mode)
    report["agent"] = name
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"{scenario.scenario_id} · агент={name} · прибыль={report['net_profit']:.0f} ₽ · "
        f"балл={report['stand_score']['total']:.2f} · контракты={report['contracts_completed']} "
        f"(срывов={report['contracts_breached']}) · LLM={report['model_calls']} · ревизор={report['audits_used']}"
    )
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    scenarios = build_scenarios("train")
    if args.limit:
        scenarios = scenarios[: max(1, args.limit)]
    profits: list[float] = []
    breaches = 0
    violations = 0
    total_calls = 0
    peak_state_bytes = 0
    for sc in scenarios:
        name, actor = resolve_actor(args.agent, sc)
        report = run_episode(sc, actor, args.mode)
        profits.append(report["net_profit"])
        breaches += report["contracts_breached"]
        violations += sum(len(d.get("policy_flags") or []) for d in report["days"])
        total_calls += int(report.get("model_calls", 0))
        peak_state_bytes = max(peak_state_bytes, int(report.get("peak_state_bytes", 0)))
        if not args.quiet:
            print(
                f"  {sc.scenario_id}: {report['net_profit']:>9.0f} ₽  "
                f"контракты={report['contracts_completed']}  срывы={report['contracts_breached']}  "
                f"LLM={report['model_calls']:>2}  аудит={report['audits_used']}"
            )
    summary = stand_score(profits, model_calls=total_calls, peak_state_bytes=peak_state_bytes)
    print(
        f"Итог ({len(profits)} сценариев): балл = {summary['total']:.2f} "
        f"(прибыль {summary['profit']:.0f} ₽, до штрафов {summary['raw_total']:.2f}, "
        f"LLM −{summary['call_penalty']:.2f}, memory −{summary['memory_penalty']:.2f} [{summary['state_peak_bytes']} Б], "
        f"срывов SLA: {breaches}, флагов регламента: {violations})"
    )
    return 0


def _cmd_studio(args: argparse.Namespace) -> int:
    app = create_app()
    try:
        import uvicorn
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    except ImportError:
        serve(app, host=args.host, port=args.port)
    return 0


def _cmd_upload(args: argparse.Namespace) -> int:
    if not args.confirm:
        raise SystemExit("Для реальной отправки на стенд добавьте флаг --confirm")
    agent_path = Path(args.agent).expanduser().resolve()
    if not agent_path.is_file() or agent_path.suffix != ".py":
        raise SystemExit(f"Файл решения не найден: {agent_path}")
    code_bytes = agent_path.read_bytes()
    token = _setting("POLYGON_TOKEN")
    if not token:
        raise SystemExit("Не задан POLYGON_TOKEN в окружении или .env")
    base_url = (_setting("POLYGON_URL") or "https://lectures.ai.dscs.pro").rstrip("/")
    parsed = urlsplit(base_url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise SystemExit("Некорректный адрес POLYGON_URL")

    boundary = "----Polygon4UploadBoundary7f3a9c"
    parts = [
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"challenge\"\r\n\r\n{args.challenge}\r\n".encode("utf-8"),
        (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"solution\"; "
            f"filename=\"{agent_path.name}\"\r\nContent-Type: text/x-python\r\n\r\n"
        ).encode("utf-8")
        + code_bytes
        + f"\r\n--{boundary}--\r\n".encode("utf-8"),
    ]
    body = b"".join(parts)
    req = request.Request(
        base_url + "/submit",
        data=body,
        method="POST",
        headers={
            "Authorization": "Bearer " + token,
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
    )
    try:
        with request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"Отправка отклонена ({exc.code}): {detail}") from None
    print(f"Отправлено: прогон #{payload['job_id']} · страница {base_url}/run/{payload['job_id']} · осталось попыток сегодня: {payload.get('left_today')}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="polygon4", description="Финальный Полигон 4: операционный центр и WhatIfSandbox")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Запустить один учебный сценарий")
    p_run.add_argument("--agent", required=True, help="Путь к agent.py или имя политики")
    p_run.add_argument("--scenario", default="train_01")
    p_run.add_argument("--mode", choices=("scripted", "live"), default="scripted")
    p_run.add_argument("--output", default="")
    p_run.set_defaults(func=_cmd_run)

    p_chk = sub.add_parser("check", help="Проверить агента на наборе открытых сценариев train")
    p_chk.add_argument("--agent", required=True)
    p_chk.add_argument("--mode", choices=("scripted", "live"), default="scripted")
    p_chk.add_argument("--limit", type=int, default=0)
    p_chk.add_argument("--quiet", action="store_true")
    p_chk.set_defaults(func=_cmd_check)

    p_st = sub.add_parser("studio", help="Запустить интерактивную веб-диспетчерскую")
    p_st.add_argument("--host", default="127.0.0.1")
    p_st.add_argument("--port", type=int, default=8780)
    p_st.set_defaults(func=_cmd_studio)

    p_up = sub.add_parser("upload", help="Отправить файл агента на стенд (тур c9)")
    p_up.add_argument("--agent", required=True)
    p_up.add_argument("--challenge", default="c9")
    p_up.add_argument("--confirm", action="store_true")
    p_up.set_defaults(func=_cmd_upload)

    ns = parser.parse_args(argv)
    return int(ns.func(ns))
