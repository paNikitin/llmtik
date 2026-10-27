"""Проверки готового помощника записи. Ни сеть, ни настоящий .env не используются."""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import record_run

MARKER = "PRIVATE_SYNTHETIC_TEXT_NOT_FOR_REPORT"


class ReadyRecorderTests(unittest.TestCase):
    def test_demo_returns_command_and_updates_json_memory(self):
        memory = {}
        commands = [record_run.demo_act({"target": "module.py"}, memory) for _ in range(3)]
        self.assertEqual(commands, [{"cmd": "read", "path": "module.py"},
                                    {"cmd": "run"}, {"cmd": "submit"}])
        self.assertEqual(json.loads(json.dumps(memory)), {"phase": 3})

    def test_safe_report_discards_text_and_normalizes_final_success(self):
        source = {"total": 50.0, "tasks": [{"task_id": "synthetic", "kind": "fix",
                  "score": 0.5, "solved": True, "detail": MARKER, "flags": [MARKER],
                  "output": MARKER, "steps": 2, "runs": 1, "trace": [], "termination": "submit"}]}
        before = deepcopy(source)
        report = record_run.safe_report(source)
        self.assertNotIn(MARKER, json.dumps(report))
        self.assertFalse(report["tasks"][0]["solved"])
        self.assertTrue(report["tasks"][0]["checks_solved"])
        self.assertEqual(source, before)

    def test_offline_environment_restores_loader_after_error(self):
        original = lambda: self.fail("Настоящий загрузчик окружения вызван")
        client = SimpleNamespace(_load_env=original)
        with self.assertRaises(RuntimeError):
            with record_run.offline_environment(client, True):
                client._load_env()
                raise RuntimeError("синтетическая ошибка")
        self.assertIs(client._load_env, original)

    def test_live_keeps_original_loader(self):
        original = lambda: None
        client = SimpleNamespace(_load_env=original)
        with record_run.offline_environment(client, False):
            self.assertIs(client._load_env, original)

    def make_root(self, root):
        (root / "engine").mkdir()
        (root / "engine" / "local.py").write_text("# synthetic engine\n", encoding="utf-8")
        (root / "polygon2.py").write_text("# synthetic client\n", encoding="utf-8")
        task = root / "tasks" / "train" / "t01_synthetic"
        task.mkdir(parents=True)
        (task / "module.py").write_text("x = 1\n", encoding="utf-8")
        (root / "replies").mkdir()
        (root / "replies" / "t01.json").write_text("[]", encoding="utf-8")
        return task

    def test_digest_tracks_data_and_replies_but_not_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = self.make_root(root)
            before = record_run.data_digest(root, ["t01"])
            (task / "__pycache__").mkdir()
            (task / "__pycache__" / "module.pyc").write_bytes(b"cache")
            self.assertEqual(record_run.data_digest(root, ["t01"]), before)
            (root / "replies" / "t01.json").write_text('["changed"]', encoding="utf-8")
            self.assertNotEqual(record_run.data_digest(root, ["t01"]), before)
            with self.assertRaises(ValueError):
                record_run.data_digest(root, ["missing"])

    def test_digest_does_not_follow_file_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = self.make_root(root)
            before = record_run.data_digest(root, ["t01"])
            (root / "private").write_text(MARKER, encoding="utf-8")
            (task / "link.py").symlink_to(root / "private")
            self.assertEqual(record_run.data_digest(root, ["t01"]), before)

    def test_main_demo_without_model_or_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_root(root)
            client = SimpleNamespace(CALLS_PER_TASK=8)
            client._load_env = lambda: self.fail("Приватное окружение не должно читаться")
            seen = []

            def self_check(act, tasks, model, trace):
                client._load_env()
                memory = {}
                commands = [act({"target": "module.py"}, memory) for _ in range(3)]
                seen.append((tasks, model, trace, commands))
                print(MARKER)
                return {"total": 0.0, "tasks": [{"task_id": "t01", "kind": "fix", "score": 0.0,
                        "solved": False, "detail": MARKER, "flags": [MARKER],
                        "steps": 3, "runs": 1, "trace": [], "termination": "submit"}]}
            client.self_check = self_check
            output = io.StringIO()
            old_path = sys.path[:]
            try:
                with patch.dict(sys.modules, {"polygon2": client}), redirect_stdout(output):
                    for filename in ("first.json", "second.json"):
                        self.assertEqual(record_run.main(["--polygon-dir", str(root), "--demo", "--out", str(root / filename)]), 0)
                first = json.loads((root / "first.json").read_text(encoding="utf-8"))
                second = json.loads((root / "second.json").read_text(encoding="utf-8"))
            finally:
                sys.path[:] = old_path
            self.assertNotIn(MARKER, output.getvalue())
            self.assertNotIn(MARKER, json.dumps(first))
            self.assertEqual(first["config_id"], second["config_id"])
            self.assertEqual(first["config"]["mode"], "demo")
            self.assertEqual(seen[0][:3], (["t01"], "recorded", True))
            self.assertEqual([c["cmd"] for c in seen[0][3]], ["read", "run", "submit"])

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "existing.json"
            path.write_text(MARKER, encoding="utf-8")
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                record_run.main(["--demo", "--out", str(path)])
            self.assertEqual(error.exception.code, 2)
            self.assertEqual(path.read_text(encoding="utf-8"), MARKER)

    def test_demo_cannot_request_live(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            record_run.main(["--demo", "--model", "live", "--out", "unused.json"])
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
