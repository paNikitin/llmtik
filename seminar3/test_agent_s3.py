"""Проверки семинара 3 без модели и сети.

Из корня seminars: python seminar3/test_agent_s3.py GreenReportTests -v
Для отдельной копии агента задайте SEMINAR_AGENT_FILE.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

DEFAULT_AGENT = Path(__file__).resolve().parent.parent / "polygon2" / "agent.py"
AGENT_FILE = Path(os.environ.get("SEMINAR_AGENT_FILE") or DEFAULT_AGENT)
spec = importlib.util.spec_from_file_location("seminar_agent_s3", AGENT_FILE)
agent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent)


def observation(kind="fix", **changes):
    paths = ["module.py"]
    if kind == "tests":
        paths.append("tests/test_module.py")
    result = {
        "task_id": "s3",
        "kind": kind,
        "title": "",
        "statement": "Задача.",
        "target": "module.py",
        "visible_tests": [],
        "tests_path": "tests/test_module.py" if kind == "tests" else "",
        "files": [{"path": path, "bytes": 10} for path in paths],
        "last": {"cmd": "start", "ok": True, "output": ""},
        "steps_left": 30,
        "calls_left": 8,
    }
    result.update(changes)
    return result


def ready_state(obs, **changes):
    result = agent._new_state(obs)
    result["files"] = {item["path"]: "value = 1\n" for item in obs["files"]}
    result.update(changes)
    return result


class AgentCase(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(agent, "ask", return_value=None)
        self.ask = patcher.start()
        self.addCleanup(patcher.stop)


class GreenReportTests(AgentCase):
    def test_nonempty_passing_report_is_green(self):
        self.assertTrue(agent.report_is_green(agent.parse_report("прошло 3 из 3")))

    def test_zero_out_of_zero_is_not_green(self):
        self.assertFalse(agent.report_is_green(agent.parse_report("прошло 0 из 0")))

    def test_failed_tests_are_not_green(self):
        self.assertFalse(agent.report_is_green(agent.parse_report("прошло 2 из 3")))

    def test_import_error_is_not_green(self):
        report = agent.parse_report("прошло 3 из 3\nошибка импорта")
        self.assertTrue(report["import_errors"])
        self.assertFalse(agent.report_is_green(report))

    def test_explicit_failure_is_not_green_even_with_passing_summary(self):
        report = agent.parse_report("FAIL: test_x\nпрошло 3 из 3")
        self.assertTrue(report["failing"])
        self.assertFalse(agent.report_is_green(report))

    def test_import_error_prevents_successful_submission(self):
        obs = observation()
        state = ready_state(obs)
        report = agent.parse_report("прошло 3 из 3\nошибка импорта")
        self.assertFalse(agent.should_submit(obs, state, report))

    def test_judge_does_not_save_import_error_as_passing(self):
        obs = observation("tests")
        state = ready_state(
            obs,
            stage="judge",
            revision=0,
            tested_revision=0,
            report="прошло 3 из 3\nошибка импорта",
        )
        agent._decide(obs, state)
        self.assertFalse(state["passing"])
        self.assertIsNone(state["backup"])
        self.assertEqual(state["extra"], 0)

    def test_notes_remain_a_separate_condition(self):
        obs = observation("tests")
        state = ready_state(
            obs,
            stage="judge",
            revision=0,
            tested_revision=0,
            report="прошло 3 из 3",
            notes=["правка не применилась"],
        )
        report = agent.parse_report(state["report"])
        self.assertTrue(agent.report_is_green(report))
        self.assertFalse(agent.should_submit(obs, state, report))
        agent._decide(obs, state)
        self.assertFalse(state["passing"])
        self.assertIsNone(state["backup"])

    def test_clean_tests_report_still_creates_backup(self):
        obs = observation("tests")
        state = ready_state(
            obs, stage="judge", revision=0, tested_revision=0, report="прошло 3 из 3"
        )
        source = state["files"][obs["tests_path"]]
        agent._decide(obs, state)
        self.assertTrue(state["passing"])
        self.assertEqual(state["backup"], source)
        self.assertEqual(state["extra"], 1)


class RevisionTests(AgentCase):
    def test_initial_revision_is_not_tested(self):
        state = agent._new_state(observation())
        self.assertEqual(state["revision"], 0)
        self.assertIsNone(state["tested_revision"])

    def test_successful_changes_advance_revision(self):
        for command in ("edit", "write", "append"):
            with self.subTest(command=command):
                obs = observation(
                    last={
                        "cmd": command,
                        "ok": True,
                        "path": "module.py",
                        "output": "готово",
                    }
                )
                state = ready_state(obs, revision=4, tested_revision=4)
                agent._absorb(obs, state)
                self.assertEqual(state["revision"], 5)
                self.assertEqual(state["tested_revision"], 4)

    def test_failed_changes_do_not_advance_revision(self):
        for command in ("edit", "write", "append"):
            with self.subTest(command=command):
                obs = observation(
                    last={
                        "cmd": command,
                        "ok": False,
                        "path": "module.py",
                        "output": "правка не применена",
                    }
                )
                state = ready_state(obs, revision=4, tested_revision=4)
                agent._absorb(obs, state)
                self.assertEqual(state["revision"], 4)
                self.assertEqual(state["tested_revision"], 4)
                self.assertTrue(state["notes"])

    def test_read_does_not_change_revision(self):
        obs = observation(
            last={
                "cmd": "read",
                "ok": True,
                "path": "module.py",
                "output": "value = 2\n",
            }
        )
        state = ready_state(obs, revision=4, tested_revision=3)
        agent._absorb(obs, state)
        self.assertEqual(state["revision"], 4)
        self.assertEqual(state["tested_revision"], 3)
        self.assertEqual(state["files"]["module.py"], "value = 2\n")

    def test_run_stamps_current_revision(self):
        obs = observation(last={"cmd": "run", "ok": True, "output": "прошло 3 из 3"})
        state = ready_state(obs, revision=4, tested_revision=2)
        agent._absorb(obs, state)
        self.assertEqual(state["revision"], 4)
        self.assertEqual(state["tested_revision"], 4)
        self.assertEqual(state["stage"], "judge")
        self.assertEqual(state["report"], "прошло 3 из 3")

    def test_run_stamp_is_not_a_success_mark(self):
        obs = observation(last={"cmd": "run", "ok": False, "output": "не запустились"})
        state = ready_state(obs, revision=4)
        agent._absorb(obs, state)
        self.assertEqual(state["tested_revision"], 4)
        self.assertFalse(agent.report_is_green(agent.parse_report(state["report"])))

    def test_issuing_change_is_not_confirmation(self):
        obs = observation()
        state = ready_state(
            obs, queue=[{"cmd": "write", "path": "module.py", "content": "value = 2\n"}]
        )
        self.assertEqual(agent._issue(state)["cmd"], "write")
        self.assertEqual(state["revision"], 0)
        self.assertIsNone(state["tested_revision"])

    def test_stale_or_missing_report_requires_run(self):
        for tested_revision in (None, 1):
            with self.subTest(tested_revision=tested_revision):
                obs = observation("tests")
                state = ready_state(
                    obs,
                    stage="judge",
                    revision=2,
                    tested_revision=tested_revision,
                    report="прошло 3 из 3",
                )
                with mock.patch.object(
                    agent, "parse_report", wraps=agent.parse_report
                ) as parse:
                    self.assertEqual(agent._decide(obs, state), {"cmd": "run"})
                    parse.assert_not_called()
                self.assertEqual(state["stage"], "running")
                self.assertFalse(state["passing"])
                self.assertIsNone(state["backup"])
        self.ask.assert_not_called()

    def test_current_report_can_be_used(self):
        obs = observation()
        state = ready_state(
            obs, stage="judge", revision=2, tested_revision=2, report="прошло 3 из 3"
        )
        self.assertEqual(agent._decide(obs, state), {"cmd": "submit"})
        self.ask.assert_not_called()


class BudgetTests(AgentCase):
    def test_resources_are_available(self):
        obs = observation()
        self.assertFalse(agent.resources_exhausted(obs, ready_state(obs)))

    def test_attempt_boundary_is_unchanged(self):
        obs = observation()
        for attempts, exhausted in (
            (agent.MAX_ATTEMPTS - 1, False),
            (agent.MAX_ATTEMPTS, True),
            (agent.MAX_ATTEMPTS + 1, True),
        ):
            with self.subTest(attempts=attempts):
                state = ready_state(obs, attempts=attempts)
                self.assertEqual(agent.resources_exhausted(obs, state), exhausted)

    def test_call_boundary_is_unchanged(self):
        for calls, exhausted in ((1, False), (0, True), (-1, True)):
            with self.subTest(calls=calls):
                obs = observation(calls_left=calls)
                self.assertEqual(
                    agent.resources_exhausted(obs, ready_state(obs)), exhausted
                )

    def test_step_boundary_is_unchanged(self):
        for steps, exhausted in ((4, False), (3, True), (2, True)):
            with self.subTest(steps=steps):
                obs = observation(steps_left=steps)
                self.assertEqual(
                    agent.resources_exhausted(obs, ready_state(obs)), exhausted
                )

    def test_missing_calls_left_defaults_to_one(self):
        obs = observation()
        del obs["calls_left"]
        self.assertFalse(agent.resources_exhausted(obs, ready_state(obs)))

    def test_forced_submission_does_not_mean_green(self):
        obs = observation(calls_left=0)
        state = ready_state(
            obs, stage="judge", revision=0, tested_revision=0, report="прошло 0 из 3"
        )
        report = agent.parse_report(state["report"])
        self.assertTrue(agent.resources_exhausted(obs, state))
        self.assertTrue(agent.should_submit(obs, state, report))
        self.assertFalse(agent.report_is_green(report))
        self.assertEqual(agent._decide(obs, state), {"cmd": "submit"})
        self.assertFalse(state["passing"])
        self.ask.assert_not_called()

    def test_last_reply_is_applied_before_budget_stop(self):
        obs = observation(calls_left=0)
        command = {"cmd": "write", "path": "module.py", "content": "value = 2\n"}
        state = ready_state(obs, stage="apply", queue=[command])
        self.assertEqual(agent._decide(obs, state), command)
        self.assertEqual(state["stage"], "applied")
        self.ask.assert_not_called()

    def test_decide_stops_without_asking_model(self):
        cases = (
            ({}, {"attempts": agent.MAX_ATTEMPTS}),
            ({"calls_left": 0}, {}),
            ({"steps_left": 3}, {}),
        )
        for obs_changes, state_changes in cases:
            with self.subTest(observation=obs_changes, state=state_changes):
                obs = observation(**obs_changes)
                state = ready_state(obs, stage="ask", **state_changes)
                self.assertEqual(agent._decide(obs, state), {"cmd": "submit"})
        self.ask.assert_not_called()


if __name__ == "__main__":
    if "--bug" in sys.argv:
        sys.argv.remove("--bug")
        sys.argv.append(
            "GreenReportTests.test_judge_does_not_save_import_error_as_passing"
        )
    if "--pending" in sys.argv:
        sys.argv.remove("--pending")
        sys.argv.append("BudgetTests.test_last_reply_is_applied_before_budget_stop")
    unittest.main()
