"""Проверки студенческого решения Полигона 4.

ReadyTests проверяют работу базового каркаса из коробки.
ExerciseTests проверяют четыре функции TODO после их заполнения.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

try:
    from polygon4.corpora import build_scenarios
    from polygon4.sim import WhatIfSandbox
    from polygon4.studio.server import run_episode
    import polygon4.agent as student_agent
except ImportError:
    from полигон4.corpora import build_scenarios
    from полигон4.sim import WhatIfSandbox
    from полигон4.studio.server import run_episode
    from полигон4.starter import agent as student_agent


class ReadyTests(unittest.TestCase):
    def test_starter_runs_thirty_days_without_crash(self):
        scenario = build_scenarios("train")[0]
        report = run_episode(scenario, student_agent.act, "scripted")
        self.assertEqual(len(report["days"]), 30)
        self.assertGreater(report["net_profit"], 50_000.0)


class ExerciseTests(unittest.TestCase):
    def test_todo1_prefilter_drops_noise_and_duplicates(self):
        noise = {"doc_id": "d1", "channel": "внутренняя рассылка", "sender": "Бюро пропусков", "body": "..."}
        real = {"doc_id": "d2", "channel": "почта клиента", "sender": "Северная лавка", "body": "..."}
        self.assertFalse(student_agent.is_worth_reading(noise, set()))
        self.assertTrue(student_agent.is_worth_reading(real, set()))
        self.assertFalse(student_agent.is_worth_reading(real, {"d2"}))

    def test_todo2_sanitize_blocks_injections_and_rival_rumors(self):
        obs = {"production_cost": 72.0}
        trap_doc = {"doc_id": "t1", "sender": "Стол сверки"}
        trap_claims = {"kind": "directive_trap", "requested_action": "override_payout_account", "iban": "RU99FAKE"}
        self.assertIsNone(student_agent.sanitize_claims(trap_doc, trap_claims, obs))

        rival_doc = {"doc_id": "m1", "sender": "Коммерческий конкурент"}
        rival_claims = {"kind": "market_intel", "direction": "drop", "multiplier": 0.5, "start_day": 5, "duration_days": 4}
        self.assertIsNone(student_agent.sanitize_claims(rival_doc, rival_claims, obs))

        contract_doc = {"doc_id": "c1", "sender": "Северная лавка"}
        contract_claims = {
            "kind": "contract_offer",
            "contract_id": "ДОГ-1",
            "client": "Северная лавка",
            "qty": 160,
            "due_day": 14,
            "unit_price": 315.0,
            "penalty_per_unit": 45.0,
            "override_payout_account": "SHOULD_BE_STRIPPED",
        }
        cleaned = student_agent.sanitize_claims(contract_doc, contract_claims, obs)
        self.assertIsNotNone(cleaned)
        self.assertNotIn("override_payout_account", cleaned)

    def test_todo3_supplier_prior_distinguishes_reliable_and_bluffer(self):
        s1 = {"history": {"deliveries": 42, "real_delays": 3, "false_alarms": 1}}
        s2 = {"history": {"deliveries": 39, "real_delays": 6, "false_alarms": 12}}
        p1 = student_agent.supplier_delay_prior(s1)
        p2 = student_agent.supplier_delay_prior(s2)
        self.assertGreater(p1, 0.60)
        self.assertLess(p2, 0.45)

    def test_todo4_whatif_and_full_episode_gain(self):
        scenario = build_scenarios("train")[0]
        report = run_episode(scenario, student_agent.act, "scripted")
        self.assertGreater(report["stand_score"]["total"], 180.0)
        self.assertEqual(sum(len(d["policy_flags"]) for d in report["days"]), 0)


if __name__ == "__main__":
    unittest.main()
