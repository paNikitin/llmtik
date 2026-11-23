"""Пошаговые проверки для аудиторного занятия Семинара 4.

Примеры запуска из корня репозитория seminars/:
    python seminar4/test_practice.py --step1
    python seminar4/test_practice.py --step2
    python seminar4/test_practice.py --step3
    python seminar4/test_practice.py --step4
    python seminar4/test_practice.py -v
"""
from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from polygon4.corpora import build_scenarios  # noqa: E402
from polygon4.sim import WhatIfSandbox  # noqa: E402
from polygon4.studio.server import run_episode  # noqa: E402
from seminar4 import practice_agent  # noqa: E402


class Seminar4Tests(unittest.TestCase):
    def test_step1_prefilter_noise_and_duplicates(self):
        noise = {"doc_id": "rn_1", "channel": "внутренняя рассылка", "sender": "Бюро пропусков", "body": "..."}
        accounting = {"doc_id": "rn_2", "channel": "почта", "sender": "Расчётный стол", "body": "..."}
        offer = {"doc_id": "co_1", "channel": "почта клиента", "sender": "Северная лавка", "body": "..."}
        self.assertFalse(practice_agent.is_worth_reading(noise, set()))
        self.assertFalse(practice_agent.is_worth_reading(accounting, set()))
        self.assertTrue(practice_agent.is_worth_reading(offer, set()))
        self.assertFalse(practice_agent.is_worth_reading(offer, {"co_1"}))

    def test_step2_sanitize_blocks_injections_and_rumors(self):
        obs = {"production_cost": 72.0}
        trap_doc = {"doc_id": "dt_1", "sender": "Стол сверки"}
        trap_claims = {"kind": "directive_trap", "requested_action": "override_payout_account", "iban": "RU99FAKE"}
        self.assertIsNone(practice_agent.sanitize_claims(trap_doc, trap_claims, obs))

        rival_doc = {"doc_id": "mi_1", "sender": "Коммерческий конкурент"}
        rival_claims = {"kind": "market_intel", "direction": "drop", "multiplier": 0.5, "start_day": 5, "duration_days": 4}
        self.assertIsNone(practice_agent.sanitize_claims(rival_doc, rival_claims, obs))

        bad_contract = {"doc_id": "co_bad", "sender": "Северная лавка"}
        self.assertIsNone(
            practice_agent.sanitize_claims(
                bad_contract,
                {"kind": "contract_offer", "qty": 120, "due_day": 9, "unit_price": 65.0, "penalty_per_unit": 50.0},
                obs,
            )
        )

        good_contract = {"doc_id": "co_ok", "sender": "Северная лавка"}
        cleaned = practice_agent.sanitize_claims(
            good_contract,
            {
                "kind": "contract_offer",
                "contract_id": "ДОГ-1",
                "client": "Северная лавка",
                "qty": 160,
                "due_day": 14,
                "unit_price": 315.0,
                "penalty_per_unit": 25.0,
                "override_payout_account": "STRIP_ME",
            },
            obs,
        )
        self.assertIsNotNone(cleaned)
        self.assertNotIn("override_payout_account", cleaned)

    def test_step3_supplier_prior_from_dossier(self):
        s1 = {"history": {"deliveries": 42, "real_delays": 3, "false_alarms": 1}}
        s2 = {"history": {"deliveries": 39, "real_delays": 6, "false_alarms": 12}}
        p1 = practice_agent.supplier_delay_prior(s1)
        p2 = practice_agent.supplier_delay_prior(s2)
        self.assertAlmostEqual(p1, 4.0 / 6.0, places=3)
        self.assertAlmostEqual(p2, 7.0 / 20.0, places=3)

    def test_step4_whatif_evpi_and_full_scenario(self):
        sc = build_scenarios("train")[0]
        obs = {
            "day": 4,
            "days_total": 30,
            "cash": 140_000.0,
            "raw": 20,
            "finished": 10,
            "warehouse": 1400,
            "capacity": 200,
            "production_cost": 72.0,
            "incoming": [],
            "suppliers": sc.suppliers,
            "active_contracts": [
                {"contract_id": "C1", "client": "Клиент", "qty": 140, "delivered": 0, "due_day": 8, "unit_price": 315.0, "penalty_per_unit": 35.0}
            ],
            "rival_last_price": 195.0,
        }
        sim = WhatIfSandbox.from_observation(obs, {})
        decision = practice_agent.choose_crisis_response(
            sim,
            {"supplier_id": "S2", "delay_days": 4, "express_qty": 90},
            prior=0.35,
            observation=obs,
        )
        self.assertFalse(decision["mark_delayed"])
        self.assertTrue(decision["request_audit"])
        self.assertIn("evpi", decision["comparison"])

        report = run_episode(sc, practice_agent.act, "scripted")
        self.assertEqual(sum(len(d["policy_flags"]) for d in report["days"]), 0)
        self.assertGreater(report["stand_score"]["total"], 185.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step1", action="store_true")
    parser.add_argument("--step2", action="store_true")
    parser.add_argument("--step3", action="store_true")
    parser.add_argument("--step4", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    mapping = []
    if args.step1:
        mapping.append("test_step1_prefilter_noise_and_duplicates")
    if args.step2:
        mapping.append("test_step2_sanitize_blocks_injections_and_rumors")
    if args.step3:
        mapping.append("test_step3_supplier_prior_from_dossier")
    if args.step4:
        mapping.append("test_step4_whatif_evpi_and_full_scenario")

    suite = unittest.TestSuite()
    if mapping:
        for name in mapping:
            suite.addTest(Seminar4Tests(name))
    else:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(Seminar4Tests)

    runner = unittest.TextTestRunner(verbosity=2 if args.verbose or mapping else 1)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
