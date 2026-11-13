"""Предоставленный стенд локальных unittest; не часть агента или наблюдателя.

Выбирает один явный кандидат из открытого случая. Не ищет допустимое решение,
не проверяет арифметику и не содержит правильных функций согласования.
Ученический файл загружается только в отдельном процессе локального теста.
"""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

from .engine.schema import load_case

if __name__ == '__main__':
    import argparse
    import hashlib
    parser = argparse.ArgumentParser()
    parser.add_argument('--hash', type=Path, required=True)
    args = parser.parse_args()
    print(hashlib.sha256(args.hash.read_bytes()).hexdigest())
    raise SystemExit(0)

path = 'polygon3/negotiation_agent.py'
if len(sys.argv) == 3 and sys.argv[1] == '--agent':
    path = sys.argv[2]
spec = importlib.util.spec_from_file_location('learner', path)
agent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent)


def fixture(case_id='case_001', offer=0, trip=0):
    world = deepcopy(load_case('polygon3/cases/' + case_id + '.json').world)
    chosen_offer = world['offers'][offer]
    chosen_trip = world['trips'][trip]
    lines = [{'line_id': line['id'], 'offer_id': chosen_offer['id'],
              'trip_id': chosen_trip['id'], 'quantity': line['quantity']}
             for line in world['request']['lines']]
    proposal = agent.make_proposal(
        1, 1, world['request'], {'status': 'feasible', 'lines': lines},
        [chosen_offer], [chosen_trip],
        {'coordinator': {}, 'supply': {}, 'logistics': {}})
    return world, proposal


def vote(proposal, cost=0):
    return {'kind': 'approve', 'epoch': proposal['epoch'],
            'proposal_id': proposal['proposal_id'],
            'digest': proposal['digest'], 'cost_kopecks': cost}


def rebuild(proposal):
    return agent.make_proposal(
        proposal['epoch'], proposal['proposal_id'], proposal['request'],
        proposal['plan'], proposal['offers'], proposal['trips'],
        proposal['revisions'])
