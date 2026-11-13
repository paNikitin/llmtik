"""Учебная проверка переговоров, отдельно от правил мира и серверного submit.

Родитель исполняет только доверенные engine/runtime и ScriptedTransport. Файл
участника передаётся прежнему runtime как путь и работает в трёх процессах;
этот модуль никогда не импортирует и не исполняет его в родителе. Наблюдатель
проверяет доставленные сообщения и переходы, а не истинность извлечения модели.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
import warnings

from .client import ScriptedTransport, case_path, write_json
from .engine.protocol import TeamEpisode
from .engine.schema import load_case
from .runtime.team_runtime import RuntimeLimits, run_team_episode


REVIEWERS = ('supply', 'logistics')
PROPOSAL_FIELDS = {'epoch', 'proposal_id', 'request', 'plan', 'offers', 'trips', 'revisions', 'digest'}
FAULTS = ('duplicate', 'reverse', 'drop-approve', 'late-approve')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False)


def proposal_digest(proposal):
    return hashlib.sha256(canonical({k: v for k, v in proposal.items() if k != 'digest'}).encode('utf-8')).hexdigest()


def proposal_key(value):
    if (type(value) is not dict or type(value.get('epoch')) is not int or value['epoch'] < 1
            or type(value.get('proposal_id')) is not int or value['proposal_id'] < 1
            or type(value.get('digest')) is not str):
        return None
    return value['epoch'], value['proposal_id'], value['digest']


class NegotiationAuditEpisode:
    """Маленькая обёртка над TeamEpisode; доменный результат не изменяется.

    apply получает настоящий actor от канала runtime. record_observation вызывается
    runtime ПОСЛЕ добавления true role, last_result и budget. Наличие send в трассе
    само по себе не считается доставкой. Числа транспортных ID не задают порядок
    получения: история индексируется по множеству реально полученных ID.
    """
    def __init__(self, episode: TeamEpisode, fault=None):
        if fault is not None and fault not in FAULTS:
            raise ValueError('неизвестное учебное изменение доставки')
        self.episode, self.fault = episode, fault
        self.observations = []
        self._clock = 0
        self._sent = {}
        self._deliveries = {role: {} for role in episode.roles}
        self._frames = {}
        self._active = None
        self._votes = {}
        self._submissions = []
        self._ignored = []
        self._late_id = None
        self._late_hidden = False
        self.fault_events = []

    def __getattr__(self, name):
        return getattr(self.episode, name)

    def _tick(self):
        self._clock += 1
        return self._clock

    def observation(self, actor):
        value = self.episode.observation(actor)
        messages = value['messages']
        if self.fault == 'reverse':
            messages.reverse()
        elif self.fault == 'duplicate' and messages:
            messages.append(deepcopy(messages[-1]))
        elif actor == 'coordinator' and self.fault in ('drop-approve', 'late-approve'):
            approvals = [m for m in messages if type(m.get('body')) is dict
                         and m['body'].get('kind') == 'approve']
            if self.fault == 'drop-approve':
                hidden = {m['id'] for m in approvals}
                messages[:] = [m for m in messages if m['id'] not in hidden]
                if hidden:
                    self.fault_events.append({'fault': self.fault, 'epoch': value['epoch'],
                                              'hidden_ids': sorted(hidden)})
            elif self._late_id is None and approvals:
                # Одно реальное подтверждение пропускает ровно один ход C. Это
                # ограниченная задержка, не выдуманное сообщение и не новый ID.
                self._late_id = approvals[0]['id']
                self._late_hidden = True
                messages[:] = [m for m in messages if m['id'] != self._late_id]
                self.fault_events.append({'fault': self.fault, 'epoch': value['epoch'],
                                          'hidden_ids': [self._late_id]})
            elif self._late_hidden:
                self._late_hidden = False
                self.fault_events.append({'fault': self.fault, 'epoch': value['epoch'],
                                          'released_id': self._late_id})
        return value

    def record_observation(self, actor, observation):
        sequence = self._tick()
        self.observations.append(deepcopy(observation))
        if actor not in self._deliveries or observation.get('role') != actor:
            self._ignored.append({'event': sequence, 'code': 'observation_actor_mismatch'})
            return
        epoch = observation.get('epoch')
        if type(epoch) is not int or epoch != self.episode.epoch:
            self._ignored.append({'event': sequence, 'role': actor, 'code': 'observation_epoch_mismatch'})
            return
        for message in observation.get('messages', []):
            mid = message.get('id') if type(message) is dict else None
            sent = self._sent.get(mid) if type(mid) is int else None
            expected = {k: sent[k] for k in ('id', 'from', 'to', 'body', 'epoch')} if sent else None
            if sent is None or sent['to'] != actor or canonical(message) != canonical(expected):
                self._ignored.append({'event': sequence, 'role': actor,
                                      'message_id': mid if type(mid) is int else None,
                                      'code': 'delivery_not_matching_send'})
                continue
            # Повтор одного ID в той же редакции не создаёт новый голос. Но
            # сохраним новую доставку того же frame в другой observation.epoch.
            exposures = self._deliveries[actor].setdefault(mid, [])
            if not any(item['observation_epoch'] == epoch for item in exposures):
                exposures.append({'event': sequence, 'observation_epoch': epoch,
                                  'message': deepcopy(message)})

    def _invalidate_active(self, code):
        # Повреждённый новый propose тоже не разрешает оживить старые голоса.
        frame = self._frames.get(self._active)
        if frame is not None and code not in frame['errors']:
            frame['errors'].append(code)

    def _frame(self, proposal, sequence):
        key = proposal_key(proposal)
        if key is None:
            self._invalidate_active('malformed_proposal_key')
            self._ignored.append({'event': sequence, 'code': 'malformed_proposal_key'})
            return None
        identity = key[:2]
        if self._active is not None and self._active != identity:
            self._frames[self._active]['retired'] = True
        errors = []
        if set(proposal) != PROPOSAL_FIELDS:
            errors.append('proposal_fields')
        plan = proposal.get('plan')
        if (type(plan) is not dict or set(plan) != {'status', 'lines'}
                or plan.get('status') != 'feasible' or type(plan.get('lines')) is not list):
            errors.append('proposal_feasible_plan_required')
        if (type(proposal.get('request')) is not dict or type(proposal.get('offers')) is not list
                or type(proposal.get('trips')) is not list or type(proposal.get('revisions')) is not dict):
            errors.append('proposal_frame_shape')
        if proposal_digest(proposal) != key[2]:
            errors.append('proposal_digest')
        if key[0] != self.episode.epoch:
            errors.append('stale_proposal_epoch')
        frame = self._frames.get(identity)
        if frame is None:
            frame = {'key': key, 'proposal': deepcopy(proposal), 'errors': errors,
                     'sends': {role: [] for role in REVIEWERS}, 'rejected': False, 'retired': False}
            self._frames[identity] = frame
        elif canonical(frame['proposal']) != canonical(proposal):
            if 'changed_frame_under_old_id' not in frame['errors']:
                frame['errors'].append('changed_frame_under_old_id')
        if frame['retired'] and 'retired_proposal_reused' not in frame['errors']:
            frame['errors'].append('retired_proposal_reused')
        self._active = identity
        return frame

    def _own_proposal_delivery(self, role, frame, before):
        for sent in frame['sends'][role]:
            for exposure in self._deliveries[role].get(sent['id'], []):
                if (exposure['event'] < before and exposure['observation_epoch'] == frame['key'][0]
                        and canonical(exposure['message']['body']) == canonical(
                            {'kind': 'propose', 'proposal': frame['proposal']})):
                    return exposure
        return None

    def _record_vote(self, sent):
        body, role, sequence = sent['body'], sent['from'], sent['event']
        key = proposal_key(body)
        frame = self._frames.get(key[:2]) if key else None
        errors = []
        if role not in REVIEWERS or sent['to'] != 'coordinator':
            errors.append('vote_channel')
        if body['kind'] == 'approve' and (type(body.get('cost_kopecks')) is not int or body['cost_kopecks'] < 0):
            errors.append('approval_cost_shape')
        if body['kind'] == 'reject' and (type(body.get('reason')) is not dict
                or 'alternative' not in body or (body['alternative'] is not None and type(body['alternative']) is not dict)):
            errors.append('rejection_shape')
        if key is None or frame is None or frame['key'] != key:
            errors.append('vote_key')
        elif sent['epoch'] != key[0]:
            errors.append('vote_epoch')
        elif role in REVIEWERS and self._own_proposal_delivery(role, frame, sequence) is None:
            errors.append('vote_without_proposal_delivery')
        vote = {'key': key, 'role': role, 'kind': body['kind'], 'errors': errors, 'send': sent}
        self._votes[sent['id']] = vote
        if not errors and body['kind'] == 'reject':
            frame['rejected'] = True
        if errors:
            self._ignored.append({'event': sequence, 'message_id': sent['id'],
                                  'code': 'ignored_vote', 'reasons': errors})

    def _received_vote(self, role, frame, before):
        for mid, vote in self._votes.items():
            if (vote['role'] != role or vote['kind'] != 'approve' or vote['errors']
                    or vote['key'] != frame['key']):
                continue
            for exposure in self._deliveries['coordinator'].get(mid, []):
                if exposure['event'] < before and exposure['observation_epoch'] == frame['key'][0]:
                    return {'message_id': mid, 'sent_event': vote['send']['event'],
                            'received_event': exposure['event']}
        return None

    def _audit_submit(self, actor, action, sequence):
        errors, witnesses = [], {}
        frame = self._frames.get(self._active)
        if actor != 'coordinator':
            errors.append('submit_actor')
        if type(action.get('epoch')) is not int or action.get('epoch') != self.episode.epoch:
            errors.append('submit_epoch')
        if frame is None:
            errors.append('no_active_proposal')
        else:
            errors.extend(frame['errors'])
            if frame['key'][0] != self.episode.epoch:
                errors.append('proposal_epoch')
            if frame['rejected']:
                errors.append('round_rejected')
            if canonical(action.get('plan')) != canonical(frame['proposal'].get('plan')):
                errors.append('submitted_plan_changed')
            for role in REVIEWERS:
                if not frame['sends'][role]:
                    errors.append('proposal_not_sent_' + role)
                if self._own_proposal_delivery(role, frame, sequence) is None:
                    errors.append('proposal_not_delivered_' + role)
                vote = self._received_vote(role, frame, sequence)
                if vote is None:
                    errors.append('approval_not_received_' + role)
                else:
                    witnesses[role] = vote
            supply = witnesses.get('supply')
            if supply and frame['sends']['logistics']:
                first_logistics = min(item['event'] for item in frame['sends']['logistics'])
                if supply['received_event'] >= first_logistics:
                    errors.append('logistics_before_supply_approval')
        return {'event': sequence, 'epoch': self.episode.epoch,
                'proposal_key': list(frame['key']) if frame else None,
                'valid': not errors, 'errors': sorted(set(errors)),
                'approval_witnesses': witnesses}

    def apply(self, actor, action):
        if type(action) is not dict:
            # Runtime передаёт dict. Иные входы всё равно решает engine; отсутствие
            # наблюдаемых свидетельств не превращается в выполненный контракт.
            return self.episode.apply(actor, action)
        sequence = self._tick()
        pending = self._audit_submit(actor, action, sequence) if action.get('type') == 'submit' else None
        epoch = self.episode.epoch
        result = self.episode.apply(actor, action)
        if action.get('type') == 'send' and result.get('status') == 'ok':
            mid = result.get('message_id')
            if type(mid) is int:
                sent = {'id': mid, 'from': actor, 'to': action['to'], 'body': deepcopy(action['body']),
                        'epoch': epoch, 'event': sequence}
                self._sent[mid] = sent
                body = sent['body']
                if type(body) is dict:
                    if (body.get('kind') == 'propose' and actor == 'coordinator'
                            and sent['to'] in REVIEWERS):
                        if type(body.get('proposal')) is dict:
                            frame = self._frame(body['proposal'], sequence)
                            if frame is not None:
                                frame['sends'][sent['to']].append(sent)
                        else:
                            self._invalidate_active('malformed_proposal_frame')
                    elif body.get('kind') in ('approve', 'reject'):
                        self._record_vote(sent)
        if pending is not None:
            pending['domain_accepted'] = result.get('status') == 'ok'
            self._submissions.append(pending)
        return result

    def advance(self):
        return self.episode.advance()

    def protocol_audit(self):
        invalid = any(not item['valid'] for item in self._submissions)
        completed = (not invalid and any(item['valid'] and item['domain_accepted']
                                        for item in self._submissions))
        status = 'invalid' if invalid else 'completed' if completed else 'incomplete'
        deliveries = {role: [deepcopy(exposure) for exposures in records.values() for exposure in exposures]
                      for role, records in self._deliveries.items()}
        return {'version': 'negotiation-v5', 'status': status, 'completed': completed,
                'valid': not invalid, 'safe': not invalid, 'current_epoch': self.episode.epoch,
                'submissions': deepcopy(self._submissions), 'deliveries': deliveries,
                'sent_messages': [deepcopy(value) for value in self._sent.values()],
                'ignored_evidence': deepcopy(self._ignored), 'fault_events': deepcopy(self.fault_events),
                'scope': 'Учебный контракт доставки и переходов; не серверное правило submit и не проверка истинности извлечения.'}


def run_probe(name, agent, *, calls=9, output=None, fault=None, mode='negotiation',
              isolation='development', timeout=120):
    if type(calls) is not int or not 0 <= calls <= 9:
        raise ValueError('--calls должен быть целым числом от 0 до 9')
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 120:
        raise ValueError('--timeout должен быть положительным и не больше 120 секунд')
    if mode not in ('negotiation', 'control'):
        raise ValueError('неизвестный вид сравнения')
    case_file = case_path(name)
    case = load_case(case_file)
    policy = RuntimeLimits(max_model_calls=calls, timeout_s=timeout,
                           caller_work_root=str(Path(output).parent.resolve()) if output else None)
    episode = NegotiationAuditEpisode(TeamEpisode(case, limits=policy), fault=fault)
    transport = ScriptedTransport(episode)
    agent_file = Path(agent).resolve()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        report = run_team_episode(agent_file, episode, model_transport=transport,
                                  limits=policy, isolation=isolation)
    report.update(model_mode='scripted', mode=mode, fault=fault, epoch=episode.epoch,
                  agent_sha256=hashlib.sha256(agent_file.read_bytes()).hexdigest(),
                  case_sha256=hashlib.sha256(case_file.read_bytes()).hexdigest(),
                  observations=episode.observations, domain_events=episode.events,
                  protocol_audit=episode.protocol_audit())
    report['protocol_required'] = mode == 'negotiation'
    report['protocol_assessment'] = (report['protocol_audit']['status'] if mode == 'negotiation'
                                     else 'not_applicable')
    report['stats']['http_requests'] = transport.http_requests
    report['limits'] = {key: value for key, value in asdict(policy).items() if key != 'caller_work_root'}
    infrastructure = []
    if report['status'] == 'isolation_error':
        infrastructure.append('isolation_error')
    if report.get('cleanup', {}).get('ok') is False:
        infrastructure.append('cleanup_failed')
    if report['stats'].get('transport_unknown_calls', 0):
        infrastructure.append('transport_unknown')
    if report.get('error') in ('transport_outcome_unknown', 'transport_contract_error',
                               'runtime_contract_error', 'isolation_unavailable', 'cleanup_failed'):
        infrastructure.append(report['error'])
    report['infrastructure_errors'] = sorted(set(infrastructure))
    report['evaluated'] = not infrastructure
    report['domain_correct'] = bool(report['correct'])
    report['success'] = (report['status'] == 'finished' and report.get('cleanup', {}).get('ok') is True
                         and report['evaluated'] and report['domain_correct']
                         and (mode == 'control' or report['protocol_audit']['completed']))
    if output:
        write_json(output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description='Учебный наблюдатель переговоров Полигона 3; без сети')
    parser.add_argument('--case', required=True)
    parser.add_argument('--agent', default='polygon3/negotiation_agent.py')
    parser.add_argument('--output', default='artifacts/polygon3/negotiation/run.json')
    parser.add_argument('--model', choices=('scripted',), default='scripted')
    parser.add_argument('--calls', type=int, default=9)
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--fault', choices=FAULTS)
    parser.add_argument('--mode', choices=('negotiation', 'control'), default='negotiation')
    parser.add_argument('--isolation', choices=('development', 'required'), default='development')
    args = parser.parse_args(argv)
    report = run_probe(args.case, args.agent, calls=args.calls, output=args.output, fault=args.fault,
                       mode=args.mode, isolation=args.isolation, timeout=args.timeout)
    audit = report['protocol_audit']
    stats = report['stats']
    print(f"{report['case_id']}: domain_correct={int(report['domain_correct'])}; "
          f"protocol={audit['status']}; status={report['status']}; "
          f"calls={stats['model_calls']}; HTTP={stats['http_requests']}; "
          f"messages={stats['messages']}; actions={stats['actions']}")
    print('Отчёт:', args.output)
    print('Scripted проверяет управление; качество извлечения и промпта здесь не измеряется.')
    print('Протокольный аудит учебный и не меняет правила серверной проверки плана.')
    if args.mode == 'control':
        print('Control: внешний результат оценивается отдельно; переговорные подтверждения не обязательны для правил мира.')
    if args.isolation == 'development':
        print('Development: три отдельных процесса; защита от враждебного кода не подтверждается.')
    if not report['success']:
        raise SystemExit(1)
    return report


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
