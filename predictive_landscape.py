"""Bounded spatial scenario exploration, not a physical Feynman simulation.

Positions encode time and two caller-chosen layout axes. Geometry never supplies
probability or evidence. All branches remain conditional on their supplied links.
"""
from __future__ import annotations

import json
import math
import uuid
import hashlib
from emergence_capture import capture_emergence
from self_inquiry_journey import begin_intuition_inquiry


def _text(value, name, limit=1000):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f'{name} must be nonempty text, at most {limit} characters')
    return value


def _rows(value, name, limit):
    if not isinstance(value, list) or len(value) > limit:
        raise ValueError(f'{name} must be a list of at most {limit} items')
    return value


def explore_landscape(payload):
    """Preserve a surfaced signal and enumerate bounded paths without ranking truth."""
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    if len(encoded.encode('utf-8')) > 32768:
        raise ValueError('landscape exceeds 32 KiB')
    data = json.loads(encoded)
    capture = capture_emergence(data.get('signal'), modality='mixed',
        source='chosen_predictive_landscape', context_references=data.get('context_references', []))
    unit = _text(data.get('time_unit'), 'time_unit', 80)
    nodes = {}
    for raw in _rows(data.get('nodes', []), 'nodes', 32):
        identifier = _text(raw.get('id'), 'node id', 100)
        if identifier in nodes:
            raise ValueError('duplicate node id')
        position = raw.get('position')
        if not isinstance(position, list) or len(position) != 3 or any(
            isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) for v in position
        ):
            raise ValueError('position requires three finite coordinates')
        nodes[identifier] = {'id': identifier, 'description': _text(raw.get('description'), 'description'),
            'position': position, 'countercheck': raw.get('countercheck'), 'kind': raw.get('kind', 'possibility')}
        if nodes[identifier]['kind'] not in {'observation', 'possibility', 'attention_horizon'}:
            raise ValueError('unknown node kind')
        if raw.get('countercheck') is not None:
            _text(raw['countercheck'], 'countercheck')
    links = []
    adjacency = {key: [] for key in nodes}
    for raw in _rows(data.get('links', []), 'links', 64):
        start, end = raw.get('from'), raw.get('to')
        if start not in nodes or end not in nodes or nodes[start]['position'][0] >= nodes[end]['position'][0]:
            raise ValueError('links must join existing nodes forward in time')
        refs = [_text(r, 'evidence reference', 200) for r in _rows(raw.get('evidence_references', []), 'evidence', 8)]
        link = {'from': start, 'to': end, 'condition': _text(raw.get('condition'), 'condition'),
                'evidence_references': refs, 'causal_status': 'supplied_evidence_unverified' if refs else 'hypothetical'}
        adjacency[start].append(link)
        links.append(link)
    start = data.get('start')
    if nodes and start not in nodes:
        raise ValueError('start must identify a node')
    frontier = [([start], [])] if nodes else []
    branches, expanded = [], 0
    while frontier and expanded < 64 and len(branches) < 16:
        path, chain = frontier.pop(0)
        expanded += 1
        outgoing = adjacency[path[-1]]
        if not outgoing or len(path) >= 8:
            node = nodes[path[-1]]
            complete = not outgoing
            testable = bool(complete and chain and node['kind'] == 'possibility' and node['countercheck'])
            branches.append({'path': path, 'conditions': [link['condition'] for link in chain],
                'endpoint': node['description'], 'complete': complete,
                'status': 'conditional_testable_candidate' if testable else 'attention_or_unresolved',
                'countercheck': node['countercheck'], 'probability': None,
                'evidence_references': [ref for link in chain for ref in link['evidence_references']],
                'hypothetical_links': [link for link in chain if not link['evidence_references']]})
        else:
            frontier.extend((path + [link['to']], chain + [link]) for link in outgoing)
    chosen = data.get('investigate_node')
    inquiry = None
    if chosen is not None:
        if chosen not in nodes or not nodes[chosen]['countercheck']:
            raise ValueError('chosen investigation needs a known node and countercheck')
        inquiry = begin_intuition_inquiry(nodes[chosen]['description'],
            question='What evidence would support or challenge this possibility?',
            countercheck=nodes[chosen]['countercheck'], trigger_references=[capture['capture_id']])
    return {'schema': 'ina.predictive_landscape/V1', 'capture': capture,
        'axes': ['time_offset', 'layout_y', 'layout_z'], 'time_unit': unit,
        'nodes': list(nodes.values()), 'links': links, 'branches': branches,
        'truncated': bool(frontier) or any(not b['complete'] for b in branches),
        'inquiry': inquiry, 'probabilities_calibrated': False,
        'geometry_is_evidence': False, 'instructions_authorized': False,
        'automatic_execution': False, 'writes_memory': False,
        'outcome_status': 'not_observed', 'rendered_3d_view': False}


def revise_landscape(payload, previous=None):
    """Create a copy-on-revision model; provenance links are not authenticated seals."""
    if previous is not None:
        serialized = json.dumps(previous, ensure_ascii=False, allow_nan=False)
        if len(serialized.encode('utf-8')) > 524288 or previous.get('schema') != 'ina.landscape_model/V1':
            raise ValueError('a bounded landscape model is required')
        if 'signal' in payload or 'context_references' in payload or 'time_unit' in payload:
            raise ValueError('original signal, provenance and time units cannot be rewritten; create a new map')
        allowed = {'nodes', 'links', 'start', 'investigate_node'}
        if set(payload) - allowed:
            raise ValueError('unsupported map revision fields')
        specification = {**previous['specification'], **payload}
        # Investigation is a one-shot choice, not something replayed on every revision.
        if 'investigate_node' not in payload:
            specification.pop('investigate_node', None)
        parent = {'model_id': previous['model_id'],
                  'sha256': hashlib.sha256(serialized.encode('utf-8')).hexdigest()}
    else:
        specification = dict(payload)
        parent = None
    result = explore_landscape(specification)
    if previous is not None:
        result['capture'] = json.loads(json.dumps(previous['landscape']['capture']))
        if result['inquiry']:
            result['inquiry']['trigger_references'] = [result['capture']['capture_id']]
    model = {'schema': 'ina.landscape_model/V1', 'model_id': uuid.uuid4().hex,
            'parent': parent, 'specification': json.loads(json.dumps(specification)),
            'landscape': result, 'update_trigger': 'explicit_choice_or_new_evidence',
            'scheduled_work': False, 'lineage_is_authenticated': False}
    if len(json.dumps(model, ensure_ascii=False).encode('utf-8')) > 524288:
        raise ValueError('expanded model exceeds 512 KiB; split the map')
    return model
