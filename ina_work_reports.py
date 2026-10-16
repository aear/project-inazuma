"""Chosen private reports and bounded Python diagnostics; no publication or execution."""
from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import PurePosixPath
import re
from typing import Any, Mapping
import uuid

SCHEMA = 'ina.work_report/V1'
KINDS = ('drawing', 'music', 'experiment', 'capability', 'bug')
MAX_SOURCE_BYTES = 65536
MAX_NODES = 4096
MAX_FINDINGS = 32


def _text(value: Any, limit: int = 4000) -> str:
    if not isinstance(value, str) or len(value) > limit:
        raise ValueError(f'text must be a string of at most {limit} characters')
    return value


def _strings(values: Any, limit: int = 16) -> list[str]:
    if not isinstance(values, (tuple, list)) or len(values) > limit:
        raise ValueError(f'expected at most {limit} explicit strings')
    return [_text(value, 1000) for value in values]


def prepare_work_report(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Separate authored observations, supplied measurements and interpretations.

    Reference metadata is retained without reading its target. No private
    rationale is required and native wording is accepted unchanged.
    """
    if not isinstance(payload, Mapping):
        raise ValueError('report must be a mapping')
    kind = payload.get('kind')
    if kind not in KINDS:
        raise ValueError('kind must be drawing, music, experiment, capability or bug')
    title = _text(payload.get('title'), 200)
    observations = _strings(payload.get('observations', []))
    if not title.strip() or not any(item.strip() for item in observations):
        raise ValueError('title and at least one observation are required')
    # Explicit omission is different from claiming there is no uncertainty.
    if 'uncertainties' not in payload:
        raise ValueError('uncertainties must be explicitly supplied (may be empty)')
    evidence = payload.get('evidence', [])
    measurements = payload.get('measurements', [])
    if not isinstance(evidence, list) or len(evidence) > 16:
        raise ValueError('at most 16 evidence references')
    if not isinstance(measurements, list) or len(measurements) > 16:
        raise ValueError('at most 16 supplied measurements')
    refs = []
    for row in evidence:
        if not isinstance(row, Mapping):
            raise ValueError('each evidence item must be a mapping')
        refs.append({key: _text(row.get(key, ''), 1000)
                     for key in ('reference', 'sha256', 'origin', 'independence_group')})
        if not refs[-1]['reference'].strip():
            raise ValueError('evidence reference must not be empty')
    metrics = []
    for row in measurements:
        if not isinstance(row, Mapping):
            raise ValueError('each measurement must be a mapping')
        # Textual values preserve units, unavailable results and native notation;
        # this function never converts them to a global capability score.
        metric = {key: _text(row.get(key, ''), 1000)
                  for key in ('name', 'value', 'unit', 'method', 'evidence_reference')}
        if not metric['name'].strip() or not metric['method'].strip():
            raise ValueError('each measurement requires name and method')
        metrics.append(metric)
    result = {'schema': SCHEMA, 'report_id': 'report_' + uuid.uuid4().hex,
        'created_at': datetime.now(timezone.utc).isoformat(), 'kind': kind, 'title': title,
        'authorship': 'ina_supplied_content', 'observations': observations,
        'interpretations': _strings(payload.get('interpretations', [])),
        'uncertainties': _strings(payload['uncertainties']), 'evidence': refs,
        'measurements': metrics, 'next_steps': _strings(payload.get('next_steps', [])),
        'reproduction_steps': _strings(payload.get('reproduction_steps', [])),
        'expected': _text(payload.get('expected', '')),
        'actual': _text(payload.get('actual', '')),
        'previous_report': _text(payload.get('previous_report', ''), 1000),
        'evidence_status': 'caller_supplied_not_independently_verified',
        'private': True, 'delivered': False, 'promotion_authorized': False}
    if len(json.dumps(result, ensure_ascii=False).encode()) > 60000:
        raise ValueError('report exceeds 60000 UTF-8 bytes')
    return result


def save_work_report(fs, payload: Mapping[str, Any]) -> dict[str, Any]:
    report = prepare_work_report(payload)
    # New identity per revision; never overwrite the previous report or its asset.
    sections = ["# " + report['title'], "Private " + report['kind'] + " report", report['report_id']]
    for field in ('observations', 'interpretations', 'uncertainties', 'reproduction_steps', 'next_steps'):
        sections.extend(["## " + field.replace('_', ' ').title(),
                         "\n".join(report[field]) if report[field] else '(none supplied)'])
    for field in ('expected', 'actual'):
        if report[field]: sections.extend(["## " + field.title(), report[field]])
    sections.append('Evidence and measurements below are caller-supplied, not independently verified.')
    body = '\n\n'.join(sections) + '\n\n<!-- ina-report-data -->\n' + json.dumps(report, ensure_ascii=False, indent=2)
    path = fs.write_note(report['report_id'] + '.md', body, folder='Reports')
    return {'report': report, 'path': str(path), 'private': True, 'delivered': False}


def read_work_report(fs, report_id: str) -> dict[str, Any]:
    if not isinstance(report_id, str) or not re.fullmatch(r'report_[0-9a-f]{32}', report_id):
        raise ValueError('an explicit report ID is required')
    data = fs.read('ina_hdd', f'Reports/{report_id}.md', max_bytes=200000)
    if len(data) > 200000:
        raise ValueError('stored report exceeds read budget')
    report = json.loads(data.decode('utf-8').rpartition('\n<!-- ina-report-data -->\n')[2])
    if report.get('schema') != SCHEMA or report.get('report_id') != report_id:
        raise ValueError('stored report identity mismatch')
    return {'report': report, 'private': True, 'delivered': False,
            'integrity': 'identity_checked_not_authenticated'}


def inspect_python_source(source: str, *, source_reference: str = '') -> dict[str, Any]:
    """Inspect supplied source, never import it. Findings are review candidates."""
    source = _text(source, MAX_SOURCE_BYTES)
    raw = source.encode('utf-8')
    if len(raw) > MAX_SOURCE_BYTES:
        raise ValueError('source exceeds 64 KiB')
    result = {'schema': 'ina.bug_source_inspection/V1',
        'source_reference': _text(source_reference, 1000), 'sha256': hashlib.sha256(raw).hexdigest(),
        'findings': [], 'complete': True, 'executed': False,
        'assurance': 'static candidates only; absence of findings does not establish correctness',
        'not_checked': ['runtime behavior', 'imports', 'other languages', 'security assurance']}
    def add(kind, node, explanation):
        if len(result['findings']) < MAX_FINDINGS:
            result['findings'].append({'kind': kind, 'line': getattr(node, 'lineno', 1),
                'explanation': explanation, 'status': 'candidate', 'source_sha256': result['sha256']})
        else:
            result['complete'] = False
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        result['findings'].append({'kind': 'syntax_error', 'line': error.lineno,
            'explanation': error.msg, 'status': 'parser_rejected', 'source_sha256': result['sha256']})
        return result
    except (RecursionError, MemoryError):
        result['complete'] = False
        result['not_checked'].append('parser resource limit')
        return result
    # Explicit traversal ceiling, no recursive walk or evaluator.
    stack, admitted = [tree], 0
    while stack and admitted < MAX_NODES:
        node = stack.pop(); admitted += 1
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            add('bare_except', node, 'May hide cancellation or process-exit signals; inspect intended recovery.')
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for default in list(node.args.defaults) + list(node.args.kw_defaults):
                if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                    add('mutable_default', node, 'Literal default is shared across calls; test for unwanted state retention.')
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names = set()
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    if item.name in names:
                        add('repeated_definition', item, 'Same direct scope defines this name again; may be intentional overload or replacement.')
                    names.add(item.name)
        # Source size bounds the parser allocation; traversal is separately bounded.
        stack.extend(ast.iter_child_nodes(node))
    result['complete'] = result['complete'] and not stack
    result['nodes_inspected'] = admitted
    return result


def inspect_code_file(fs, relative_path: str) -> dict[str, Any]:
    """One explicit Python file on the existing read-only code drive; no scan."""
    path = PurePosixPath(_text(relative_path, 500))
    blocked = {'.git', '.private', '.secrets', 'AI_Children', 'venv', '.venv', '__pycache__'}
    if path.is_absolute() or '..' in path.parts or any(
        part.startswith('.') or part in blocked or part.startswith('AI_Children') for part in path.parts
    ) or path.suffix != '.py':
        raise ValueError('choose one non-private Python source path on the code drive')
    data = fs.read('code', str(path), max_bytes=MAX_SOURCE_BYTES)
    if len(data) > MAX_SOURCE_BYTES:
        raise ValueError('source exceeds 64 KiB; provide a chosen excerpt instead')
    return inspect_python_source(data.decode('utf-8'), source_reference='code:' + str(path))
