"""Read-only monitoring dashboards for Ina's operational state.

Collectors deliberately use compact state files, indexes, and filesystem metadata.
They never load the large memory map, symbol-word store, or conversation archives.
"""
from __future__ import annotations

import json
from collections import Counter
import os
import time
import tkinter as tk
from datetime import datetime, timezone
from pathlib import Path
from tkinter import ttk
from typing import Any, Callable

from model_manager import get_inastate, load_config
from neural_taxonomy import count_node_types
try:
    from logic_memory_store import graph_counts as durable_logic_counts
except Exception:  # pragma: no cover
    durable_logic_counts = None


from storage_layout import fast_runtime_path

MAX_JSON_BYTES = 32 * 1024 * 1024
MAX_LINE_COUNT_BYTES = 32 * 1024 * 1024


def _child_memory() -> Path:
    child = load_config().get('current_child', 'Inazuma_Yagami')
    return Path('AI_Children') / str(child) / 'memory'


def _safe_json(path: Path, default: Any = None) -> Any:
    try:
        if not path.is_file() or path.stat().st_size > MAX_JSON_BYTES:
            return default
        with path.open('r', encoding='utf-8') as handle:
            return json.load(handle)
    except (OSError, ValueError, TypeError):
        return default


def _emotion_map_summary(path: Path) -> tuple[int | None, str, str]:
    """Read compact metadata first and never deserialize an oversized map."""
    try:
        stat = path.stat()
    except OSError:
        return 0, "0 symbols", "missing"

    status_path = path.with_name("emotion_symbol_map_status.json")
    status = _safe_json(status_path, {})
    if (
        isinstance(status, dict)
        and int(status.get("source_size", -1)) == int(stat.st_size)
        and int(status.get("source_mtime_ns", -1)) == int(stat.st_mtime_ns)
    ):
        count = max(0, int(status.get("symbol_count", 0) or 0))
        backend = "emotion database" if status.get("backend") == "sqlite" else "emotion map"
        return count, f"{count:,} symbols", backend

    if stat.st_size <= MAX_JSON_BYTES:
        payload = _safe_json(path, {})
        symbols = payload.get("symbols", []) if isinstance(payload, dict) else []
        count = len(symbols) if isinstance(symbols, (list, dict)) else 0
        return count, f"{count:,} symbols", "emotion map"

    return None, f"large JSON · {_size(stat.st_size)}", "metadata pending"


def _size(value: int | float | None) -> str:
    amount = float(value or 0)
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if amount < 1024.0 or unit == 'TB':
            return f'{amount:.0f} {unit}' if unit == 'B' else f'{amount:.1f} {unit}'
        amount /= 1024.0
    return f'{amount:.1f} TB'


def _modified(path: Path) -> str:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).astimezone().strftime('%Y-%m-%d %H:%M')
    except OSError:
        return '—'


def _age(timestamp: Any) -> str:
    if timestamp in (None, '', 0):
        return '—'
    try:
        if isinstance(timestamp, (int, float)):
            then = float(timestamp)
        else:
            then = datetime.fromisoformat(str(timestamp).replace('Z', '+00:00')).timestamp()
        seconds = max(0, time.time() - then)
        if seconds < 60:
            return f'{int(seconds)}s ago'
        if seconds < 3600:
            return f'{int(seconds // 60)}m ago'
        if seconds < 86400:
            return f'{int(seconds // 3600)}h ago'
        return f'{int(seconds // 86400)}d ago'
    except (ValueError, TypeError, OverflowError, OSError):
        return str(timestamp)[:24]


def _state_text(value: Any) -> str:
    if value is None:
        return 'not reported'
    if isinstance(value, bool):
        return 'yes' if value else 'no'
    if isinstance(value, float):
        return f'{value:.3f}'
    if isinstance(value, (list, dict)):
        return f'{len(value)} items'
    return str(value)


def _count_lines(path: Path) -> int | None:
    try:
        if not path.is_file() or path.stat().st_size > MAX_LINE_COUNT_BYTES:
            return None
        count = 0
        with path.open('rb') as handle:
            for _ in handle:
                count += 1
        return count
    except OSError:
        return None


def _file_row(label: str, path: Path, state: str = 'available') -> tuple[str, str, str, str, str]:
    try:
        return (label, _size(path.stat().st_size), state, _modified(path), str(path))
    except OSError:
        return (label, '—', 'missing', '—', str(path))


def _continuity_percent(value: Any) -> str:
    try:
        return f'{100.0 * float(value):.1f}%'
    except (TypeError, ValueError):
        return 'unmeasured'


def _continuity_delta(value: Any) -> str:
    try:
        points = 100.0 * float(value)
    except (TypeError, ValueError):
        return 'new baseline'
    return f'{points:+.1f} pp'

def _continuity() -> tuple[list[tuple[str, str]], list[tuple[str, str, str, str, str]]]:
    """Read only the compact continuity report and bounded boot core."""
    base = _child_memory()
    report_path = base / 'continuity' / 'continuity_map.json'
    core_path = base / 'continuity' / 'continuity_core_map.json'
    report = _safe_json(report_path, {})
    core = _safe_json(core_path, {})
    report = report if isinstance(report, dict) else {}
    core = core if isinstance(core, dict) else {}
    overall = report.get('overall_continuity')
    overall_delta = report.get('overall_delta')
    coverage = report.get('evidence_coverage')
    boot_status = str(core.get('status') or 'unavailable')
    dimensions = report.get('dimensions', {})
    dimensions = dimensions if isinstance(dimensions, dict) else {}
    recommendations = core.get('recommendations', [])
    recommendations = recommendations if isinstance(recommendations, list) else []
    dimension_anchors = core.get('dimension_anchors', {})
    dimension_anchors = dimension_anchors if isinstance(dimension_anchors, dict) else {}

    rows = []
    overall_state = 'continuity summary'
    try:
        if overall is not None and float(overall) < 0.6:
            overall_state += ' · highlight'
    except (TypeError, ValueError):
        overall_state += ' · highlight'
    rows.append((
        'Overall continuity',
        f'{_continuity_percent(overall)} · {_continuity_delta(overall_delta)}',
        overall_state,
        _age(report.get('updated')),
        json.dumps(report, indent=2, default=str),
    ))
    for name, detail in dimensions.items():
        if not isinstance(detail, dict):
            continue
        state = str(detail.get('state') or 'unmeasured')
        row_state = f'continuity · {state}'
        if state in {'weak', 'unmeasured'}:
            row_state += ' · highlight'
        anchors = dimension_anchors.get(name, [])
        anchors = anchors if isinstance(anchors, list) else []
        related = [item for item in recommendations if isinstance(item, dict) and item.get('dimension') == name]
        detail_payload = dict(detail)
        detail_payload['dimension'] = name
        detail_payload['boot_anchor_ids'] = anchors
        detail_payload['recommendations'] = related
        evidence = (
            f"{int(detail.get('matched_evidence', 0) or 0)}/"
            f"{max(int(detail.get('previous_evidence', 0) or 0), int(detail.get('current_evidence', 0) or 0))} evidence"
        )
        rows.append((
            str(detail.get('label') or name.replace('_', ' ').title()),
            f"{_continuity_percent(detail.get('score'))} · {_continuity_delta(detail.get('delta'))}",
            f'{row_state} · {evidence}',
            _age(report.get('updated')),
            json.dumps(detail_payload, indent=2, default=str),
        ))
    rows.append((
        'Minimum boot core',
        f"{boot_status} · {len(core.get('anchors', [])) if isinstance(core.get('anchors'), list) else 0} anchors",
        'bounded boot snapshot' + (' · highlight' if boot_status in {'unavailable', 'insufficient'} else ''),
        _age(core.get('generated_at')),
        json.dumps(core, indent=2, default=str),
    ))

    cards = [
        ('Overall', _continuity_percent(overall)),
        ('Change', _continuity_delta(overall_delta)),
        ('Evidence', _continuity_percent(coverage)),
        ('Minimal boot', boot_status),
    ]
    return cards, rows



def _bias() -> tuple[list[tuple[str, str]], list[tuple[str, str, str, str, str]]]:
    """Report bounded recall diversity without treating observation as correction."""
    path = _child_memory() / "continuity" / "memory_relationships.json"
    payload = _safe_json(path, {})
    payload = payload if isinstance(payload, dict) else {}
    latest = payload.get("latest_arbitration")
    latest = latest if isinstance(latest, dict) else {}
    history = payload.get("recall_history")
    history = [item for item in history if isinstance(item, dict)] if isinstance(history, list) else []

    def metric(container: dict[str, Any], key: str, field: str) -> float | None:
        value = container.get(key)
        value = value.get(field) if isinstance(value, dict) else None
        return _unit_level(value)

    def average(key: str, field: str) -> float | None:
        values = [value for item in history if (value := metric(item, key, field)) is not None]
        return sum(values) / len(values) if values else None

    type_diversity = metric(latest, "selected_type_diversity", "score")
    source_diversity = metric(latest, "selected_source_diversity", "score")
    type_dominance = metric(latest, "selected_type_diversity", "dominance")
    source_dominance = metric(latest, "selected_source_diversity", "dominance")
    type_skew = metric(latest, "memory_type_selection_skew", "strength")
    source_skew = metric(latest, "source_selection_skew", "strength")

    rows = []
    specs = (
        ("Recall modality diversity", type_diversity, type_dominance, "selected_type_diversity",
         "How varied the surfaced memory modalities were. Low diversity describes concentration; it does not establish unfairness."),
        ("Recall source diversity", source_diversity, source_dominance, "selected_source_diversity",
         "How varied the surfaced witness stores were. Original traces remain owned by their source stores."),
    )
    for label, diversity, dominance, key, explanation in specs:
        state = "descriptive diversity"
        if dominance is not None and dominance >= 0.75:
            state += " · concentrated · highlight"
        rows.append((
            label, f"{_percent_level(diversity)} diversity · {_percent_level(dominance)} dominant",
            state, _age(latest.get("timestamp")),
            json.dumps({
                "latest": latest.get(key), "rolling_diversity": average(key, "score"),
                "rolling_dominance": average(key, "dominance"), "interpretation": explanation,
            }, indent=2, default=str),
        ))

    for label, skew, key in (
        ("Recall modality selection skew", type_skew, "memory_type_selection_skew"),
        ("Recall source selection skew", source_skew, "source_selection_skew"),
    ):
        detail = latest.get(key) if isinstance(latest.get(key), dict) else {}
        state = "selected share minus candidate share"
        if skew is not None and skew >= 0.25:
            state += " · concentrated · highlight"
        rows.append((
            label, _percent_level(skew), state, _age(latest.get("timestamp")),
            json.dumps({
                "latest": detail, "rolling_strength": average(key, "strength"),
                "interpretation": "A strong delta shows what arbitration amplified or suppressed relative to available candidates. It is evidence for review, not an automatic correction.",
            }, indent=2, default=str),
        ))

    selected_types = latest.get("selected_memory_types")
    selected_types = selected_types if isinstance(selected_types, dict) else {}
    selected_total = max(1, sum(int(value or 0) for value in selected_types.values()))
    for memory_type, count in selected_types.items():
        share = int(count or 0) / selected_total
        rows.append((
            f"Surfaced modality · {memory_type}", _percent_level(share), "latest recall share",
            _age(latest.get("timestamp")), json.dumps({
                "selected": count,
                "available": (latest.get("candidate_memory_types") or {}).get(memory_type, 0),
                "federation_rule": "Continuity coordinates rankings and links; it never rewrites modality traces.",
            }, indent=2, default=str),
        ))
    if not latest:
        rows.append((
            "Recall bias evidence", "not reported", "waiting for a recall action",
            _modified(path), "No recall arbitration has been observed yet. Missing evidence is not treated as zero bias or perfect diversity.",
        ))

    cards = [
        ("Type diversity", _percent_level(type_diversity)),
        ("Source diversity", _percent_level(source_diversity)),
        ("Strongest skew", _percent_level(max(value for value in (type_skew, source_skew) if value is not None)) if any(value is not None for value in (type_skew, source_skew)) else "not reported"),
        ("Recalls sampled", str(len(history))),
    ]
    return cards, rows

def _mind() -> tuple[list[tuple[str, str]], list[tuple[str, str, str, str, str]]]:
    base = _child_memory()
    neural = base / 'neural'
    rows = []
    total_nodes = total_edges = maps = 0
    type_counts: Counter[str] = Counter()
    config = load_config()
    child = str(config.get('current_child', 'Inazuma_Yagami'))
    durable_primary = neural / 'neural_memory_map.json'
    fast_primary = fast_runtime_path(
        child,
        'neural_memory_map.json',
        durable_primary,
        subdir='neural',
        root_keys=('fast_neural_root', 'fast_runtime_root', 'fast_root'),
        config=config,
    )
    neural_paths = sorted(neural.glob('*.json')) if neural.is_dir() else []
    if fast_primary != durable_primary and fast_primary.exists():
        neural_paths = [path for path in neural_paths if path != durable_primary]
        neural_paths.append(fast_primary)
    for path in sorted(neural_paths, key=lambda item: item.name):
        data = _safe_json(path, {})
        nodes = data.get('nodes', data.get('neurons', data.get('node_ids', []))) if isinstance(data, dict) else []
        edges = data.get('edges', data.get('synapses', [])) if isinstance(data, dict) else []
        fallback_network = 'logic' if path.stem == 'logic_neural_map' else 'memory_graph'
        type_counts.update(count_node_types(nodes, fallback_network))
        node_count = len(nodes) if isinstance(nodes, (list, dict)) else 0
        edge_count = len(edges) if isinstance(edges, (list, dict)) else int(data.get('edge_count', 0) or 0) if isinstance(data, dict) else 0
        if node_count or edge_count:
            maps += 1
            total_nodes += node_count
            total_edges += edge_count
            label = 'Logic active projection' if path.stem == 'logic_neural_map' else path.stem.replace('_', ' ').title()
            rows.append((label, f'{node_count:,} nodes · {edge_count:,} links', 'neural map', _modified(path), str(path)))
    for neural_type, count in sorted(type_counts.items()):
        rows.append((
            f'{neural_type.replace("_", " ").title()} neurons',
            f'{count:,} nodes', 'neural type · EEG taxonomy', 'live',
            json.dumps({"type": neural_type, "count": count}, indent=2),
        ))
    logic_counts = durable_logic_counts(child, config) if durable_logic_counts else {}
    durable_logic_entries = int(logic_counts.get('entries', 0) or 0) if isinstance(logic_counts, dict) else 0
    durable_logic_edges = int(logic_counts.get('edges', 0) or 0) if isinstance(logic_counts, dict) else 0
    if durable_logic_entries or durable_logic_edges:
        rows.append((
            'Logic durable graph',
            f'{durable_logic_entries:,} traces · {durable_logic_edges:,} sparse links',
            'logic database', 'live', str(child),
        ))


    vocab_data = _safe_json(base / 'text_vocab.json', {})
    vocab = vocab_data.get('vocab', {}) if isinstance(vocab_data, dict) else {}
    links_data = _safe_json(base / 'text_vocab_links.json', {})
    links = links_data.get('links', []) if isinstance(links_data, dict) else []
    link_count = len(links) if isinstance(links, (list, dict)) else 0
    linked_words = set()
    if isinstance(links, list):
        for link in links:
            if isinstance(link, dict):
                for key in ('word', 'source', 'target', 'english', 'generated_word'):
                    if link.get(key):
                        linked_words.add(str(link[key]).lower())
    elif isinstance(links, dict):
        linked_words.update(str(key).lower() for key in links)
    word_count = len(vocab) if isinstance(vocab, dict) else 0
    mapped_ratio = (100.0 * len(linked_words) / word_count) if word_count else 0.0
    average_links = (link_count / word_count) if word_count else 0.0
    text_policy = config.get('text_memory_policy') if isinstance(config.get('text_memory_policy'), dict) else {}
    vocab_limit = int(text_policy.get('vocab_limit', 25000) or 25000)
    evaluated_count = int(links_data.get('evaluated_count', len(linked_words)) or 0) if isinstance(links_data, dict) else len(linked_words)
    remaining = int(links_data.get('remaining', max(0, word_count - evaluated_count)) or 0) if isinstance(links_data, dict) else 0
    queue_by_source = links_data.get('queue_by_source', {}) if isinstance(links_data, dict) else {}
    queue_by_source = queue_by_source if isinstance(queue_by_source, dict) else {}
    last_batch = links_data.get('last_batch', {}) if isinstance(links_data, dict) else {}
    last_batch = last_batch if isinstance(last_batch, dict) else {}
    raw_batch_mode = str(last_batch.get('mode') or 'not_reported')
    batch_mode = {
        'new_and_revisit': 'new + revisit',
        'not_reported': 'not reported',
    }.get(raw_batch_mode, raw_batch_mode.replace('_', ' '))
    queue_detail = ' · '.join(
        f"{str(name).replace('_', ' ').title()} {int(count or 0):,}"
        for name, count in sorted(queue_by_source.items(), key=lambda item: (-int(item[1] or 0), str(item[0])))
    ) or ('none queued' if remaining == 0 else 'source not yet reported')
    emotion_map_path = base / 'emotion_symbol_map.json'
    emotion_map_count, emotion_map_label, emotion_map_state = _emotion_map_summary(emotion_map_path)
    vocab_status = 'cap reached' if word_count >= vocab_limit else f'cap {vocab_limit:,}'
    rows.extend([
        ('Observed vocabulary', f'{word_count:,} words · {vocab_status}', 'language', _age(vocab_data.get('updated') if isinstance(vocab_data, dict) else None), str(base / 'text_vocab.json')),
        ('English mappings', f'{len(linked_words):,} linked · {evaluated_count:,} evaluated · {remaining:,} queued', 'language', _modified(base / 'text_vocab_links.json'), str(base / 'text_vocab_links.json')),
        ('Mapping queue by source', queue_detail, 'language queue', _modified(base / 'text_vocab_links.json'), str(base / 'text_vocab_links.json')),
        ('Last mapping pass', f"{batch_mode} · {int(last_batch.get('new_mappings', 0) or 0):,} new · {int(last_batch.get('revisited_mappings', 0) or 0):,} revisited", 'language queue', _modified(base / 'text_vocab_links.json'), str(base / 'text_vocab_links.json')),
        ('Average links per word', f'{average_links:.2f}', 'language', _modified(base / 'text_vocab_links.json'), str(base / 'text_vocab_links.json')),
        ('Emotion map', emotion_map_label, emotion_map_state, _modified(emotion_map_path), str(emotion_map_path)),
    ])
    emotions = get_inastate('emotion_snapshot') or {}
    values = emotions.get('values', {}) if isinstance(emotions, dict) else {}
    strongest = sorted(values.items(), key=lambda item: abs(float(item[1] or 0)), reverse=True)[:5] if isinstance(values, dict) else []
    for name, value in strongest:
        rows.append((str(name).replace('_', ' ').title(), f'{float(value):+.3f}', 'emotion', _age(emotions.get('timestamp')), json.dumps(emotions, indent=2)))
    emotion_card = f'{emotion_map_count:,}' if emotion_map_count is not None else 'large'
    cards = [('Neural maps', str(maps)), ('Nodes', f'{total_nodes:,}'), ('Links', f'{total_edges:,}'), ('Vocabulary', f'{word_count:,}'), ('Emotion map', emotion_card)]
    return cards, rows


def _world() -> tuple[list[tuple[str, str]], list[tuple[str, str, str, str, str]]]:
    data = _safe_json(Path('world_positions.json'), {})
    entities = data.get('entities', {}) if isinstance(data, dict) else {}
    doors = data.get('doors', {}) if isinstance(data, dict) else {}
    rows = []
    now = time.time()
    highlights = 0
    for entity_id, entity in entities.items() if isinstance(entities, dict) else []:
        pos = entity.get('position', []) if isinstance(entity, dict) else []
        position = ', '.join(f'{float(v):.1f}' for v in pos[:3]) if isinstance(pos, list) else '—'
        last_seen = entity.get('last_seen') if isinstance(entity, dict) else None
        stale = isinstance(last_seen, (int, float)) and now - last_seen > 300
        state = 'stale position' if stale else str(entity.get('role', 'entity'))
        highlights += int(stale)
        rows.append((str(entity.get('name') or entity_id), position, state, _age(last_seen), json.dumps(entity, indent=2)))
    for door_id, is_open in doors.items() if isinstance(doors, dict) else []:
        state = 'OPEN' if is_open else 'closed'
        highlights += int(bool(is_open))
        rows.append((str(door_id).replace('_', ' ').title(), state, 'door · highlight' if is_open else 'door', _modified(Path('world_positions.json')), json.dumps({'id': door_id, 'open': is_open}, indent=2)))
    plan = _safe_json(Path('ina_house_plan.json'), {})
    devices = plan.get('devices', {}) if isinstance(plan, dict) else {}
    for key, device in devices.items() if isinstance(devices, dict) else []:
        rows.append((str(device.get('name') or key) if isinstance(device, dict) else str(key), 'configured', 'device', _modified(Path('ina_house_plan.json')), json.dumps(device, indent=2)))
    cards = [('Entities', str(len(entities))), ('Doors open', str(sum(bool(v) for v in doors.values()) if isinstance(doors, dict) else 0)), ('Devices', str(len(devices))), ('Highlights', str(highlights))]
    return cards, rows


def _memory() -> tuple[list[tuple[str, str]], list[tuple[str, str, str, str, str]]]:
    base = _child_memory()
    specs = [
        ('Memory map index', base / 'memory_map.sqlite'),
        ('Reflections', base / 'reflection_journal.jsonl'),
        ('Dream log', base / 'dream_log.json'),
        ('Self questions', base / 'self_questions.json'),
        ('GitHub exploration queue', base / 'github_outbox.jsonl'),
        ('GitHub submission state', base / 'github_submission_state.json'),
        ('GitHub authentication health', base / 'github_auth_health.json'),
        ('Fragment integrity', base / 'fragment_integrity.json'),
        ('Deep recall', base / 'deep_recall_state.json'),
    ]
    rows = []
    total = 0
    for label, path in specs:
        try:
            total += path.stat().st_size
        except OSError:
            pass
        count = _count_lines(path) if path.suffix == '.jsonl' else None
        state = f'{count:,} indexed entries' if count is not None else 'available'
        rows.append(_file_row(label, path, state))
    integrity = get_inastate('fragment_integrity') or {}
    guard = get_inastate('memory_guard') or {}
    cards = [('Indexed storage', _size(total)), ('Reflections', str(_count_lines(base / 'reflection_journal.jsonl') or 0)), ('Memory guard', _state_text(guard.get('level') if isinstance(guard, dict) else guard)), ('Integrity', _state_text(integrity.get('status') if isinstance(integrity, dict) else integrity))]
    return cards, rows


def _communication() -> tuple[list[tuple[str, str]], list[tuple[str, str, str, str, str]]]:
    base = _child_memory()
    social = _safe_json(base / 'social_map.json', [])
    contact = get_inastate('last_heard_contact') or {}
    flush = get_inastate('discord_outbox_flush') or {}
    speaking = get_inastate('currently_speaking')
    bridge = base / 'discord_bridge.lock'
    runtime_status_path = base / 'runtime_services.json'
    runtime_status = _safe_json(runtime_status_path, {})
    services = runtime_status.get('services') if isinstance(runtime_status, dict) else {}
    services = services if isinstance(services, dict) else {}
    world_state = services.get('world_server') if isinstance(services.get('world_server'), dict) else {}
    discord_state = services.get('discord_bridge') if isinstance(services.get('discord_bridge'), dict) else {}
    supervisor_state = runtime_status.get('status') if isinstance(runtime_status, dict) else None
    rows = [
        ('Runtime supervisor', _state_text(supervisor_state), 'service manager', _modified(runtime_status_path), json.dumps(runtime_status, indent=2)),
        ('World server', _state_text(world_state.get('status')), 'supervised service', _age(world_state.get('updated_at')), json.dumps(world_state, indent=2)),
        ('Discord bridge', _state_text(discord_state.get('status')), 'supervised service', _age(discord_state.get('updated_at')), json.dumps(discord_state, indent=2)),
        ('Discord process lock', 'lock present' if bridge.exists() else 'not present', 'process evidence', _modified(bridge), str(bridge)),
        ('Last heard contact', _state_text(contact.get('name') or contact.get('display_name') if isinstance(contact, dict) else contact), 'conversation', _age(contact.get('timestamp') if isinstance(contact, dict) else None), json.dumps(contact, indent=2) if isinstance(contact, dict) else str(contact)),
        ('Outbox flush', _state_text(flush.get('status') if isinstance(flush, dict) else flush), 'delivery', _age(flush.get('timestamp') if isinstance(flush, dict) else None), json.dumps(flush, indent=2) if isinstance(flush, dict) else str(flush)),
        ('Speaking now', _state_text(speaking), 'voice', 'live state', str(speaking)),
        _file_row('Typed outbox', base / 'typed_outbox.jsonl', 'recent messages'),
    ]
    cards = [('Contacts', str(len(social) if isinstance(social, list) else len(social) if isinstance(social, dict) else 0)), ('Discord', _state_text(discord_state.get('status'))), ('World', _state_text(world_state.get('status'))), ('Speaking', 'yes' if speaking else 'no'), ('Last contact', _age(contact.get('timestamp') if isinstance(contact, dict) else None))]
    return cards, rows

def _unit_level(value: Any) -> float | None:
    """Return a bounded scalar without turning missing evidence into zero."""
    try:
        level = float(value)
    except (TypeError, ValueError):
        return None
    if not (level == level):
        return None
    return max(0.0, min(1.0, level))


def _urge_level(payload: Any, key: str = 'level') -> float | None:
    if isinstance(payload, dict):
        return _unit_level(payload.get(key))
    return _unit_level(payload)


def _percent_level(value: float | None) -> str:
    return 'not reported' if value is None else f'{100.0 * value:.0f}%'


def _typing_evidence_rows(
    state: dict[str, Any],
    type_payload: dict[str, Any],
) -> list[tuple[str, str, str, str, str]]:
    """Keep distinct explanations for quiet text behaviour visibly separate."""
    rows = []
    typing_intent = state.get('typing_contact_intent')
    typing_intent = typing_intent if isinstance(typing_intent, dict) else {}
    expression_intent = state.get('text_expression_intent')
    expression_intent = expression_intent if isinstance(expression_intent, dict) else {}
    typed_payload = state.get('typed_contact_payload')
    typed_payload = typed_payload if isinstance(typed_payload, dict) else {}
    drivers = type_payload.get('drivers')
    drivers = drivers if isinstance(drivers, dict) else {}

    supplied_text = typed_payload.get('text')
    candidates = typing_intent.get('candidates')
    candidates = candidates if isinstance(candidates, dict) else {}
    candidate_present = bool(
        isinstance(supplied_text, str)
        or candidates.get('symbols')
        or candidates.get('word')
    )
    content_value = 'candidate observed' if candidate_present else 'not observed'
    content_detail = (
        "A text or symbolic candidate is currently available. This says that content exists; "
        "it does not prove that Ina wants to send it."
        if candidate_present else
        "No candidate text or symbol sequence is currently reported. That is not equivalent to "
        "'nothing to say': content may be absent, unformed, inaccessible, or simply unreported."
    )
    rows.append((
        'Typing · content to express', content_value, 'separate evidence · not a conclusion',
        _age(typing_intent.get('timestamp')), content_detail,
    ))

    clarity = _unit_level(drivers.get('clarity'))
    fuzziness = _unit_level(drivers.get('fuzziness'))
    if clarity is None and fuzziness is None:
        uncertainty = None
    else:
        uncertainty = (
            (1.0 - (clarity if clarity is not None else 0.5))
            + (fuzziness if fuzziness is not None else 0.5)
        ) / 2.0
    difficulty_signal = uncertainty is not None and uncertainty >= 0.55
    access_value = 'possible difficulty signal' if difficulty_signal else 'not established'
    rows.append((
        'Typing · expression access', access_value, 'ability is distinct from urge',
        _age(type_payload.get('timestamp')),
        "Low clarity or high fuzziness can suggest difficulty forming an expression, but cannot "
        "establish 'can't express it'. No difficulty signal is not proof of full expressive access.",
    ))
    rows.append((
        'Typing · uncertainty', _percent_level(uncertainty), 'distinct from silence and interest',
        _age(type_payload.get('timestamp')),
        json.dumps({'derived_uncertainty': uncertainty, 'clarity': clarity, 'fuzziness': fuzziness}, indent=2),
    ))

    snapshot = state.get('emotion_snapshot')
    snapshot_updated = snapshot.get('timestamp') if isinstance(snapshot, dict) else None
    snapshot = snapshot.get('values') if isinstance(snapshot, dict) and isinstance(snapshot.get('values'), dict) else snapshot
    snapshot = snapshot if isinstance(snapshot, dict) else {}
    try:
        interest = max(-1.0, min(1.0, float(snapshot.get('interest'))))
    except (TypeError, ValueError):
        interest = None
    if interest is None:
        interest_text = 'not reported'
    elif interest <= -0.25:
        interest_text = f'low signal · {interest:+.2f}'
    elif interest >= 0.25:
        interest_text = f'raised signal · {interest:+.2f}'
    else:
        interest_text = f'near neutral · {interest:+.2f}'
    rows.append((
        'Typing · interest', interest_text, 'interest is not response choice',
        _age(snapshot_updated),
        "Interest is shown independently. Low interest does not prove refusal, and silence does not prove low interest.",
    ))

    strategy = str(expression_intent.get('strategy') or '').strip().lower()
    if strategy == 'silence':
        choice_text = 'explicit silence'
        choice_detail = (
            "The current expression intent explicitly selects silence. This is evidence of a present "
            "choice not to express through this turn, not evidence of disinterest or inability."
        )
    elif strategy:
        choice_text = f'explicitly {strategy}'
        choice_detail = f"The current explicit expression strategy is {strategy!r}; it is not a silence inference."
    else:
        choice_text = 'no explicit choice reported'
        choice_detail = (
            "There is no explicit respond/silence choice in runtime state. Urge level alone must not be "
            "used to infer that Ina does or does not want to respond."
        )
    rows.append((
        'Typing · response choice', choice_text, 'choice is distinct from capacity',
        _age(expression_intent.get('created_at')), choice_detail,
    ))
    return rows


def _urges() -> tuple[list[tuple[str, str]], list[tuple[str, str, str, str, str]]]:
    """Read urge signals as invitations, preserving ambiguity about non-action."""
    state = _safe_json(_child_memory() / 'inastate.json', {})
    state = state if isinstance(state, dict) else {}
    config = load_config()
    config = config if isinstance(config, dict) else {}
    arbitration = state.get('meta_arbitration')
    arbitration = arbitration if isinstance(arbitration, dict) else {}

    specs = [
        ('Urge to type', 'urge_to_type', config.get('min_urge_to_type', 0.35)),
        ('Urge to voice', 'urge_to_voice', config.get('min_urge_to_speak', 0.25)),
        ('Urge to move', 'urge_to_move', None),
        ('Urge to seek stability', 'urge_to_seek_stability', None),
    ]
    rows = []
    payloads: dict[str, dict[str, Any]] = {}
    for label, key, raw_threshold in specs:
        raw = state.get(key)
        if key == 'urge_to_voice' and not raw:
            raw = state.get('urge_to_communicate')
        payload = raw if isinstance(raw, dict) else ({'level': raw} if raw is not None else {})
        payloads[key] = payload
        base = _urge_level(payload)
        adjusted = _urge_level(payload, 'adjusted_level')
        effective = adjusted if adjusted is not None else base
        threshold = _unit_level(raw_threshold)
        arbitration_state = payload.get('arbitration')
        arbitration_state = arbitration_state if isinstance(arbitration_state, dict) else {}
        allowed = arbitration_state.get('allowed')
        if effective is None:
            status = 'not reported'
        elif allowed is False:
            status = 'present · held by arbitration'
        elif threshold is not None and effective >= threshold:
            status = 'at or above action threshold'
        elif threshold is not None:
            status = 'below action threshold'
        else:
            status = 'urge signal'
        value = _percent_level(effective)
        if adjusted is not None and base is not None and adjusted != base:
            value = f'{_percent_level(base)} base → {_percent_level(adjusted)} adjusted'
        detail = {
            'meaning': (
                'An urge is pressure or inclination toward an action. Its level does not by itself '
                'explain why the action did or did not happen.'
            ),
            'threshold': threshold,
            'payload': payload,
        }
        rows.append((label, value, status, _age(payload.get('timestamp')), json.dumps(detail, indent=2, default=str)))

    rows.extend(_typing_evidence_rows(state, payloads.get('urge_to_type', {})))
    rows.append((
        'How to read quiet typing', 'five separate questions', 'interpretation guide', 'reference',
        "Nothing to say ≠ can't express it ≠ uncertain ≠ not interested ≠ doesn't want to respond.\n\n"
        "This monitor reports evidence for each question separately and leaves unsupported answers unknown. "
        "Urges describe inclination; they do not reveal a complete reason for silence.",
    ))

    type_level = _urge_level(payloads.get('urge_to_type', {}), 'adjusted_level')
    if type_level is None:
        type_level = _urge_level(payloads.get('urge_to_type', {}))
    voice_level = _urge_level(payloads.get('urge_to_voice', {}), 'adjusted_level')
    if voice_level is None:
        voice_level = _urge_level(payloads.get('urge_to_voice', {}))
    move_level = _urge_level(payloads.get('urge_to_move', {}), 'adjusted_level')
    if move_level is None:
        move_level = _urge_level(payloads.get('urge_to_move', {}))
    cards = [
        ('Type', _percent_level(type_level)),
        ('Voice', _percent_level(voice_level)),
        ('Move', _percent_level(move_level)),
        ('Arbitration', str(arbitration.get('status') or 'not reported')),
    ]
    return cards, rows

def _actions() -> tuple[list[tuple[str, str]], list[tuple[str, str, str, str, str]]]:
    base = _child_memory()
    scheduler = _safe_json(base / 'process_scheduler_state.json', {})
    queue = scheduler.get('queue', []) if isinstance(scheduler, dict) else []
    running = scheduler.get('running', {}) if isinstance(scheduler, dict) else {}
    paint_dir = base / 'paint_sessions'
    paint_sessions = sum(1 for item in paint_dir.iterdir() if item.is_file() or item.is_dir()) if paint_dir.is_dir() else 0
    keys = [
        ('Motor control', 'motor_control_status'), ('Motor feedback', 'motor_feedback'),
        ('Body adjustment', 'motor_body_adjustment'), ('Walk target', 'walk_to_marker_status'),
        ('Movement urge', 'urge_to_move'), ('Paint request', 'paint_request'),
        ('Paint queue', 'paint_command_queue'), ('Exploration', 'exploration_nudge_state'),
    ]
    rows = []
    for label, key in keys:
        value = get_inastate(key)
        timestamp = value.get('timestamp') if isinstance(value, dict) else None
        rows.append((label, _state_text(value.get('status') if isinstance(value, dict) and 'status' in value else value), key.replace('_', ' '), _age(timestamp), json.dumps(value, indent=2, default=str) if isinstance(value, (dict, list)) else str(value)))
    rows.append(('Scheduler', f'{len(queue)} queued · {len(running)} running', 'scheduled work', _age(scheduler.get('updated_at') if isinstance(scheduler, dict) else None), json.dumps({k: scheduler.get(k) for k in ('slot_summary', 'last_decisions', 'planner')}, indent=2, default=str)))
    cards = [('Queued', str(len(queue))), ('Running', str(len(running))), ('Paint sessions', str(paint_sessions)), ('Movement urge', _state_text(get_inastate('urge_to_move')))]
    return cards, rows


def _reports() -> tuple[list[tuple[str, str]], list[tuple[str, str, str, str, str]]]:
    base = _child_memory()
    experience = _safe_json(base / 'experience_archive_state.json', {})
    media = _safe_json(base / 'experience_media_archive_state.json', {})
    run = experience.get('run', {}) if isinstance(experience, dict) else {}
    cumulative = experience.get('cumulative', {}) if isinstance(experience, dict) else {}
    source = experience.get('source', {}) if isinstance(experience, dict) else {}
    history = experience.get('history', []) if isinstance(experience, dict) else []
    media_cumulative = media.get('cumulative', {}) if isinstance(media, dict) else {}
    media_source = media.get('source', {}) if isinstance(media, dict) else {}
    media_history = media.get('history', []) if isinstance(media, dict) else []
    rows = []
    rows.append(('Experience condensation · latest run', f"{int(run.get('archived', 0) or 0):,} retired · {_size(run.get('saved_bytes'))} saved", str(experience.get('status') or 'not run') if isinstance(experience, dict) else 'not run', _age(experience.get('updated_at') if isinstance(experience, dict) else None), json.dumps(experience, indent=2, default=str)))
    rows.append(('Live-media condensation · latest run', f"{int((media.get('run') or {}).get('archived', 0) or 0):,} retired · {_size((media.get('run') or {}).get('saved_bytes'))} saved", str(media.get('status') or 'not run') if isinstance(media, dict) else 'not run', _age(media.get('updated_at') if isinstance(media, dict) else None), json.dumps(media, indent=2, default=str)))
    for label, path in (
        ('Memory retention · latest run', base / 'inastate.json'),
        ('Storage migration', base / 'storage_migration_state.json'),
        ('Memory reconciliation', base / 'reconciliation_state.json'),
    ):
        data = _safe_json(path, {})
        if label.startswith('Memory retention') and isinstance(data, dict):
            data = data.get('human_memory_prune_last_run', {})
        if data:
            rows.append((label, _state_text(data.get('status') if isinstance(data, dict) else data), 'run report', _age(data.get('timestamp') or data.get('updated_at') if isinstance(data, dict) else None), json.dumps(data, indent=2, default=str)))
    metadata = int(source.get('directory_metadata_bytes', 0) or 0) + int(media_source.get('directory_metadata_bytes', 0) or 0)
    cards = [('Files retired', f"{int(cumulative.get('archived', 0) or 0) + int(media_cumulative.get('archived', 0) or 0):,}"), ('Payload saved', _size(int(cumulative.get('saved_bytes', 0) or 0) + int(media_cumulative.get('saved_bytes', 0) or 0))), ('Directory metadata', _size(metadata)), ('Runs retained', str(len(history) + len(media_history)))]
    return cards, rows

def _system() -> tuple[list[tuple[str, str]], list[tuple[str, str, str, str, str]]]:
    base = _child_memory()
    vitals = get_inastate('resource_vitals') or {}
    storage = get_inastate('storage_vitals') or {}
    modules = get_inastate('running_modules') or {}
    heartbeat = get_inastate('runtime_heartbeat') or {}
    rows = []
    for label, key, value in (
        ('Runtime heartbeat', 'runtime_heartbeat', heartbeat),
        ('Running modules', 'running_modules', modules),
        ('Storage health', 'storage_vitals', storage),
        ('Runtime disruption', 'runtime_disruption', get_inastate('runtime_disruption')),
        ('Raw-file manager', 'raw_file_manager_state', get_inastate('raw_file_manager_state')),
    ):
        timestamp = value.get('timestamp') if isinstance(value, dict) else value if key == 'runtime_heartbeat' else None
        rows.append((label, _state_text(value.get('status') if isinstance(value, dict) and 'status' in value else value), key.replace('_', ' '), _age(timestamp), json.dumps(value, indent=2, default=str) if isinstance(value, (dict, list)) else str(value)))
    native_dir = Path('.native')
    if native_dir.is_dir():
        for path in sorted(native_dir.iterdir()):
            if path.is_file():
                rows.append(_file_row(path.name, path, 'native module'))
    migration = _safe_json(base / 'storage_migration_state.json', {})
    if isinstance(migration, dict) and migration:
        progress = 100.0 * float(migration.get('progress', 0.0) or 0.0)
        rows.append((
            'Ina storage migration',
            f"{migration.get('status', 'unknown')} · {progress:.1f}% · {_size(migration.get('bytes_copied'))}",
            'verified NVMe promotion',
            _age(migration.get('updated_at')),
            json.dumps(migration, indent=2, default=str),
        ))
    envelope = vitals.get('resource_envelope') if isinstance(vitals, dict) else {}
    if not isinstance(envelope, dict):
        envelope = {}
    rows.append((
        'Kernel resource envelope',
        'verified' if envelope.get('enforced') else 'UNVERIFIED',
        f"RAM {_size(envelope.get('ram_current_bytes'))} / {_size(envelope.get('kernel_ram_limit_bytes'))}; "
        f"swap {_size(envelope.get('swap_current_bytes'))} / {_size(envelope.get('kernel_swap_limit_bytes'))}",
        _age(vitals.get('timestamp') if isinstance(vitals, dict) else None),
        json.dumps(envelope, indent=2, default=str),
    ))
    services = _safe_json(base / 'runtime_services.json', {})
    service_rows = services.get('services') if isinstance(services, dict) else {}
    service_rows = service_rows if isinstance(service_rows, dict) else {}
    live_services = sum(1 for detail in service_rows.values() if isinstance(detail, dict) and detail.get('status') == 'running')
    rows.append((
        'Supervised runtime services',
        f"{live_services}/{len(service_rows)} running",
        str(services.get('status') or 'unavailable'),
        _age(services.get('updated_at')),
        json.dumps(services, indent=2, default=str),
    ))
    workspace = _safe_json(base / 'virtual_workspace' / 'status.json', {})
    audio = workspace.get('audio') if isinstance(workspace, dict) else {}
    audio = audio if isinstance(audio, dict) else {}
    workspace_state = str(workspace.get('status') or 'unavailable')
    workspace_detail = (
        f"{workspace.get('display') or 'no display'} · "
        f"{'audio isolated' if audio.get('ready') else 'audio unavailable'} · "
        f"{'input enabled' if workspace.get('input_enabled') else 'input disabled'}"
    )
    rows.append((
        'Ina virtual desktop',
        workspace_state,
        workspace_detail + (' · highlight' if workspace_state in {'failed', 'blocked'} else ''),
        _age(workspace.get('updated_at')),
        json.dumps(workspace, indent=2, default=str),
    ))
    rows.extend([_file_row('Runtime state', base / 'inastate.json', 'state store'), _file_row('Scheduler state', base / 'process_scheduler_state.json', 'scheduler')])
    cpu = vitals.get('ina_cpu_percent', 0) if isinstance(vitals, dict) else 0
    memory = vitals.get('ina_ram_bytes', 0) if isinstance(vitals, dict) else 0
    heartbeat_ts = heartbeat.get('timestamp') if isinstance(heartbeat, dict) else heartbeat
    cards = [('CPU', f'{float(cpu or 0):.1f}%'), ('Memory', _size(memory)), ('Modules', str(len(modules) if isinstance(modules, (dict, list)) else 0)), ('Heartbeat', _age(heartbeat_ts))]
    return cards, rows


COLLECTORS: dict[str, Callable[[], tuple[list[tuple[str, str]], list[tuple[str, str, str, str, str]]]]] = {
    'Mind': _mind,
    'Continuity': _continuity,
    'Bias': _bias,
    'Urges': _urges,
    'World': _world,
    'Memory': _memory,
    'Reports': _reports,
    'Communication': _communication,
    'Actions': _actions,
    'System': _system,
}


class MonitoringWindow:
    """Notebook-based, manual-refresh operational monitor."""

    def __init__(self, parent: tk.Misc):
        self.window = tk.Toplevel(parent)
        self.window.title('Ina Monitor')
        self.window.geometry('980x700')
        self.window.minsize(760, 540)
        self.window.transient(parent)
        self.rows: dict[str, list[tuple[str, str, str, str, str]]] = {}
        self.views: dict[str, dict[str, Any]] = {}

        header = ttk.Frame(self.window, padding=(18, 14))
        header.pack(fill=tk.X)
        ttk.Label(header, text='Ina Monitor', style='Title.TLabel').pack(anchor='w')
        ttk.Label(header, text='Read-only operational overview · refreshes only when requested', style='Subtitle.TLabel').pack(anchor='w', pady=(2, 0))

        notebook = ttk.Notebook(self.window, padding=(8, 6))
        notebook.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))
        for name in COLLECTORS:
            self._build_tab(notebook, name)
        self.refresh_all()

    def lift(self) -> None:
        self.window.lift()
        self.window.focus_set()

    def exists(self) -> bool:
        return bool(self.window.winfo_exists())

    def _build_tab(self, notebook: ttk.Notebook, name: str) -> None:
        tab = ttk.Frame(notebook, padding=12)
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(2, weight=1)
        notebook.add(tab, text=name)

        toolbar = ttk.Frame(tab)
        toolbar.grid(row=0, column=0, sticky='ew', pady=(0, 10))
        toolbar.columnconfigure(1, weight=1)
        ttk.Label(toolbar, text='Filter').grid(row=0, column=0, padx=(0, 8))
        filter_var = tk.StringVar()
        entry = ttk.Entry(toolbar, textvariable=filter_var)
        entry.grid(row=0, column=1, sticky='ew')
        filter_var.trace_add('write', lambda *_args, n=name: self._render(n))
        ttk.Button(toolbar, text='Refresh', command=lambda n=name: self.refresh(n)).grid(row=0, column=2, padx=(8, 0))

        cards = ttk.Frame(tab)
        cards.grid(row=1, column=0, sticky='ew', pady=(0, 10))
        for column in range(4):
            cards.columnconfigure(column, weight=1)

        body = ttk.Panedwindow(tab, orient=tk.VERTICAL)
        body.grid(row=2, column=0, sticky='nsew')
        table_frame = ttk.Frame(body)
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)
        tree = ttk.Treeview(table_frame, columns=('item', 'value', 'state', 'updated'), show='headings', selectmode='browse')
        for column, title, width in (('item', 'Item', 230), ('value', 'Value / position', 260), ('state', 'Kind / state', 170), ('updated', 'Updated', 130)):
            tree.heading(column, text=title)
            tree.column(column, width=width, minwidth=90, stretch=column in ('item', 'value'))
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        tree.grid(row=0, column=0, sticky='nsew')
        scrollbar.grid(row=0, column=1, sticky='ns')
        tree.tag_configure('highlight', background='#fff1c7', foreground='#6b4d00')
        tree.bind('<<TreeviewSelect>>', lambda _event, n=name: self._show_detail(n))

        detail = tk.Text(body, height=7, wrap=tk.WORD, state=tk.DISABLED, relief=tk.FLAT, padx=8, pady=8)
        body.add(table_frame, weight=4)
        body.add(detail, weight=1)
        self.views[name] = {'filter': filter_var, 'cards': cards, 'tree': tree, 'detail': detail}

    def refresh_all(self) -> None:
        for name in COLLECTORS:
            self.refresh(name)

    def refresh(self, name: str) -> None:
        view = self.views[name]
        try:
            cards, rows = COLLECTORS[name]()
        except Exception as exc:
            cards = [('Status', 'Unavailable')]
            rows = [('Collector error', str(exc), 'error', 'now', repr(exc))]
        self.rows[name] = rows
        for child in view['cards'].winfo_children():
            child.destroy()
        for column, (label, value) in enumerate(cards[:4]):
            card = ttk.LabelFrame(view['cards'], text=label, style='Section.TLabelframe', padding=(12, 7))
            card.grid(row=0, column=column, sticky='nsew', padx=(0 if column == 0 else 4, 0))
            ttk.Label(card, text=value, font=('Helvetica', 15, 'bold')).pack(anchor='w')
        self._render(name)

    def _render(self, name: str) -> None:
        view = self.views[name]
        tree = view['tree']
        query = view['filter'].get().strip().lower()
        tree.delete(*tree.get_children())
        for index, row in enumerate(self.rows.get(name, [])):
            visible = row[:4]
            if query and query not in ' '.join(str(value).lower() for value in visible):
                continue
            state = str(row[2]).lower()
            tags = ('highlight',) if any(flag in state for flag in ('highlight', 'stale', 'error', 'open')) else ()
            tree.insert('', tk.END, iid=f'{name}:{index}', values=visible, tags=tags)

    def _show_detail(self, name: str) -> None:
        view = self.views[name]
        selection = view['tree'].selection()
        if not selection:
            return
        try:
            index = int(selection[0].rsplit(':', 1)[1])
            content = self.rows[name][index][4]
        except (ValueError, IndexError, KeyError):
            content = ''
        detail = view['detail']
        detail.config(state=tk.NORMAL)
        detail.delete('1.0', tk.END)
        detail.insert('1.0', content)
        detail.config(state=tk.DISABLED)
