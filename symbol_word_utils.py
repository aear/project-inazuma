import math
from typing import Any, Dict, Iterator, List, Optional


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    if not v1 or not v2:
        return 0.0
    length = min(len(v1), len(v2))
    if length <= 0:
        return 0.0
    dot = sum(float(v1[i]) * float(v2[i]) for i in range(length))
    norm1 = math.sqrt(sum(float(v1[i]) * float(v1[i]) for i in range(length)))
    norm2 = math.sqrt(sum(float(v2[i]) * float(v2[i]) for i in range(length)))
    return dot / (norm1 * norm2 + 1e-8)


def normalize_vector(value: Any) -> Optional[List[float]]:
    if not isinstance(value, list) or not value:
        return None
    cleaned: List[float] = []
    for item in value:
        if isinstance(item, (int, float)):
            cleaned.append(float(item))
    return cleaned or None


def proto_confidence(uses: int, base: float = 0.2) -> float:
    uses = max(1, int(uses))
    return round(min(0.9, base + math.log1p(uses) / 5.0), 3)


def sequence_from_entry(key: str, entry: Dict[str, Any]) -> List[str]:
    seq = entry.get('sequence')
    if isinstance(seq, list):
        cleaned = [str(symbol_id) for symbol_id in seq if symbol_id]
        if cleaned:
            return cleaned
    core = str(key or '')
    if core.startswith('pair:'):
        core = core[5:]
    if '_' not in core:
        return []
    return [part for part in core.split('_') if part]


def candidate_summary(candidate: Dict[str, Any]) -> str:
    entry = candidate.get('entry') if isinstance(candidate.get('entry'), dict) else {}
    for field in ('summary', 'generated_word', 'word'):
        value = entry.get(field)
        if value:
            return str(value)
    labels = entry.get('labels')
    if isinstance(labels, list):
        cleaned = [str(label) for label in labels if label]
        if cleaned:
            return ' + '.join(cleaned[:3])
    sequence = candidate.get('sequence') if isinstance(candidate.get('sequence'), list) else []
    if sequence:
        return ' + '.join(str(symbol_id) for symbol_id in sequence[:3])
    return str(candidate.get('symbol_word_id') or candidate.get('key') or '')


def iter_symbol_word_candidates(word_state: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    if not isinstance(word_state, dict):
        return

    words = word_state.get('words') if isinstance(word_state.get('words'), list) else []
    for word in words:
        if not isinstance(word, dict):
            continue
        symbol_word_id = str(word.get('symbol_word_id') or '').strip()
        if not symbol_word_id:
            continue
        yield {
            'kind': 'word',
            'key': symbol_word_id,
            'symbol_word_id': symbol_word_id,
            'sequence': [],
            'symbol': word.get('symbol'),
            'entry': word,
        }

    for store_name, kind in (
        ('multi_symbol_words', 'multi_symbol_word'),
        ('proto_words', 'proto_word'),
    ):
        store = word_state.get(store_name)
        if not isinstance(store, dict):
            continue
        for raw_key, entry in store.items():
            if not isinstance(entry, dict):
                continue
            key = str(raw_key or '').strip()
            if not key:
                continue
            sequence = sequence_from_entry(key, entry)
            symbol_word_id = key if key.startswith('pair:') else f'pair:{key}'
            yield {
                'kind': kind,
                'key': key,
                'symbol_word_id': symbol_word_id,
                'sequence': sequence,
                'symbol': sequence[0] if sequence else None,
                'entry': entry,
            }


def score_symbol_word_candidates(
    pred_vec: List[float],
    transformer: Any,
    word_state: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    best: Optional[Dict[str, Any]] = None
    for candidate in iter_symbol_word_candidates(word_state):
        entry = candidate.get('entry') if isinstance(candidate.get('entry'), dict) else {}
        vec = normalize_vector(entry.get('vector'))
        if vec is None:
            summary = candidate_summary(candidate)
            if not summary:
                continue
            try:
                encoded = transformer.encode({'summary': summary})
            except Exception:
                continue
            vec = normalize_vector(encoded.get('vector') if isinstance(encoded, dict) else None)
            if vec is None:
                continue
        score = cosine_similarity(pred_vec, vec)
        if candidate.get('kind') != 'word':
            try:
                reliability = float(entry.get('confidence', 0.0) or 0.0)
            except Exception:
                reliability = 0.0
            reliability = max(0.0, min(1.0, reliability))
            score = (score * 0.85) + (reliability * 0.15)
        enriched = dict(candidate)
        enriched['summary'] = candidate_summary(candidate)
        enriched['vector'] = vec
        enriched['confidence'] = float(score)
        if best is None or float(enriched['confidence']) > float(best['confidence']):
            best = enriched
    return best
