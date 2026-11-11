"""Fixed-path metadata inspection; never reads thoughts, tokens or experience text."""
from pathlib import Path


def inspect_language_paths(memory_root,fast_index_root=None):
    root=Path(memory_root);rows=[];times={}
    for name in ('text_vocab.json','symbol_to_token.json','text_vocab_links.json','text_vocab_links.sqlite',
                 'experiences/experience_graph.json'):
        path=root/name
        try:
            stat=path.stat();times[name]=stat.st_mtime_ns
            rows.append({'artifact':name,'tier':'durable','bytes':stat.st_size,'modified_ns':stat.st_mtime_ns,'status':'present'})
        except OSError:rows.append({'artifact':name,'tier':'durable','status':'unavailable'})
    if fast_index_root is not None:
        for name in ('text_vocab_links.sqlite','symbol_to_token.sqlite'):
            try:
                stat=(Path(fast_index_root)/name).stat()
                rows.append({'artifact':name,'tier':'configured_fast_index','bytes':stat.st_size,'modified_ns':stat.st_mtime_ns,'status':'present'})
            except OSError:rows.append({'artifact':name,'tier':'configured_fast_index','status':'unavailable'})
    old=times.get('experiences/experience_graph.json');new=times.get('text_vocab.json')
    return {'schema':'ina.language_path_status/V1','artifacts':rows,
            'graph_older_than_vocab':old<new if old is not None and new is not None else None,
            'freshness_is_verified':False,'meaning_learning_is_verified':False,
            'scope':'metadata_only; timestamps do not establish learning, freshness or causation',
            'automatic_repair':False,'migration_selected':False,'private_payloads_read':False}
