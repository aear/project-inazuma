# Cold Storage (Compact Fragments)

Cold storage keeps a tiny core per fragment plus optional shards that help
reconstruction later. The compacted fragment file remains in
`AI_Children/<child>/memory/fragments/cold`, but heavy payloads are removed.

## Layout

- `AI_Children/<child>/memory/cold_storage/cold_core.jsonl`
  - Append-only cores (identity, anchors, structure, checksums, provenance).
- `AI_Children/<child>/memory/cold_storage/shards/`
  - Optional shard blobs keyed by fragment id + shard type.
- `AI_Children/<child>/memory/cold_storage/heal_tickets.jsonl`
  - Append-only queue for background healing jobs.

## Compaction Hook

`memory_graph.MemoryManager.rebalance_tiers()` can compact fragments when they
move into the cold tier. This is opt-in via `cold_storage_policy`.

Example `config.json` snippet:

```json
{
  "cold_storage_policy": {
    "enabled": true,
    "auto_compact": false,
    "symbol_limit": 12,
    "word_limit": 12,
    "tag_limit": 12,
    "token_sketch_bits": 64,
    "shard_importance_threshold": 0.4,
    "max_shards": 4,
    "always_shards": ["token_sketch"]
  }
}
```

Set `auto_compact` to true if you want tier rebalancing to rewrite cold
fragments automatically. Leave it off to keep the behavior manual.
