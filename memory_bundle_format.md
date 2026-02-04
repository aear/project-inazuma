# Memory Bundle Format (v1)

Goal: reduce file count without deleting information by packing tiny files
into chunk files with a separate index.

## Layout

```
bundles/
  bundle_000001.pack
  bundle_000001.index.jsonl
  bundle_000002.pack
  bundle_000002.index.jsonl
  bundle_manifest.json
```

## Chunk file (`*.pack`)

Raw binary payloads concatenated in the order they were bundled. There is no
per-file header in the chunk file; the index is the source of truth.

## Index file (`*.index.jsonl`)

JSONL with one entry per file:

```
{"rel_path":"path/to/file.json","bundle":"bundle_000001","offset":0,"size":123,"sha256":"...","mtime_ns":1712345678901234567,"mode":33188}
```

Fields:
- `rel_path`: path relative to the bundling root.
- `bundle`: bundle id (matches the chunk filename stem).
- `offset`: byte offset in the chunk file.
- `size`: byte length of the payload.
- `sha256`: integrity checksum of the payload.
- `mtime_ns`: file modification time in nanoseconds.
- `mode`: original file mode bits.

## Manifest (`bundle_manifest.json`)

Stores format version, config, and a summary report to make audits repeatable.

## Notes

- `memory_bundler.py` writes bundles without deleting originals.
- File-count reduction happens when originals are retired after verification.
- The format is append-only and supports future readers without rewrite.

## Inastate trigger

`model_manager.py` listens for `bundle_request` in inastate and writes
`bundle_status` with the latest report. The request can include:

```
{
  "root": "AI_Children/Inazuma_Yagami/memory",
  "include": ["**/*.json"],
  "exclude": ["bundles/**"],
  "apply": false
}
```
