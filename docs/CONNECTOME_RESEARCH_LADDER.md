# Connectome research ladder

Connectomes are external scientific witnesses, not blueprints for replacing
Ina's mind. Source snapshots remain immutable. Ina may analyse them and mutate
bounded derived copies in the governed experiment lab; any production change
still requires independent evidence, benchmarks, and human review.

Ina may also design an original connectome. That candidate remains an isolated
copy with no live-write capability. Before it can even be presented for
promotion, it must report capability, correctness, safety, robustness, resource
use, background interference, human-visible quality, rollback, held-out cases,
and adversarial cases. The proposal is conspicuously flagged for human review.

**Experiment priority is honesty, then safety, then correctness, then
efficiency.** Failures, uncertainties, unavailable measurements, and conflicting
evidence are mandatory fields, including when the lists are empty. Incomplete
disclosure blocks judgement and review; a faster or more correct-looking result
cannot compensate for it.

## Acquisition order

1. **C. elegans — graph primitives.** Start with the MIT-licensed OpenWorm
   Connectome Toolbox and pin a Git revision plus file hashes. The graph is
   small enough to test directionality, chemical versus electrical edges,
   weighted hubs, reciprocity, components, lesion tolerance, and motifs in
   full. Do not mistake one reconstructed wiring diagram for activity or a
   universal optimisation target.
2. **FlyWire FAFB v783 — large-scale organisation.** Use the official Codex
   static CSV snapshot rather than scraping the live service. The download
   requires a human Google sign-in. Begin with `connections_princeton.csv.gz`
   and the annotation tables; do not acquire raw EM imagery. Preserve the
   dataset version, original filenames, hashes, licence, and citations.
3. **MICrONS — structure and activity.** Pin a CAVE materialization and query a
   bounded cell cohort. Pair its structural records with bounded functional NWB
   data from DANDI. The official documentation says the connectivity table is
   tens of gigabytes and imagery/segmentation are petabyte-scale, so wholesale
   mirroring is opt-in only.
4. **Ina experiments.** Compare biological graph signals with an exported,
   read-only summary of Ina's relevant subsystem. A biological feature is a
   hypothesis generator. Promotion needs multiple independent signals:
   capability, correctness, resource use, interference, and human-visible
   quality, including a retained historical baseline.

## First acquisition receipt (2026-09-17)

The official OpenWorm Connectome Toolbox is present locally in the ignored
`connectome_data/sources/openworm_connectome_toolbox` directory. It is pinned at
Git revision `b9c0b4a7bc2ccf47d3ce7aac624e1b3e2ea86254` (MIT-licensed repository).
The initial source table is `cect/data/herm_full_edgelist.csv`, SHA-256
`142693f17556148d7f962835b18ac6dd5af18b7467eef61815ebc1dd5474c0ca`.

The bounded loader observed 7,379 directed rows, 448 unique endpoints, 4,681
chemical rows, 2,698 electrical rows, total weight 39,702, and one weakly
connected component. The 448 endpoints include non-neuron cells such as muscles
and hypodermis; this must not be misreported as the classic 302-neuron count.
These are ingestion checks, not conclusions about useful architecture.

## Storage and provenance

- Put immutable source downloads on durable HDD storage under a directory that
  is not committed. Put rebuildable indexes and bounded working copies on the
  configured fast tier.
- Every snapshot records dataset, version/materialization, source URI, SHA-256,
  acquisition time, licence/citation note, schema, and validation result.
- Reject unknown schemas, missing endpoint columns, non-finite weights, and
  over-limit imports. Stream CSV/CSV.GZ; never load MICrONS bulk tables into RAM.
- A derived copy records its parent snapshot hash and every mutation. Derived
  copies cannot be used as new source truth.

## EEG presentation

The overlay is a separate `connectome_reference` network layer. It is labelled
`immutable_reference` or `experimental_copy`, uses deterministic positions, and
is bounded by explicit node/edge limits. It never writes `neural_memory_map`,
`logic_neural_map`, or `typed_neural_graph`. A useful next UI step is a three-way
view: biological reference, Ina subsystem, and difference/experiment layer.

## Initial experiments

- Remove or weaken high-degree, high-strength, and random nodes separately;
  compare reachability, task quality, and resource cost rather than assuming a
  hub is important from degree alone.
- Compare sparse recurrent motifs and modular bottlenecks against simpler
  candidate graphs at equal task capability and memory budget.
- In MICrONS, test whether structural predictors survive comparison with actual
  activity. Retain disagreements: structure alone is not function.
- For Ina, experiments run once by default, against copies, with no autonomous
  continuation budget and no production-tree mutation.

## Primary sources

- OpenWorm Connectome Toolbox: https://github.com/openworm/ConnectomeToolbox
- FlyWire Codex downloads: https://codex.flywire.ai/api/download?dataset=fafb
- MICrONS access guidance: https://tutorial.microns-explorer.org/faq.html
- MICrONS flagship dataset paper: https://doi.org/10.1038/s41586-025-08790-w
