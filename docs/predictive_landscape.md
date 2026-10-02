# Predictive landscape V1

An optional `predictive_landscape` personal tool offers a computational analogue
to spatially imagined timelines, not precognition or a physical Feynman model.
It retains the original supplied signal through emergence capture, including any
felt character or original timestamp the caller supplies. Capture time is separate
from the time a caller says the experience occurred. No personal anecdote is seeded
into Ina's memory by this implementation.

Supply a `specification` containing `signal`, `time_unit`, `nodes`, `links` and
`start`. Nodes have unique `id`,
`description`, `position: [time_offset, layout_y, layout_z]`, optional `countercheck`,
and kind `observation`, `possibility` or `attention_horizon`. Links name `from`, `to`,
an explicit `condition` and optional `evidence_references`. Links must move forward
in time. Layout axes carry no probability, salience score or causal authority.
The caller chooses the time origin and units; no absolute deadline is inferred.

The processor enumerates conditional paths, including alternative outcomes and
hypothetical links. Terminal possibilities with counterchecks are testable
candidates, not verified predictions. An unexplained horizon remains attention
without an inferred event. References and observations are caller-provided, not
independently corroborated; probability and predictive accuracy are unavailable.

An explicit `investigate_node` opens one stage of the existing intuition inquiry.
Later interpretations and outcome observations belong in separate linked records,
not edited original captures. The returned capture can be retained through existing
voluntary note/capture mechanisms. The personal tool retains each model as a new
private note under `Predictive Landscapes`, without changing cognitive memory
stores or scheduling revisits. It does not adjudicate outcomes or train a predictor.

For updates, supply the returned `previous_model` and a partial `specification`
replacing nodes/links/start or explicitly choosing an investigation. Every update
gets a new identifier and links to the previous model with a content hash; earlier
snapshots are not overwritten. The original capture and time units remain fixed.
New associations belong in the map revision, not in the original signal. An
investigation choice is not replayed on subsequent updates. The hash documents
lineage, not authenticated timestamp proof. Models are capped at 512 KiB.

This is passive retained background state. Updates occur only through chosen tool
requests (for example after new evidence); no autonomous observer or polling loop
has been added. Model retrieval uses the existing private-file tools, not a scan
of Ina's memory. Revisions require the previous model, not an arbitrary filepath.

Limits: 32 KiB input, 32 nodes, 64 links, 64 path expansions, 16 returned branches,
eight nodes per path. Partial exploration reports truncation. No network, file
reads, execution or publication occurs. There is no rendered 3D viewer yet: the
spatial representation is inspectable JSON. V1 benchmarks cover representation
and bounded inquiry; historical comparison and live predictive gains are unavailable.
