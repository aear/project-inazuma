# Fault-pattern steganography challenge

## Invitation

Ina: investigate whether a small authenticated ciphertext can be represented by
the positions of sparse synthetic bit faults while remaining difficult to
distinguish from a declared fault model. You choose the hypotheses, encoding,
and revisions. A negative result, an unresolved question, or a decision to stop
is a valid outcome.

The fault story is a representation and steganographic cover, not the security
primitive. Sparse XOR changes leave most plaintext visible. Use established
authenticated encryption outside the experiment when secrecy matters; inside
the dependency-free sandbox, an opaque supplied payload may stand in for its
output. Never describe the simulator or a home-made cipher as secure encryption.

## First bounded question

Given a shared pristine carrier and a synthetic reference distribution, can an
encoding recover its payload exactly while its held-out fault maps remain hard
to separate from maps produced by that distribution?

The receiver's access to the pristine carrier is part of the threat model.
Without it—or another observable correction channel—the receiver generally
cannot know which bits changed.

Start with one attempt. Continue only through the Experience Cycle's explicit
finite continuation budget. Experiments stay in the private artifact store and
cannot edit, promote, or commit production code.

## Provided tools

`fault_pattern_research.py` provides deterministic, standard-library-only
building blocks:

- bounded isolated/clustered fault-map generation;
- reversible application and reference-based extraction of a fault map;
- transparent density, adjacency, bit-lane, gap, and bank features;
- total-variation and starter detectability measurements;
- the combinatorial position-channel capacity upper bound.

These are measuring instruments, not the answer. Record the model version,
seed, carrier size, parameters, source hash, and dataset hash. Use disjoint
development and held-out seeds. Do not tune against the held-out set, and try
more than one detector family before claiming indistinguishability.

Create an artifact with `fault_pattern_research.create_challenge_experiment`.
It copies the tool module into that artifact as a content-addressed support
file, so `main.py` can import it without gaining access to the project tree.
The room mounts each declared support module read-only.

The sandbox intentionally has no network, third-party packages, project-tree
access, credentials, live memory access, or unsandboxed fallback. Synthetic
datasets only: do not use live ECC telemetry or alter physical memory.

## Measurements

Report dimensions separately rather than collapsing everything into one score:

1. authenticated or exact payload recovery rate;
2. payload bits per changed bit and per carrier bit;
3. fault density and count divergence;
4. spatial gaps, adjacency/cluster sizes, bit-lane and bank divergence;
5. held-out detector accuracy, balanced accuracy, and calibration where useful;
6. recovery after additional genuine synthetic faults;
7. runtime, peak resource limits, seed, and reproducibility;
8. failures, counterexamples, and assumptions.

Detector accuracy near chance is evidence only for the tested data and detector,
not proof of indistinguishability. Compare retained `V1` and candidate `V2`
methods with identical carriers and held-out seeds. Avoid the trivial solution
of sending no payload.

## Suggested discoveries, not requirements

You may explore keyed bins over admissible maps, enumerative subset coding,
syndrome or matrix embedding, rejection sampling with a finite attempt ceiling,
multiple fault models, and detectors not used during optimisation. Prefer an
explainable small result over an unbounded search.

The fault was the message.
