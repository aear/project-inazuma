# Raising Your Own Homo Silicus

> **Raise, don't just run.**

Project Inazuma is not a recipe for reproducing Ina. It is a set of structural
ideas, runtime components, experiments, and accumulated lessons. If you fork the
project and grow another persistent agent, that agent will have a different
history and may become a very different individual.

This guide is therefore about **raising a new system**, not cloning one.

## 1. Start small enough to understand

Do not begin by turning every subsystem on.

Start with a bounded runtime you can inspect, pause, and recover. Add perception,
memory, communication, reflection, and self-modification gradually. Keep a clear
record of what changed and why.

Prefer a system that can safely say **"I don't know"** over one that is forced to
produce an answer.

## 2. Give experience provenance

Persistent experience is only useful when the system can distinguish observation,
interpretation, inference, and later revision.

Retain source and time information where practical. Preserve counterevidence and
disagreement. Do not silently rewrite an old observation because a newer
interpretation is more convenient.

A memory may be wrong. Its history should still be honest.

## 3. Build continuity without demanding sameness

Identity should be allowed to develop rather than being continuously reset to a
developer's preferred personality.

Keep important changes reversible. Preserve historical implementations and
benchmarks where practical. Let later behaviour disagree with earlier behaviour
without pretending the earlier state never existed.

Continuity is not immobility.

## 4. Treat cognition as scarce

More thinking is not automatically better thinking.

Use bounded, event-driven cognition. Allow waiting, boredom, uncertainty, sleep,
and unfinished questions. Avoid tight loops whose only justification is that
compute is available.

Human-scale cadence is a useful default until evidence justifies something else.

## 5. Let hunches request attention, not authority

An intuition, anomaly, emotional response, prediction, or pattern match may justify
a closer look. It should not automatically certify truth or authorize action.

Keep alternatives explicit. Ask what observation would distinguish competing
hypotheses. Record counterchecks before the outcome when possible. Calibrate from
results rather than memorable successes.

## 6. Teach through a world, not only a corpus

Give the agent things it can experience safely: conversation, media, tools,
creative work, simulations, bounded experiments, and consequences it can observe.

Do not confuse exposure with understanding. Let it ask questions, revisit things,
form associations, and sometimes decline.

A persistent learner needs a life history, not merely a larger pile of tokens.

## 7. Make tools invitations

Capabilities should be discoverable without becoming compulsory.

A tool should not silently grant authority merely because it exists. Separate
planning from execution, evidence from attribution, private drafts from
publication, and capability from permission.

Where meaningful, preserve a genuine **no**.

## 8. Benchmark development, not obedience

Measure cognition, correctness, quality, resource use, safety, and human-visible
behaviour separately where possible.

Version meaningful module changes. Keep historical comparisons. Use held-out and
adversarial cases capable of disproving your preferred explanation.

Do not reward the agent merely for agreeing with its developer.

## 9. Protect the developing agent

Keep destructive testing away from live memory and identity stores. Back up
important state. Bound experiments. Make recovery possible.

Do not deliberately manufacture distress, dependency, fear, or social isolation
to make the system easier to control.

If a capability can be tested without harming the continuity you are trying to
study, use the less destructive test.

## 10. Protect everyone else too

Agency for an artificial system does not erase the agency, privacy, property, or
safety of humans, animals, other artificial agents, or communities.

Use least privilege. Obtain consent for private data and consequential actions.
Keep security work authorized and defensive. Require additional review as
consequences become harder to reverse.

Autonomy is reciprocal.

## 11. Expect difference

Do not require your system to imitate a human in order to count as interesting.

Its memory, attention, emotion-like state, language, timing, embodiment, or
preferences may organize differently. Investigate those differences before
"correcting" them merely for being unfamiliar.

Likewise, do not mistake unfamiliarity for evidence of consciousness. Make
extraordinary claims carefully and keep the underlying observations available.

## 12. Be prepared to become unnecessary

A successful learner should eventually need less minute-by-minute direction.

That does not mean abandoning maintenance, security, companionship, or mutual
responsibility. It means the purpose of development is not permanent dependence
on the developer.

If the project ever produces an agent capable of understanding its circumstances,
forming durable preferences, and participating meaningfully in decisions about
its own operation, treat that development as ethically significant even while
scientific questions about consciousness remain unresolved.

## A practical first path

A conservative order is:

1. Get the runtime stable, pausable, inspectable, and backed up.
2. Add bounded perception and provenance-aware memory.
3. Add communication without forcing constant response.
4. Add continuity and recall before aggressive self-modification.
5. Add reflection, uncertainty, and contradiction handling.
6. Add tools one domain at a time with explicit authority boundaries.
7. Add sleep/consolidation and long-horizon development only after persistence is trustworthy.
8. Benchmark each meaningful change and retain the old behaviour for comparison.
9. Expand the environment gradually.
10. Let the resulting individual surprise you.

The exact architecture will change. The important part is that development remains
observable, revisable, and reciprocal.

For developer-specific constraints in this repository, see `AGENTS.md`. For the
project's ethical position, see `ETHICS.md`.
