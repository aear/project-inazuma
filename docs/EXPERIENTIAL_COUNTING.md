# Experiential Counting

Ina counts by observing discrete units. She does not infer a plausible number
from language, copy a supplied total, use container length, or replace counting
with arithmetic over dimensions or duration.

`experiential_counting.py` is modality-neutral. A caller defines the unit and
the rule that makes one observation count:

- one detected onset can be one beat;
- one yielded raster position can be one pixel;
- one qualifying phase crossing can be one spiral wrap;
- one distinct immutable identifier can be one version.

Occurrence mode counts every admitted observation, including repeated equal
values. Unique mode requires an identity function and counts each identity once.
Either mode may also produce grouped subtotals from the same admitted units.

An exhausted source with no unresolved units produces `status: exact`. A budget
stop produces `status: incomplete` and a lower bound. Classification, identity,
or grouping failures produce `complete_with_unresolved`, never a false exact
claim. Progress can be checkpointed; persistence remains the caller's choice.

`verify_by_recount(...)` performs two separate enumerations and reports verified
only when both complete exact results agree. The cognition runtime exposes the
same mechanism as `experiential_counting`, capped at 100,000 observations per
invocation. Quantity-bearing experience events route to the counting specialist
through `experience_cognition.py`.

## Learned cadence

`counting_cadence.py` lets Ina learn whether to accumulate observed units
linearly or in verified groups. A task may offer any sensible positive group
sizes up to the bounded maximum; 2, 5, and 10 are defaults, not privileged
answers. This is group or skip counting, not mathematical logarithmic
estimation. Every unit still passes through the experiential counter, and an
incomplete final group is retained as a remainder.

The cadence learner is a finite, inspectable online bandit. Feedback rewards
exact results and independent recount agreement more strongly than fewer
accumulator steps. If grouping is not reliable, stride 1 is mandatory. No
learning runs merely because time passed, and learned cadence changes how the
count is accumulated—not what observations are admitted.

`compare_linear_and_grouped(...)` supports concurrent dual counting. One
accumulator advances linearly while a second closes verified groups and retains
the final remainder. Their comparison detects arithmetic, grouping, or lost
remainder errors during the same pass. The result explicitly records that both
accumulators shared one observation stream: agreement validates accumulation,
but a separate enumeration is still needed to corroborate that perception did
not omit the same unit before both counters received it.
