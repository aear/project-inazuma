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
