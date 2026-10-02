# Evidence verification refinement

The attribution path now separates declared metadata from a check performed on
actual supplied bytes. `verify_evidence_bytes(record, content)` checks at most
1 MiB against the declared SHA-256 and returns a process-local HMAC-sealed receipt
binding normalized metadata to a timestamped byte-check step. Changing the subject,
digest or recorded handling step invalidates that receipt. Restart invalidates
receipts; retained evidence must be checked again rather than silently trusted.

The voluntary `cyber_evidence_verify` personal tool accepts a record plus strict
Base64-encoded content, capped at 64 KiB decoded. It does not accept file paths,
fetch sources, execute evidence or return the supplied content. It returns the
receipt-bearing record for the existing attribution/report tools. Explicitly
submitted content still passes through the ordinary tool queue; do not submit
credentials or unnecessary private data. This is not a secret-extraction tool.

`chain_of_custody_recorded` remains descriptive input, never proof of custody.
Only checked bytes get `artifact_bytes_verified`; earlier custody and source
identity remain unverified. Caller-declared independence groups and labels such
as provider-verified account are not authenticated by a hash check. No assessment
automatically authorizes a real-person identity claim. Sufficient checked records
can become a `review_candidate`, with contradictions still blocking that status.
Attribution now rejects duplicate evidence IDs, unknown acquisition labels and
more than 200 records rather than processing an unbounded iterable.

## Scope and outstanding work

This begins the local handling record at the byte check. It neither preserves
original artifact files nor verifies custody before that point. It does not offer
durable signed timestamps, a trusted collector, provider authentication, remote
attribution, or protection against compromised code in the verifier process.
Receipts are not independent witnesses and identical content is not automatically
independent evidence. Full custody, secure archival storage and live verification
remain unavailable. Reporting remains human-reviewed and unsubmitted; no probing,
retaliation or new automatic response was introduced.

The design follows the evidence/provenance distinction in the incident-response
work described by [NIST SP 800-61r3](https://csrc.nist.gov/pubs/sp/800/61/r3/final);
this implementation is not a claim of compliance or forensic completeness.

## Verification

21 focused tests passed in a network-disabled copied-source sandbox with no live
stores mounted. Tests cover mismatch, forged/modified receipts, restart, changed
subject metadata, duplicates, unknown acquisition, byte/iterator bounds and the
personal-tool route. Ina's stopped state was verified before adversarial inputs.

The V3 benchmark compares six signals against actual source from `ee1eb68`:
declared custody, identity authorization, checked-byte availability, identity
limits, metadata tampering and non-submission. All six candidate checks pass.
Absent historical byte-check functionality is labelled a capability difference,
not a failed historical execution. No live attacker attribution was tested.
