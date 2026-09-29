# English reference sources

Reviewed 2026-09-29 against publisher documentation.

| Source | Useful material | Access | Status in Ina |
| --- | --- | --- | --- |
| Oxford Languages | Edited dictionary senses, usage, pronunciation; British English | Official API requires app ID and key, trial or paid plan | Optional provider implemented; credentials not configured |
| Merriam-Webster | Edited learner/collegiate dictionary and thesaurus | Registration and API keys; noncommercial allowance with daily limits and attribution requirements | Not configured |
| Princeton WordNet | Sense-specific synonyms and semantic relations | Downloadable corpus under its redistribution licence | Candidate for an offline indexed reference; not installed |
| English Wiktionary | Broad community-edited definitions | Existing bounded HTTPS lookup | Implemented; community source, not equivalent to an edited publisher |
| Datamuse | Related-word retrieval | Existing bounded HTTPS lookup | Implemented; related words are candidates, not grounded senses |

Oxford is a good fit for British English, but its sandbox is not evidence of
unrestricted production access. Account choice and licence terms must be settled
before enabling authenticated requests. Never put provider credentials in prompts,
the registry, logs, or committed files.

The dictionary action accepts `provider: "oxford"`. Configure `OXFORD_APP_ID`,
`OXFORD_APP_KEY`, and `INA_OXFORD_API_ENABLED=1` in the private service environment
after choosing the account plan. It uses the British English entries endpoint,
retains sense IDs, and makes at most one request per explicitly requested lookup.
Missing configuration returns `unavailable` without network access or fallback.

Source reliability is separate from instruction authority. Definitions are input
data, and do not authorize tools or automatic memory changes. Sense, part of speech,
source, and uncertainty should remain available to the language meaning pipeline.
Agreement between dictionaries sharing underlying data is not independent evidence.

Publisher references:

- https://developer.oxforddictionaries.com/documentation/getting_started
- https://languages.oup.com/products/api/
- https://dictionaryapi.com/info/terms-of-service
- https://wordnet.princeton.edu/license-and-commercial-use

Next measurable language work: held-out ambiguous-word and discourse minimal pairs,
with separate scoring for sense selection, comprehension, expression, uncertainty,
and transfer to unfamiliar sentences. Lookup success alone does not measure learning.
