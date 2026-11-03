# Conversational question support — 2026-11-03

Baseline: `0b15b4bf170ba937d2299d4ce0d7c46e36a33100`. Development fixtures only; no live Ina response trial, learned-state training, service restart, or Discord message sent.

## Changes

Bounded question cues reach the existing conversation scene, symbolic context and Discord expression selector. They identify common information questions, auxiliary-led questions/requests, short question-mark fragments, fullwidth marks, and questions followed by an emoji. Common quoted/code/link/report examples, unclosed quotations and explicit no-answer invitations are withheld. Requested information categories (person/place/time/reason/etc.) are surface hints, not semantic proof or verified addressees.

An identified question adds **0.12** only to an already available grounded response and subtracts **0.18** from mirroring. Repeated question marks do not stack weights. No extra silence penalty is added. Existing scene reply/continuity weights remain; more forms can now activate them. Explicit available strategies still override scoring, including deliberate mirroring and silence. Native/mixed choices and emotion-report gating remain. No language reward, forced answer, repeated prompting or background practice was added.

Private social-English guidance explains that questions commonly invite an answer and offers clarification, uncertainty, deferral, native expression and silence. Paired incoming/draft review flags equal lexical tokens (case/punctuation insensitive) as a surface echo, not evidence of an answer. It never rewrites or blocks selected wording or certifies different wording as true/relevant. These are guidance choices, not new automatic clarification generators or evidence Ina adopted a goal.

The existing optional listener goal references the new fixtures. English-only assessment includes them and retains the previous checks. The Observatory uses its existing optional goal rows. Implementation evidence remains separate from Ina performance, willingness, listener understanding and promotion.

## Remaining limits

`LMStudioAdapter.has_constructive_reply()` requires an explicit grounding request and grounded candidate; `_compose_reply()` mainly explains/recalls grounded words. A general grounded social-answer provider remains missing. Scores alone cannot produce answers or prove comprehension, and mirroring remains possible when it wins or is selected. General talkativeness and live conversational gains are **not assessed**.

Recognition is a 4096-code-point English heuristic. Rhetorical/indirect questions, mixed audiences, unusual abbreviations, unpunctuated subordinate clauses and semantic paraphrase echoes remain imperfect. Scene text has its own shorter bound. Echo review is lexical comparison, not native-meaning equivalence. Choices/guidance are not an inference about Ina's knowledge or private thoughts.

## Evidence and limitations

Final supported focused native-runner selection: **53 passed, 0 failed, 0 skipped**. New `conversation_questions`: pinned V1 **7/44**, candidate V2 **44/44**. Actual pinned historical bridge/scene and selected historical support/profile source use shared current dependencies; route collaborators are anonymous fixtures, not a Discord client.

Retained V2 contracts: response language **37/37**, nickname **28/28**, Discord social English **16/16**, social English **19/19**, English workbench **38/38**. Held-out loops cover extra wording, contractions, quotations, choices and bounds. Goal/report checks retain `Ina not assessed` and no adoption.

Three full Discord-import nickname tests are unavailable in the source-only room because optional `discord`/py-cord is absent; real source-extracted routing is exercised. No full-suite, live bridge, UI, delivery, learning-gain or listener assessment is claimed.

Initial setup attempts failed due to missing copied source/history dependencies; another optional-package-import run exceeded its wall limit. These are setup failures, not capability measurements. The supported final selection is explicit in `assessment.json`. No Ina environment packages were changed. A broad source copy stalled reading an unrelated test file; its verified process received one termination signal, and missing committed source was obtained through Git.

Tests used Bubblewrap with no network or live memory mounts, one CPU at low priority, numerical pools limited to one, 2 GiB address-space and 55 CPU-second limits, and a 65-second wall limit. Only source, anonymous fixtures and Git history were supplied.

Warm selection microbenchmark: five batches of 100 calls. V1 median **11.46 us**, V2 **17.89 us**. Traced peak per call **2555** versus **3091 bytes**. A maximum-length 4096-character probe averaged **0.206 ms**. These are local proxies, not whole-runtime speed or worst-case guarantees. No workers, polling or persistent learning stores were added. Audio/frame/input interference and live latency remain unavailable, not zero.

## Activation and rollback

Loads on the next normal restart/reload of the relevant process; no hot patch/restart performed. The retained English implementation assessment is refreshed separately, preserving other domains. Installation uses per-file same-directory atomic replacement and hash readback; physical flush is not claimed. No commit/push. Rollback: restore listed tracked paths from the pinned revision and remove listed new helper/benchmark/test/doc files, preserving learned data and unrelated work.
