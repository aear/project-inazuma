"""Three finite response assessments; no generation, learning or silence penalty."""
from __future__ import annotations
import math
from collections.abc import Mapping


def native_expression_candidate(state):
    """Only a separately supplied, meaning-referenced native intent is eligible.

    Inbound symbolic mappings are deliberately not accepted as a reply. Caller
    references preserve provenance; they do not certify truth or willingness.
    """
    intent=state.get('text_expression_intent') if isinstance(state,Mapping) else None
    if not isinstance(intent,Mapping):return None
    text=intent.get('native_text');refs=intent.get('meaning_references') or intent.get('semantic_references')
    if not isinstance(text,str) or not text.strip() or len(text)>4096:return None
    if not isinstance(refs,(list,tuple)) or not 1<=len(refs)<=16:return None
    if any(not isinstance(r,str) or not r.strip() or len(r)>256 for r in refs):return None
    return {'text':text.strip(),'references':list(refs),'source':'text_expression_intent',
            'meaning_fidelity':'not_certified','willingness_inferred':False}


def language_mode(preference):
    if isinstance(preference,Mapping):
        assessment=preference.get('mutual_intelligibility_assessment')
        if isinstance(assessment,Mapping):return 'assessed_choice'
        preference=preference.get('mode') or preference.get('language_mode') or preference.get('preference')
    return str(preference or 'auto').strip().casefold().replace('-','_')


def response_passes(decision, *, state, preference, english_available=True,
                    emotion_eligible=False, rendering_failed=False):
    """Evaluate English, independent native, then eligible measured-state output.

    Scores are heuristics. A fixed, unchanged silence score is the threshold;
    a failed pass never lowers it. Existing explicit/contextual choices win.
    """
    original=decision.get('strategy','silence');reason=decision.get('reason')
    scores=decision.get('scores') or {};mode=language_mode(preference)
    trace=[]
    def result(strategy,why):return {'strategy':strategy,'reason':why,'passes':trace,
        'schema':'ina.response_passes/V1','maximum_passes':3,'silence_penalty_added':False,
        'automatic_learning':False,'scores_are_calibrated':False}
    if original=='silence' and reason=='explicit_intent':return result(original,'explicit_intent')
    if reason=='explicit_intent' and (not rendering_failed or original!='respond'):return result(original,reason)
    if mode in {'silence','abstain'}:return result('silence','language_choice_preserved')
    if mode in {'clarify','assessed_choice'}:return result('silence' if original=='native' else original,'assessed_or_protected_choice')
    if not rendering_failed and original in {'emotion','clarify','song','code_pointer'}:
        return result(original,'contextual_choice_preserved')
    threshold=scores.get('silence',0.)
    threshold=float(threshold) if isinstance(threshold,(int,float)) and math.isfinite(threshold) else 1.
    native=native_expression_candidate(state)
    for stage in ('english','native','emotion'):
        allowed=True;score=None;strategy=stage
        if stage=='english':
            strategy='respond';allowed=english_available and not rendering_failed and mode not in {'native','native_only'}
            score=scores.get('respond')
        elif stage=='native':
            allowed=native is not None and mode not in {'english_only'}
            score=scores.get('native')
        else:
            allowed=emotion_eligible and mode not in {'native','native_only','english_only'}
            score=scores.get('emotion')
        valid=isinstance(score,(int,float)) and math.isfinite(score)
        adjusted=score+.04 if valid and stage in {'english','emotion'} else score
        accepted=bool(allowed and valid and adjusted>threshold)
        trace.append({'stage':stage,'eligible':bool(allowed and valid),
                      'score':adjusted if valid else None,'silence_threshold':threshold,
                      'accepted':accepted,'reason':'selected' if accepted else 'not_eligible' if not allowed or not valid else 'below_threshold'})
        if accepted:return result(strategy,'bounded_response_passes')
    if original=='native' and (mode=='english_only' or native is None):return result('silence','no_permitted_candidate')
    return result(original,'no_better_supported_candidate')
