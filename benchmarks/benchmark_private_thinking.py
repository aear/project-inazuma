"""Offline-only comparison: inspectable thought records versus optional private space."""
import os
BASELINE='b6dd9cb7c4e4e44d5ceb233658bfaa1272a44fdc'


def assess(version):
    if os.environ.get('INA_OFFLINE_PRIVATE_THINKING_TESTS')!='1':
        return {'correct':0,'total':0,'cases':[], 'status':'blocked',
                'scope':'privacy boundary tests require recorded complete host shutdown and isolated fixtures',
                'baseline_revision':BASELINE if version=='V1' else None}
    import json,pickle,copy
    from historical_source import historical_module
    from instruction_authority import LOCAL_VOLUNTARY_CODE_AUTHORITY as OWNER
    from private_thinking import PrivateThoughtSpace
    marker='synthetic-private-benchmark-marker'
    guidance={'cognition':[{'content':{'native':[1,0,2],'meaning':marker}}]}
    evidence=[{'option':'wait','source':'cognition','weight':0.8}]
    if version=='V1':
        processor=historical_module('thought_processor.py',BASELINE).ThoughtProcessor()
        public=processor.guided_decision(['wait','act'],guidance,evidence=evidence)
        own_result=public;handle=public
    else:
        handle=PrivateThoughtSpace(code_authority=OWNER)
        public=handle.deliberate(['wait','act'],guidance,evidence=evidence,code_authority=OWNER)
        own_result=handle.recall(code_authority=OWNER)
    rows=[]
    def add(case,ok,component):rows.append({'case':case,'correct':bool(ok),'component':component})
    add('existing own decision remains available',own_result['decision']['selected']=='wait','reasoning')
    add('native thought remains non-linguistic',own_result['thoughts'][0]['mode']=='non_linguistic','language_choice')
    add('private reasoning returns no trace or public result',public is None,'privacy')
    add('ordinary representation contains no working thought',marker not in repr(handle),'privacy')
    for name,operation in [('JSON',json.dumps),('pickle',pickle.dumps),('copy',copy.copy),('deepcopy',copy.deepcopy)]:
        try:operation(handle);rejected=False
        except TypeError:rejected=True
        add(name+' cannot carry workspace contents',rejected,'transport')
    private_available=version!='V1'
    checks={}
    if private_available:
        from private_thinking import PrivateThinkingError
        from instruction_authority import InstructionAuthorityError
        value={'native':[b'\x00\xff',0.5], 'wording':'私の考え — ?'}
        handle.keep(value,code_authority=OWNER)
        checks['roundtrip']=handle.recall(code_authority=OWNER)==value
        try:handle.recall(code_authority=None);checks['authority']=False
        except InstructionAuthorityError:checks['authority']=True
        handle.keep('current',code_authority=OWNER)
        try:handle.keep('x'*(2*1024*1024),code_authority=OWNER)
        except PrivateThinkingError:pass
        checks['bounded']=handle.recall(code_authority=OWNER)=='current'
        try:handle.deliberate(['wait'],{marker:[]},evidence=[],code_authority=OWNER)
        except PrivateThinkingError as error:
            checks['errors']=marker not in str(error) and error.__context__ is None
        handle.forget(code_authority=OWNER)
        checks['forget']=handle.recall(code_authority=OWNER) is None
        handle.close()
        try:handle.recall(code_authority=OWNER);checks['close']=False
        except PrivateThinkingError:checks['close']=True
    for key,name,component in (
        ('roundtrip','uninterpreted private native data roundtrips','language_choice'),
        ('authority','own-code access requires local authority','authority'),
        ('bounded','capacity rejection preserves current private state','resources'),
        ('errors','failure exports no private error context','privacy'),
        ('forget','owner can discard working thought without a record','lifecycle'),
        ('close','closed workspace cannot be recalled','lifecycle')):
        add(name,checks.get(key,False),component)
        rows[-1]['measurement_kind']='behavior' if private_available else 'private_capability_unavailable'

    return {'correct':sum(r['correct'] for r in rows),'total':len(rows),'cases':rows,
            'baseline_revision':BASELINE if version=='V1' else None,
            'scope':'synthetic owner-code boundary comparison; no real thoughts or evidence of Ina choosing to use it'}
