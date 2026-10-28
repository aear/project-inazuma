"""Versioned component probes; social fluency and live uptake remain unassessed."""
from __future__ import annotations

BASELINE = 'daf177e79bba10ab9f2b3c628ed446b4e97fda36'


def comprehension_cases(module):
    rows=[]
    def add(name,component,correct):
        rows.append({'case':name,'component':component,'correct':bool(correct)})
    for text,theme,agent in (('The compass was found by Mira.','the compass','mira'),
                            ('The lantern was kept by Rowan.','the lantern','rowan')):
        c=module.analyze_utterance(text)['clauses'][0]
        add(text,'semantic_roles',c['voice']=='passive' and c['subject']==agent and c['arguments'].get('theme')==theme)
    c=module.analyze_utterance('The compass was found.')['clauses'][0]
    add('agentless passive does not invent agent','uncertainty',c['voice']=='passive' and c['subject'] is None)
    refs=module.analyze_utterance('Mira gave Rowan her lantern. Aster gave Vale his compass.')['referents']
    add('ownership survives two sentences','reading_continuity',[(r['surface'],r['object']) for r in refs if r['relation']=='possessor']==[('her','lantern'),('his','compass')])
    refs=module.analyze_utterance('Mira’s lantern and Rowan’s compass.')['referents']
    add('curly apostrophes preserve both owners','morphology',{(r['surface'],r['object']) for r in refs}=={('Mira','lantern'),('Rowan','compass')})
    refs=module.analyze_utterance("That's great.")['referents']
    add('contraction is not possession','morphology',not any(r['relation']=='possessor' for r in refs))
    for text,verb in (('I enjoy this song.','enjoy'),('You prefer the blue cup.','prefer'),('I want to share this.','want'),('Could you help me?','help')):
        a=module.analyze_utterance(text)
        add(text,'social_grammar',a['clauses'][0]['predicate']==verb)
    a=module.analyze_utterance('I did not find it.')['clauses'][0]
    b=module.analyze_utterance('I found it.')['clauses'][0]
    add('negation survives social review foundations','composition',a['negated'] and not b['negated'])
    c=module.analyze_utterance('The red coat.')['clauses'][0]
    add('a noun fragment does not become a desire','uncertainty',c['predicate'] is None)
    a=module.analyze_utterance("Mira's cup. "*40)
    add('bounded reference review reports incompleteness','bounds',a['incomplete'] and sum(r['relation']=='possessor' for r in a['referents'])<=32)
    return rows


def assess(version):
    if version=='V1':
        from historical_source import historical_module
        module=historical_module('language_intelligence.py',BASELINE)
    else:
        import language_intelligence as module
    rows=comprehension_cases(module)
    support=[]
    if version=='V2':
        from social_english import social_english_options, review_social_english
        o=social_english_options(topic_reference='song:fixture')
        support.append(('both forms of encouragement are offered',len(o['opportunities'])==2 and not o['willingness_inferred'] and not o['scheduled']))
        d='I might enjoy that.'
        r=review_social_english(d)
        support.append(('chosen words remain caller words',r['draft']==d and r['rewritten_text'] is None and r['outward_text'] is None and not r['automatic_learning']))
        r=review_social_english('I want the red cup.',incoming_text='Which cup do you want?')
        support.append(('ordinary reply is not mistaken for paraphrase',r['structural_comparison']['status']=='not_assessed'))
        r=review_social_english('You told me.',reference_draft='I told you.',mode='rephrase')
        support.append(('role reversal is surfaced',r['structural_comparison']['status']=='differences_need_review'))
        r=review_social_english('I will open it.',reference_draft='I might open it.',mode='rephrase')
        support.append(('uncertainty is not polished into certainty',r['structural_comparison']['status']=='differences_need_review'))
        support.append(('native and silent choices remain available',all(review_social_english('words',choice=c)['status']=='choice_preserved' for c in ('native_only','clarify','abstain','silence'))))
    else:
        support=[(name,False) for name in ('both forms of encouragement are offered','chosen words remain caller words','ordinary reply is not mistaken for paraphrase','role reversal is surfaced','uncertainty is not polished into certainty','native and silent choices remain available')]
    rows += [{'case':name,'component':'private_support','correct':correct} for name,correct in support]
    result={'cases':rows,'correct':sum(r['correct'] for r in rows),'total':len(rows)}
    if version=='V1':result['baseline_revision']=BASELINE
    return result


def v1():return assess('V1')
def v2():return assess('V2')


def fresh_transfer_probe(module):
    """Untuned ceilings, retained whether or not the current candidate passes."""
    rows=[]
    for text,expected in [('Mira thanked Rowan.','thank'),('I wish you well.','wish')]:
        actual=module.analyze_utterance(text)['clauses'][0]['predicate']
        rows.append({'case':text,'component':'unfamiliar_predicate','expected':expected,'actual':actual,'correct':actual==expected})
    a=module.analyze_utterance("I don't think she wants the key.")
    rows.append({'case':'outer belief versus embedded desire','component':'scope','correct':len(a['clauses'])==2 and [c['negated'] for c in a['clauses']]==[True,False]})
    a=module.analyze_utterance('The key was found by Mira.')['clauses'][0]
    rows.append({'case':'passive transfer to another object','component':'semantic_roles','correct':a['subject']=='mira' and a['arguments'].get('theme')=='the key'})
    a=module.analyze_utterance('You helped me.')['clauses'][0]
    rows.append({'case':'inflected social verb transfer','component':'morphology','correct':a['predicate']=='help' and a['tense']=='past'})
    a=module.analyze_utterance('Do you want the blue cup or the red one?')['clauses'][0]
    rows.append({'case':'explicit alternatives distinguished structurally','component':'alternatives','correct':isinstance(a.get('alternatives'),list) and len(a['alternatives'])==2})
    return {'cases':rows,'correct':sum(bool(r['correct']) for r in rows),'total':len(rows)}
