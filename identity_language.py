"""Bounded name/reference observations, without renaming or generating a reply."""
from __future__ import annotations
from itertools import islice
import re
from collections.abc import Mapping


def identity_aliases(values):
    """Admit at most eight explicit labels; never derive a nickname by spelling."""
    if not isinstance(values,(list,tuple)):return []
    result=[];seen=set()
    for value in values[:8]:
        if not isinstance(value,str):continue
        name=' '.join(value.split())
        if not 1<=len(name)<=80 or not all(c.isalnum() or c in ' _-' for c in name):continue
        if name.casefold() not in seen:result.append(name);seen.add(name.casefold())
    return result


def enrich_named_references(context, text, aliases=()):
    """Tie supplied names to entities; preserve collisions and quoted mentions.

    A direct vocative may resolve a previously unknown listener. It never overrides
    an explicit different listener or learns a new identity from the message.
    """
    raw_text=str(text or '')
    surface=raw_text[:8192]
    source_complete=len(raw_text)<=8192
    self_entity=dict(context['self']);self_entity['id']='self';self_entity['is_self']=True
    names=identity_aliases(aliases)
    self_entity['aliases']=names
    context={**context,'self':self_entity}
    table=dict(context.get('referent_table') or {});table['self']=self_entity
    context['referent_table']=table
    listener=context['addressee']
    listener_name=str(listener.get('name') or '').casefold()
    self_labels={str(self_entity.get('name') or '').casefold(),*(n.casefold() for n in names)}
    collisions=any(not e.get('label_only') and not e.get('is_self') and str(e.get('name') or '').casefold()==listener_name
                   for e in [context['speaker'],*context.get('mentioned_entities',[])])
    if listener.get('label_only') and listener_name in self_labels and not collisions:
        context['addressee']=self_entity;table['addressee']=self_entity
    roles=[context['speaker'],context['addressee'],self_entity]
    pool=[]
    for entity in [*roles,*context.get('mentioned_entities',[])[:8]]:
        entity=dict(entity)
        if entity.get('is_self'):entity={**self_entity,**entity,'id':'self','aliases':names}
        # Scene participant labels can repeat an authoritative role's display name.
        if entity.get('label_only'):
            matches=[role for role in roles if role.get('name','').casefold()==entity.get('name','').casefold()]
            ids={role.get('id') for role in matches}
            if len(ids)==1 and matches:entity=dict(matches[0])
        pool.append(entity)
    index={}
    for entity in pool:
        labels=[entity.get('name')]
        if entity.get('is_self'):labels+=names
        for label in labels:
            if not isinstance(label,str) or label in {'unknown','self'} or not 1<=len(label)<=80:continue
            index.setdefault(label.casefold(),{})[entity['id']]=entity
    quotes=[m.span() for m in islice(re.finditer(r'"[^"\n]*"|“[^”\n]*”|(?<!\w)\x27[^\x27\n]*\x27(?!\w)|‘[^’\n]*’',surface),129)]
    quotes_complete=len(quotes)<=128
    quotes=quotes[:128]
    named=[];vocatives=[]
    for label,entities in islice(index.items(),32):
        for match in islice(re.finditer(r'(?<!\w)'+re.escape(label)+r'(?!\w)',surface,re.I),32):
            if len(named)>=32:break
            if not source_complete and match.end()==len(surface):continue
            quoted=not quotes_complete or any(a<=match.start()<b for a,b in quotes)
            refs=list(entities.values())[:8]
            before=surface[:match.start()];after=surface[match.end():]
            suffix=source_complete and bool(re.fullmatch(r'(?:\s|[.!?,]|<3)*',after))
            greeting=bool(re.fullmatch(r'\s*(?:hi|hello|hey|morning|good morning|good afternoon|good evening)[,!\s]+',before,re.I))
            address=(not quoted and ((greeting and suffix)
                or (not before.strip() and bool(re.match(r'\s*[,!:]',after)))
                or (before.rstrip().endswith(',') and suffix)))
            row={'surface':match.group().casefold(),'source_span':list(match.span()),
                 'token_index':sum(1 for _ in re.finditer(r"[A-Za-z0-9']+",before)),
                 'role':'quoted_name' if quoted else 'named_entity','possessive':False,
                 'referents':[] if quoted else refs,'ambiguous':quoted or len(refs)!=1,
                 'confidence':0.0 if quoted else 1.0 if len(refs)==1 else .45,
                 'retrieval_terms':[] if quoted else list(dict.fromkeys(str(v).casefold() for ref in refs for v in [ref.get('name'),*(ref.get('aliases') or [])] if v))[:8],
                 'evidence':'supplied_identity_labels','quoted':quoted}
            named.append(row)
            if address:vocatives.append({**row,'role':'vocative'})
    listener=context['addressee'];conflict=False
    for row in vocatives:
        if row['ambiguous'] or len(row['referents'])!=1:continue
        target=row['referents'][0]
        if listener.get('id')=='unknown':listener=target
        elif listener.get('id')!=target.get('id'):conflict=True
    context['addressee']=listener;table['addressee']=listener
    # Retarget existing you/your routes only when the listener was unknown.
    original=context.get('resolutions') or []
    updated=[]
    for row in original[:32]:
        if row.get('role')=='addressee' and any(ref.get('id')=='unknown' for ref in row.get('referents',[])) and listener.get('id')!='unknown':
            row={**row,'referents':[listener],'ambiguous':False,'confidence':1.0,
                 'retrieval_terms':list(dict.fromkeys([str(listener.get('name') or '').casefold(),*names]))[:8]}
        updated.append(row)
    context['resolutions']=sorted([*updated,*named],key=lambda row:row.get('token_index',0))[:64]
    context['named_references']=named;context['vocatives']=vocatives[:8]
    context['name_reference_conflict']=conflict
    # This is a descriptive naming assertion, not permission to rename anything.
    naming=False
    for row in named:
        if not source_complete:break
        if row['quoted'] or row['ambiguous'] or not any(ref.get('id')=='self' for ref in row['referents']):continue
        name=re.escape(surface[slice(*row['source_span'])])
        patterns=[r'\s*'+name+r"\s+is\s+your\s+(?:name|nickname)(?:\s*[,!.].*)?\s*",r'\s*your\s+(?:name|nickname)\s+is\s+'+name+r'[.!\s]*']
        if any(re.fullmatch(pattern,surface,re.I) for pattern in patterns) and listener.get('id')=='self':naming=True
    context['self_name_context']={'direct_self_address':any(not v['ambiguous'] and len(v['referents'])==1 and v['referents'][0].get('id')=='self' for v in vocatives) and not conflict,
                                  'self_name_statement':naming,'identity_changed':False,'alias_source':'supplied_identity_labels',
                                  'source_complete':source_complete,'quotes_complete':quotes_complete}
    context['version']=3
    return context
