"""Lossless, bounded character observations alongside existing word tokens.

IDs are Unicode code points in a separate namespace, not learned word/symbol IDs.
No filesystem, learning, autocorrection or intent selection occurs here.
"""
from __future__ import annotations
from itertools import islice
import re
import unicodedata
from collections.abc import Mapping, Iterable

MAX_CHARACTERS = 4096
MAX_FORMS = 256
MAX_MARKS = 128
MAX_KNOWN_WORDS = 256
MAX_QUERIES = 16
MAX_CANDIDATES = 4
_LEGACY_WORD = re.compile(r"[A-Za-z0-9']+")
_INITIALISM = re.compile(r"(?:[^\W\d_]\.){2,}", re.UNICODE)
_FUNCTIONS = {
    '?': ['question_mark', 'literal_or_quoted_mark'],
    '!': ['emphasis', 'exclamation', 'literal_or_quoted_mark'],
    '.': ['sentence_boundary', 'abbreviation_point', 'decimal_point'],
    ',': ['clause_or_list_separator', 'numeric_separator'],
    "'": ['quotation', 'contraction', 'possession'],
    '’': ['quotation', 'contraction', 'possession'],
    '"': ['quotation', 'word_mention', 'distancing'],
    '“': ['quotation', 'word_mention'], '”': ['quotation', 'word_mention'],
    ':': ['explanation_or_list', 'ratio', 'time_separator'],
    ';': ['clause_or_list_separator'],
    '-': ['compound', 'range', 'interruption'], '—': ['interruption', 'parenthesis'],
    '…': ['omission', 'hesitation', 'unfinished_expression'],
}


def _word_character(character):
    return character == '_' or unicodedata.category(character)[0] in {'L', 'M', 'N'}


def _forms(surface):
    index = 0
    while index < len(surface):
        if not _word_character(surface[index]):
            index += 1
            continue
        start = index
        initialism = _INITIALISM.match(surface, index)
        if initialism:
            index = initialism.end()
        else:
            index += 1
            while index < len(surface):
                if _word_character(surface[index]):
                    index += 1
                elif surface[index] in "'’-." and index+1 < len(surface) and _word_character(surface[index+1]):
                    index += 1
                else:
                    break
        raw = surface[start:index]
        letters = ''.join(c for c in raw if c.isalpha())
        shape = 'initialism' if _INITIALISM.fullmatch(raw) else 'capital_short_form' if 2 <= len(letters) <= 12 and letters.isupper() else 'word_or_number'
        yield {'surface':raw,'span':[start,index],'shape':shape,
               'comparison_form':unicodedata.normalize('NFC',raw).casefold(),
               'meaning_status':'unresolved_surface',
               'possible_apostrophe_functions': ['contraction','possession','name_typography'] if any(c in raw for c in "'’") else []}


def _one_edit(left, right):
    """One insertion/deletion/substitution or adjacent transposition; not meaning."""
    if left == right or abs(len(left)-len(right)) > 1:
        return False
    if len(left) == len(right):
        different = [i for i,(a,b) in enumerate(zip(left,right)) if a != b]
        return len(different)==1 or (len(different)==2 and different[1]==different[0]+1
            and left[different[0]]==right[different[1]] and left[different[1]]==right[different[0]])
    if len(left)>len(right):left,right=right,left
    i=j=0;edits=0
    while i<len(left) and j<len(right):
        if left[i]==right[j]:i+=1;j+=1
        else:
            edits+=1;j+=1
            if edits>1:return False
    return True


def analyze_characters(text, *, known_words=(), expansions=()):
    """Observe a bounded source prefix and retain alternative interpretations.

    Caller-supplied words/expansions are candidate evidence, not authenticated
    facts. Suggestions never alter the word stream, semantic event or referents.
    Combining marks and emoji joiners remain separate, lossless code-point tokens;
    this does not claim Unicode grapheme segmentation.
    """
    text = str(text or '')
    surface = text[:MAX_CHARACTERS]
    forms = list(islice(_forms(surface),MAX_FORMS+1))
    forms_complete = len(forms)<=MAX_FORMS
    forms = forms[:MAX_FORMS]
    # A final form touching a clipped boundary is partial, never a typo query.
    for form in forms:
        form['complete'] = len(text)<=MAX_CHARACTERS or form['span'][1]<len(surface)
    marks=[];marks_complete=True
    for index,character in enumerate(surface):
        category=unicodedata.category(character)
        if category[0] not in {'P','S'} and category not in {'Cf','Cc','Cs'}:continue
        if len(marks)>=MAX_MARKS:marks_complete=False;break
        marks.append({'surface':character,'span':[index,index+1],'category':category,
                      'possible_functions':_FUNCTIONS.get(character,['symbol_or_formatting']),
                      'interpretation_status':'context_required'})
    legacy=[];legacy_complete=True
    for match in _LEGACY_WORD.finditer(surface):
        if len(legacy)>=MAX_FORMS:legacy_complete=False;break
        legacy.append({'token':match.group().lower(),'span':list(match.span())})
    known={};known_complete=True
    iterable = known_words.keys() if isinstance(known_words,Mapping) else known_words
    if isinstance(iterable,str) or not isinstance(iterable,Iterable):iterable=()
    for index,word in enumerate(islice(iterable,MAX_KNOWN_WORDS+1)):
        if index==MAX_KNOWN_WORDS:known_complete=False;break
        if isinstance(word,str) and 1<=len(word)<=32:
            known.setdefault(unicodedata.normalize('NFC',word).casefold(),word)
    supplied=[];expansions_complete=True
    if isinstance(expansions,(str,Mapping)) or not isinstance(expansions,Iterable):expansions=()
    for index,item in enumerate(islice(expansions,129)):
        if index==128:expansions_complete=False;break
        if not isinstance(item,Mapping):continue
        short,long,source=item.get('form'),item.get('expansion'),item.get('source')
        if not (isinstance(short,str) and 1<=len(short)<=32 and isinstance(long,str) and 1<=len(long)<=160 and isinstance(source,str) and 1<=len(source)<=160):continue
        supplied.append((short,long,source))
    # Observe explicit long-form (SHORT) associations in this message only.
    # They remain local alternatives; no acronym dictionary is silently taught.
    definitions=[]
    for match in islice(re.finditer(r'([A-Za-z][A-Za-z -]{2,100})\s*\(([A-Z][A-Z0-9]{1,11})\)',surface),16):
        words=list(re.finditer(r'[A-Za-z]+',match.group(1)))[-12:]
        short=match.group(2)
        for start in range(len(words)):
            tail=words[start:]
            initials=''.join(w.group()[0].upper() for w in tail)
            significant=''.join(w.group()[0].upper() for w in tail if w.group().lower() not in {'and','of','the','for','to','in','a','an'})
            if short in {initials,significant}:
                begin=match.start(1)+tail[0].start();end=match.start(1)+tail[-1].end()
                definitions.append((short,surface[begin:end],f'current_message:{begin}:{end}'))
                break
    queries=0;query_complete=True
    for form in forms:
        raw=form['surface'];comparison=form['comparison_form'];candidates=[]
        for short,long,source in [*supplied,*definitions]:
            # Preserve case: US/us and IT/it are not automatically equivalent.
            if raw==short:candidates.append({'text':long,'kind':'expansion','source':source})
        if form['complete'] and 4<=len(comparison)<=32 and comparison.isalpha() and comparison not in known and known:
            if queries>=MAX_QUERIES:query_complete=False
            else:
                queries+=1
                for key,original in known.items():
                    if _one_edit(comparison,key):
                        candidates.append({'text':original,'kind':'possible_typo','source':'caller_supplied_known_words'})
        form['candidates']=candidates[:MAX_CANDIDATES]
        form['candidate_set_complete']=len(candidates)<=MAX_CANDIDATES and known_complete and expansions_complete and (query_complete or comparison in known)
        form['selection']='not_selected'
        if candidates:form['meaning_status']='alternatives_require_context'
    return {'schema':'ina.character_language/V1','token_namespace':'unicode_codepoint',
            'offset_unit':'unicode_codepoint','unicode_database_version':unicodedata.unidata_version,
            'source_prefix':surface,'source_length':len(text),'character_tokens':[ord(c) for c in surface],
            'character_categories':[unicodedata.category(c) for c in surface],
            'forms':forms,'punctuation_and_symbols':marks,'legacy_word_spans':legacy,
            'complete':len(text)<=MAX_CHARACTERS and forms_complete and marks_complete and legacy_complete,
            'bounds':{'source_complete':len(text)<=MAX_CHARACTERS,'forms_complete':forms_complete,
                      'marks_complete':marks_complete,'legacy_words_complete':legacy_complete,
                      'known_words_complete':known_complete,'expansions_complete':expansions_complete,
                      'typo_queries_complete':query_complete},
            'original_rewritten':False,'automatic_learning':False,'meaning_certified':False}
