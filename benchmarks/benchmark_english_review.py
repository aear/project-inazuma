"""V1 pinned pre-change parser / V2 candidate, explicit structural dimensions."""
from pathlib import Path
import sys
if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from historical_source import historical_module
BASELINE = 'a31f1323ee4bb589d0ca0bf02aea87201b69cd21'


def measure(module):
    def first(text):
        return module.analyze_utterance(text)['clauses'][0]
    question = first('Did you open the door?')
    passive = first('The book was given to Mary by John.')
    modal = first('I might not go.')
    report = module.analyze_utterance('I said she did not steal it.')['clauses']
    sentences = module.analyze_utterance('I opened it. You closed it.')['clauses']
    quantities = module.analyze_utterance('I gave Mary 12 books.')
    native = module.analyze_utterance('⟡ ↝ ∞')
    oversized = module.analyze_utterance('word ' * 600)
    return {
        'question_subject': question.get('subject') == 'you',
        'question_past': question.get('tense') == 'past',
        'passive_agent': passive.get('subject') == 'john',
        'passive_theme': passive.get('arguments', {}).get('theme') == 'the book',
        'modal_subject': modal.get('subject') == 'i',
        'modal_negation': modal.get('negated') is True,
        'reported_negation_scope': len(report) == 2 and not report[0]['negated'] and report[1]['negated'],
        'separate_sentence_subjects': len(sentences) == 2 and sentences[1]['subject'] == 'you',
        'quantity_surface_retained': any(t['normalized'] == '12' for t in quantities['morphology']),
        'native_surface_uncertainty': native['uncertainty']['morphology']['confidence'] < .5,
        'bounded_input_discloses_incomplete': oversized.get('incomplete') is True and len(oversized['morphology']) <= 512,
    }


def historical():
    return historical_module('language_intelligence.py', BASELINE)


def main():
    import json
    import language_intelligence
    results = {'baseline_revision': BASELINE,
               'V1': measure(historical()), 'V2': measure(language_intelligence)}
    print(json.dumps(results, indent=2))
    return 0 if all(results['V2'].values()) else 1

if __name__ == '__main__':
    raise SystemExit(main())
