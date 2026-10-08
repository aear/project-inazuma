"""V1/V2 selector comparison; Discord route separately tested by source extraction."""
from pathlib import Path
import ast
from typing import Optional, Any


def measure(selector):
    message = {'native_text': '∘⊙', 'gloss_text': 'candidate', 'text': 'Native: ∘⊙\nHuman guess: candidate'}
    return {'native_only_respected': selector(message, 'native_only') == ('∘⊙', 'native'),
            'english_available': selector(message, 'english') == ('candidate', 'english'),
            'mixed_preserved': selector(message, 'mixed')[0] == message['text']}


if __name__ == '__main__':
    import sys
    from language_processing import select_symbolic_message_text
    tree = ast.parse(Path(sys.argv[1]).read_text(encoding='utf-8'))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'select_symbolic_message_text']
    ns = {'Optional': Optional, 'Dict': dict, 'Any': Any}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), sys.argv[1], 'exec'), ns)
    current = measure(select_symbolic_message_text)
    print({'V1_pinned_selector': measure(ns['select_symbolic_message_text']), 'V2': current,
           'live_conversation_quality': 'unavailable'})
    raise SystemExit(0 if all(current.values()) else 1)
