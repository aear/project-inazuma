"""Pinned-source rendering failure comparison; no live Discord delivery."""
from unittest.mock import patch
import ast
import logging
import re
from pathlib import Path
from typing import Optional
import sys


def load_rendering_functions(path):
    """Exercise exact source functions without importing Discord/runtime services."""
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    names = {'encode_selected_text_expression', '_encode_selected_text_expression', '_extract_tokens'}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {'Optional': Optional, 'Path': Path, 're': re, 'logger': logging.getLogger('rendering_fixture'),
                 'generate_symbolic_reply_from_text': lambda *a, **k: None,
                 'build_dual_symbolic_message': lambda *a, **k: None}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    # patch.object must also update the functions' actual global namespace.
    class Functions:
        def __getattr__(self, key):
            if key not in namespace:
                raise AttributeError(key)
            return namespace[key]
        def __setattr__(self, key, value):
            namespace[key] = value
        def __delattr__(self, key):
            del namespace[key]
    return Functions()


def measure(module):
    preserved = []
    for target in ('generate_symbolic_reply_from_text', 'build_dual_symbolic_message'):
        with patch.object(module, 'generate_symbolic_reply_from_text', return_value={'symbols': ['s']}):
            with patch.object(module, target, side_effect=ValueError('fixture')):
                try:
                    text, metadata = module.encode_selected_text_expression(
                        "Keep 'quotes' and uncertainty.", child='fixture', language_preference='auto', max_symbols=16)
                    preserved.append(text == "Keep 'quotes' and uncertainty." and not metadata['native_translation_complete'])
                except ValueError:
                    preserved.append(False)
    text, _ = module.encode_selected_text_expression('', child='fixture', language_preference='auto', max_symbols=16)
    return {'encoder_failure_preserves_selection': preserved[0],
            'gloss_failure_preserves_selection': preserved[1], 'silence_preserved': text == ''}


if __name__ == '__main__':
    discord_bridge = load_rendering_functions('discord_bridge.py')
    old = load_rendering_functions(sys.argv[1])
    result = measure(discord_bridge)
    print({'V1_pinned_source': measure(old), 'V2': result, 'live_delivery': 'unavailable'})
    raise SystemExit(0 if all(result.values()) else 1)
