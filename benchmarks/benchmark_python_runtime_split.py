"""Interpreter selection fixtures; no application processes or package installs."""
import ast
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

BASELINE='6ddc708052af29c993925a935ed4f2dda2a3e49b'


def assess(version):
    from historical_source import historical_text
    import runtime_environment as current
    rows=[]
    def add(name,ok,component='selection',kind='behavior'):
        rows.append({'case':name,'correct':bool(ok),'component':component,'measurement_kind':kind})
    with TemporaryDirectory() as temp:
        root=Path(temp);legacy=root/'venv/bin/python';legacy.parent.mkdir(parents=True);legacy.touch()
        core=root/'.runtime/core315';(core/'bin').mkdir(parents=True);(core/'bin/python').touch()
        if version=='V1':
            tree=ast.parse(historical_text('runtime_services.py',BASELINE))
            expr=next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SERVICE_COMMANDS' for t in n.targets))
            commands=eval(compile(ast.Expression(expr),'<pinned service commands>','eval'),{'sys':SimpleNamespace(executable=str(legacy))})
            def choose(service=None): return commands[service or 'world_server'][0]
        else:
            def choose(service=None): return current.project_python(root,service=service)
        add('unreviewed candidate does not change current Python',choose()==str(legacy),'review_boundary')
        add('bridge remains on supported Python before promotion',choose('discord_bridge')==str(legacy),'compatibility')
        (core/'.ina-ready').touch()
        add('reviewed core selected for GUI',choose()==str(core/'bin/python'))
        add('world inherits reviewed core',choose('world_server')==str(core/'bin/python'))
        add('workspace inherits reviewed core',choose('virtual_workspace')==str(core/'bin/python'))
        add('Discord retains supported Python after core promotion',choose('discord_bridge')==str(legacy),'compatibility')
        (core/'bin/python').unlink()
        add('missing promoted Python cannot silently revert versions',choose()==str(core/'bin/python'),'failure_boundary')
        (core/'.ina-ready').unlink()
        add('explicit marker rollback restores previous selection',choose()==str(legacy),'rollback')
    base={'PATH':'/usr/bin','EXAMPLE':'preserve'}
    prepared=current.python_environment('/fixture/venv/bin/python',base) if version!='V1' else dict(base)
    add('generic Python children resolve selected environment',prepared['PATH'].split(':')[0]=='/fixture/venv/bin','child_environment')
    add('child environment preserves unrelated settings',prepared['EXAMPLE']=='preserve' and base=={'PATH':'/usr/bin','EXAMPLE':'preserve'},'compatibility')
    return {'correct':sum(r['correct'] for r in rows),'total':len(rows),'cases':rows,
            'baseline_revision':BASELINE if version=='V1' else None,
            'scope':'pinned service command expression with current interpreter set to retained legacy; candidate real resolver with disposable paths; no runtime startup'}
