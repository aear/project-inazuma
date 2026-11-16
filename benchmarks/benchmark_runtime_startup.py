"""Finite supervisor fixtures, pinned source; no real bridges or learned stores."""
import ast
from pathlib import Path
from types import SimpleNamespace
import queue
import time
from datetime import datetime, timezone

BASELINE = '6ddc708052af29c993925a935ed4f2dda2a3e49b'


def fixture(source, *, environment_ok=True):
    calls = []
    class Process:
        pid = 42
        def poll(self): return None
        def terminate(self): calls.append('terminate')
    class Timer:
        def __init__(self, delay, callback): self.delay=delay; self.callback=callback; self.daemon=False
        def start(self): calls.append(('timer',self.delay))
        def cancel(self): pass
    class Thread:
        def __init__(self,*a,**k): pass
        def start(self): pass
    def launch(*a,**k): calls.append('spawn'); return Process()
    nodes=[n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='RuntimeServiceSupervisor']
    tree=ast.parse('from __future__ import annotations'); tree.body.extend(nodes)
    env={'__file__':'/fixture/runtime_services.py', 'Path':Path, 'queue':queue, 'os':SimpleNamespace(getpid=lambda:1),
         'threading':SimpleNamespace(Thread=Thread,Timer=Timer), 'time':time,
         '_now':lambda:datetime.now(timezone.utc).isoformat(),
         'supervisor_status_path':lambda child:Path('/tmp/fixture-status.json'),
         'load_config':lambda:{}, 'SERVICE_COMMANDS':{'discord_bridge':['fixture-python','discord_bridge.py']},
         'python_environment':lambda exe:{}, 'project_python':lambda root,service=None:'fixture-python', 'safe_popen':launch, 'atomic_write_json':lambda *a,**k:None,
         'log_to_statusbox':lambda msg:calls.append(('log',msg)),
         'check_environment':lambda *a,**k:{'ok':environment_ok,'reason':'fixture_missing'},
         'environment_message':lambda report:'fixture environment missing', 'DISCORD_MODULES':('discord',)}
    exec(compile(ast.fix_missing_locations(tree),'<supervisor fixture>','exec'),env)
    return env['RuntimeServiceSupervisor']('Fixture'),calls


def cases(source):
    rows=[]
    def add(name,ok,component): rows.append({'case':name,'correct':bool(ok),'component':component})
    s,c=fixture(source,environment_ok=False);s._launch('discord_bridge')
    add('missing dependencies do not spawn', 'spawn' not in c,'environment')
    add('missing dependencies publish actionable blocked state',s.state['services'].get('discord_bridge',{}).get('status')=='blocked_environment','diagnostics')
    s,c=fixture(source);s._launch('discord_bridge');s._launch('discord_bridge')
    add('healthy duplicate launch suppressed',c.count('spawn')==1,'ownership')
    s,c=fixture(source);s.failures['discord_bridge']=5;s._schedule_restart('discord_bridge')
    add('crash budget stops automatic retries',not any(isinstance(x,tuple) and x[0]=='timer' for x in c),'retry_budget')
    add('crash pause visible',s.state['services'].get('discord_bridge',{}).get('status')=='blocked_crash_loop','diagnostics')
    s,c=fixture(source);s.failures['discord_bridge']=1;s._schedule_restart('discord_bridge')
    add('initial transient failure retains delayed retry',('timer',1.0) in c,'compatibility')
    s,c=fixture(source);s.failures['discord_bridge']=5;s._restart('discord_bridge')
    add('operator restart renews finite budget',s.failures['discord_bridge']==0 and ('timer',0.0) in c,'operator_choice')
    s,c=fixture(source);s._launch('discord_bridge');p=s.processes['discord_bridge'];s._restart('discord_bridge');s._handle_exit('discord_bridge',{'process':p,'returncode':0})
    add('operator restart waits for exit and preserves non-crash count','terminate' in c and s.failures['discord_bridge']==0 and ('timer',0.0) in c,'compatibility')
    s,c=fixture(source);s._launch('discord_bridge');p=s.processes['discord_bridge'];s._handle_exit('discord_bridge',{'process':p,'returncode':1})
    add('exit reason is logged',any(isinstance(x,tuple) and x[0]=='log' and 'code=1' in x[1] for x in c),'diagnostics')
    s,c=fixture(source);s.enabled['discord_bridge']=False;s._launch('discord_bridge')
    add('disabled service remains disabled','spawn' not in c and s.state['services']['discord_bridge']['status']=='disabled','compatibility')
    s,c=fixture(source);s.stopping=True;s._schedule_restart('discord_bridge')
    add('shutdown never schedules retry',not c,'compatibility')
    return rows


def assess(version):
    from historical_source import historical_text
    source=historical_text('runtime_services.py',BASELINE) if version=='V1' else (Path(__file__).resolve().parents[1]/'runtime_services.py').read_text()
    rows=cases(source)
    gui=historical_text('GUI.py',BASELINE) if version=='V1' else (Path(__file__).resolve().parents[1]/'GUI.py').read_text()
    before_imports=gui.split('import tkinter',1)[0]
    rows.append({'case':'project environment selected before GUI/runtime imports',
                 'component':'launcher_source_wiring','measurement_kind':'source_wiring',
                 'correct':'prepare_gui(Path(__file__).resolve().parent)' in before_imports})
    return {'correct':sum(x['correct'] for x in rows),'total':len(rows),'cases':rows,
            'baseline_revision':BASELINE if version=='V1' else None,
            'scope':'extracted real supervisor class with finite fake processes/timers; no live startup or delivery'}
