"""Install per-user radio files and sandboxed Caddy services; no firewall or DNS edits.

Run only with an official checksum-verified Caddy binary. HTTPS is prepared but
never started here; the localhost preview is a separate, explicit service.
"""
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile


def atomic_copy(source: Path, destination: Path, mode: int):
    with tempfile.NamedTemporaryFile(dir=destination.parent,prefix='.radio-update-',delete=False) as handle:
        staged=Path(handle.name)
    try:
        shutil.copyfile(source,staged);staged.chmod(mode);os.replace(staged,destination)
    finally:staged.unlink(missing_ok=True)


def install(repo: Path, binary: Path, *, home: Path):
    root=home/'.local/share/ina-radio';public=root/'public'
    if root.exists():
        if not (root/'installation.json').is_file():raise ValueError('existing unrelated radio directory; inspect first')
    for path in [root,public,public/'media',root/'server-data',root/'server-config',home/'.local/bin',home/'.config/systemd/user']:
        if path.is_symlink() or any(p.is_symlink() for p in path.parents):raise PermissionError('symlink installation root')
        path.mkdir(parents=True,exist_ok=True,mode=0o700)
    ca_bundle=next((Path(p) for p in ('/etc/pki/tls/certs/ca-bundle.crt','/etc/ssl/certs/ca-certificates.crt',
        '/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem') if Path(p).is_file()),None)
    if ca_bundle is None:raise RuntimeError('trusted CA bundle unavailable; HTTPS setup refused')
    (root/'installation.json').write_text(json.dumps({'schema':'ina.radio_installation/V1','phase':'validating','public_started':False}))
    target=home/'.local/bin/ina-radio-caddy'
    if target.exists():
        # An update is an explicit operator installation, not a public request.
        if target.is_symlink():raise PermissionError('binary symlink')
    atomic_copy(binary,target,0o755)
    for name in ('index.html','player.js','style.css'):
        p=public/name
        if p.is_symlink():raise PermissionError('web asset symlink')
        atomic_copy(repo/'radio_web'/name,p,0o600)
    if not (public/'station.json').exists():
        (public/'station.json').write_text(json.dumps({'schema':'ina.radio_program/V1','clips':[],'repeat':False}))
        (public/'station.json').chmod(0o600)
    template=(repo/'deployment/radio/Caddyfile.template').read_text()
    configs={}
    for kind,address in [('preview','http://127.0.0.1:8787'),('public','radio.ai-nation.co.uk')]:
        tls='' if kind=='preview' else 'tls {\n issuer acme {\n disable_http_challenge\n }\n }'
        cfg=root/(kind+'.Caddyfile');cfg.write_text(template.replace('SITE_ADDRESS',address).replace('TLS_CONFIGURATION',tls).replace('BIND_CONFIGURATION','bind 127.0.0.1' if kind=='preview' else 'bind 0.0.0.0'));cfg.chmod(0o600)
        configs[kind]=cfg
        commands=['/usr/bin/bwrap','--unshare-all','--share-net','--die-with-parent','--new-session','--clearenv',
            '--ro-bind','/usr','/usr','--symlink','usr/bin','/bin','--symlink','usr/lib','/lib','--symlink','usr/lib64','/lib64',
            '--proc','/proc','--dev','/dev','--tmpfs','/tmp','--dir','/etc',
            '--ro-bind','/etc/ld.so.cache','/etc/ld.so.cache',
            '--ro-bind','/etc/resolv.conf','/etc/resolv.conf',
            '--dir','/etc/ssl','--dir','/etc/ssl/certs',
            '--ro-bind',str(ca_bundle),'/etc/ssl/certs/ca-certificates.crt',
            '--ro-bind',str(target),'/opt/caddy','--ro-bind',str(public),'/srv/radio',
            '--ro-bind',str(cfg),'/etc/Caddyfile','--bind',str(root/'server-data'),'/data',
            '--bind',str(root/'server-config'),'/config','--chdir','/tmp',
            '--setenv','HOME','/tmp','--setenv','XDG_DATA_HOME','/data','--setenv','XDG_CONFIG_HOME','/config',
            '--setenv','PATH','/usr/bin:/bin','--setenv','GOMAXPROCS','2',
            '--setenv','SSL_CERT_FILE','/etc/ssl/certs/ca-certificates.crt',
            '/opt/caddy','run','--config','/etc/Caddyfile','--adapter','caddyfile']
        # systemd ExecStart has its own quoting, not shell expansion. Percent must be escaped.
        quoted=' '.join('"'+arg.replace('\\','\\\\').replace('"','\\"').replace('%','%%')+'"' for arg in commands)
        unit=home/'.config/systemd/user'/('ina-radio-'+kind+'.service')
        unit.write_text('[Unit]\nDescription=Ina Radio '+kind+' (static listener only)\nStartLimitIntervalSec=60\nStartLimitBurst=3\n\n[Service]\nType=simple\nExecStart='+quoted+'\nRestart=on-failure\nRestartSec=10\nNoNewPrivileges=yes\nUMask=0077\nMemoryMax=192M\nCPUQuota=50%\nTasksMax=64\nLimitNOFILE=128\nKillMode=control-group\nTimeoutStopSec=10\n\n[Install]\nWantedBy=default.target\n')
        # Validate the exact deployment config inside the same filesystem boundary.
        validation=commands.copy()
        pos=max(i for i,value in enumerate(validation) if value=='/opt/caddy');validation=validation[:pos]+['/opt/caddy','validate','--config','/etc/Caddyfile','--adapter','caddyfile']
        subprocess.run(validation,check=True,capture_output=True,timeout=15)
    config=repo/'config.json'
    if config.is_symlink():raise PermissionError('runtime config symlink')
    if not config.is_file():raise ValueError('local runtime config is missing; source-only setup cannot enable tools')
    original=config.read_bytes();backup=root/'config-before-radio.json'
    if not backup.exists():backup.write_bytes(original);backup.chmod(0o600)
    data=json.loads(original);data['radio_station']={'enabled':True,'root':str(root),'public_url':'https://radio.ai-nation.co.uk/'}
    staged=config.with_name('.config.radio-install.tmp')
    with staged.open('x') as handle:json.dump(data,handle,ensure_ascii=False,indent=2);handle.write('\n')
    staged.chmod(0o600);os.replace(staged,config)
    (root/'installation.json').write_text(json.dumps({'schema':'ina.radio_installation/V1','public_url':'https://radio.ai-nation.co.uk/',
        'preview_url':'http://127.0.0.1:8787/','public_port':8443,'router_external_port':443,'public_started':False},indent=2))
    subprocess.run(['systemctl','--user','daemon-reload'],check=True,timeout=15)
    return {'root':str(root),'preview_url':'http://127.0.0.1:8787/','public_started':False}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--caddy',type=Path,required=True);a=p.parse_args()
    print(json.dumps(install(Path(__file__).resolve().parents[1],a.caddy,home=Path.home()),indent=2))

if __name__=='__main__':main()
