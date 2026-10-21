"""Adversarial local Caddy boundary tests: no host ports or live media."""
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.request
import urllib.error
import pytest


def test_caddy_read_only_routes_and_headers(tmp_path):
    binary=os.environ.get('INA_TEST_CADDY')
    if not binary:pytest.skip('verified Caddy test binary not mounted')
    public=tmp_path/'public';public.mkdir()
    (public/'media').mkdir()
    (public/'index.html').write_text('INA RADIO FIXTURE')
    (public/'station.json').write_text('{"clips":[]}')
    (tmp_path/'private_marker').write_text('SECRET MUST NEVER BE SERVED')
    ident='a'*64;(public/'media'/(ident+'.opus')).write_bytes(b'OggS-fixture')
    with socket.socket() as probe:
        probe.bind(('127.0.0.1',0));port=probe.getsockname()[1]
    template=(Path(__file__).resolve().parents[1]/'deployment/radio/Caddyfile.template').read_text()
    cfg=tmp_path/'Caddyfile';cfg.write_text(template.replace('SITE_ADDRESS',f'http://127.0.0.1:{port}').replace('/srv/radio',str(public)).replace('TLS_CONFIGURATION','').replace('BIND_CONFIGURATION','bind 127.0.0.1'))
    validate=subprocess.run([binary,'validate','--config',str(cfg),'--adapter','caddyfile'],capture_output=True,timeout=10)
    assert validate.returncode==0,validate.stderr.decode()
    with (tmp_path/'server.log').open('wb') as log:
        proc=subprocess.Popen([binary,'run','--config',str(cfg),'--adapter','caddyfile'],stdout=log,stderr=log,
                              env={**os.environ,'XDG_DATA_HOME':str(tmp_path/'data'),'XDG_CONFIG_HOME':str(tmp_path/'config'),'GOMAXPROCS':'2'})
        try:
            base=f'http://127.0.0.1:{port}'
            def request(path,method='GET',headers=None):
                req=urllib.request.Request(base+path,method=method,headers=headers or {})
                try:return urllib.request.urlopen(req,timeout=3)
                except urllib.error.HTTPError as error:return error
            for _ in range(50):
                try:
                    response=request('/')
                    if response.status==200:break
                except OSError:pass
                time.sleep(.05)
            else:raise AssertionError('Caddy did not start')
            assert b'INA RADIO FIXTURE' in response.read()
            assert "frame-ancestors 'none'" in response.headers['Content-Security-Policy']
            assert response.headers['X-Content-Type-Options']=='nosniff'
            assert response.headers['Cache-Control']=='no-store'
            for path in ('/../private_marker','/%2e%2e/private_marker','/%252e%252e/private_marker',
                         '/catalog.json','/admin/','/config/','/metrics','/media/','/foo.wav',
                         '/media/'+ident+'.opus.bak','/station.json/private_marker'):
                denied=request(path)
                assert denied.status>=400,path
                assert b'SECRET MUST NEVER BE SERVED' not in denied.read()
            for method in ('POST','PUT','DELETE','OPTIONS','PATCH'):
                assert request('/station.json',method=method).status==405
            partial=request('/media/'+ident+'.opus',headers={'Range':'bytes=0-3'})
            assert partial.status==206 and partial.read()==b'OggS'
            assert request('/station.json').status==200
            assert request('/media/'+ident+'.opus',method='HEAD').status==200
        finally:
            proc.terminate()
            try:proc.wait(timeout=5)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
