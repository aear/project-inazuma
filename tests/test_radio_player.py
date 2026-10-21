from pathlib import Path
import shutil
import subprocess
import pytest


def test_player_timeline_and_safe_rendering():
    node=shutil.which('node')
    if not node:pytest.skip('Node unavailable for player VM check')
    root=Path(__file__).resolve().parents[1]
    result=subprocess.run([node,str(root/'tests/radio_player_check.js'),str(root/'radio_web/player.js')],capture_output=True,text=True,timeout=10)
    assert result.returncode==0,result.stdout+result.stderr
