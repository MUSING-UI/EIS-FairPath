"""Data verification must detect corruption before a numerical run."""
import hashlib
import json

from eis_fairpath.cli import main


def data_root(tmp_path):
    (tmp_path / "data").mkdir()
    data = tmp_path / "data" / "tiny.json"
    data.write_bytes(b'{"papers": []}\n')
    manifest = {"files": [{"path": "data/tiny.json", "bytes": data.stat().st_size,
                           "sha256": hashlib.sha256(data.read_bytes()).hexdigest()}]}
    (tmp_path / "data" / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path, data


def test_verify_accepts_original_bytes(tmp_path, capsys):
    root, _ = data_root(tmp_path)
    assert main(["verify-data", "--root", str(root)]) == 0
    assert "verified" in capsys.readouterr().out.lower()


def test_verify_rejects_changed_bytes(tmp_path, capsys):
    root, data = data_root(tmp_path)
    data.write_bytes(b'{"papers": [1]}\n')
    assert main(["verify-data", "--root", str(root)]) == 2
    assert "mismatch" in capsys.readouterr().err.lower()
