import json

import pytest
from app.gpu import output_protocol
from app.gpu.artifacts import ArtifactError


@pytest.mark.parametrize('native,valid,accepted', [(True, True, True), (True, False, False), (False, True, False)])
def test_native_readiness_report_stays_bound_to_trusted_snapshot(tmp_path, monkeypatch, native, valid, accepted):
    report = tmp_path / 'native-ready.json'
    report.write_text(json.dumps({'native_comfy': True, 'commit': 'a' * 40,
                                  'stage_seconds': {} if valid else None}))
    monkeypatch.setattr(output_protocol, 'verify_output', lambda job, item: (
        {'mime': 'application/json', 'size_bytes': report.stat().st_size}, report))
    job = {'snapshot': {'native_comfy': native, 'manifest': {'comfy_commit': 'a' * 40}}}
    if accepted:
        assert output_protocol.read_diagnostic_report(job, {})['native_comfy'] is True
    else:
        with pytest.raises(ArtifactError):
            output_protocol.read_diagnostic_report(job, {})
