import pytest
from app.instance_routes import RetentionInput
from pydantic import ValidationError


@pytest.mark.parametrize('body', [
    {'retention_mode': 'release'},
    {'retention_mode': 'retain', 'idle_minutes': -1},
])
def test_retention_rejects_unsupported_http_values(body):
    with pytest.raises(ValidationError):
        RetentionInput.model_validate(body)


def test_retention_request_accepts_long_duration_for_generation_instances():
    request = RetentionInput.model_validate({'retention_mode': 'retain', 'idle_minutes': 120})
    assert request.idle_minutes == 120
