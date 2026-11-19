"""Persist native ComfyUI integration keys and prompt ownership."""

from alembic import op
from app.db import NativeApiKeyRow, NativeApiPromptRow

revision = "0005_native_api"
down_revision = "0004_model_library"
branch_labels = None
depends_on = None


def upgrade():
    for model in (NativeApiKeyRow, NativeApiPromptRow):
        model.__table__.create(op.get_bind(), checkfirst=True)


def downgrade():
    raise RuntimeError("原生 API 密钥及任务归属记录需保留；请回退应用或恢复备份")
