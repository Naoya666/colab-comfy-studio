"""Persist the keys used by the creation ComfyUI facade."""

from alembic import op
from app.db import CreationApiKeyRow

revision = "0006_creation_api"
down_revision = "0005_native_api"
branch_labels = None
depends_on = None


def upgrade():
    CreationApiKeyRow.__table__.create(op.get_bind(), checkfirst=True)


def downgrade():
    raise RuntimeError("创作 API 密钥需保留；请回退应用或恢复备份")
