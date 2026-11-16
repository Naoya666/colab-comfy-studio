"""Add independent model catalog, transfers and workflow bindings (no weight migration)."""
from alembic import op
from app.gpu.repository import ModelDependencyRow, ModelDownloadRow, ModelLibraryRow

revision = "0004_model_library"
down_revision = "0003_accounts_instances"
branch_labels = None
depends_on = None


def upgrade():
    for model in (ModelLibraryRow, ModelDownloadRow, ModelDependencyRow):
        model.__table__.create(op.get_bind(), checkfirst=True)


def downgrade():
    raise RuntimeError("模型库索引与引用不可自动删除；回退应用并保留增量表，或恢复已验证备份")
