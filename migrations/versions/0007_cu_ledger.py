"""Persist the Colab CU balance ledger and its legacy accounting cutover."""

from alembic import op
from app.gpu.repository import GpuCcuLedgerRow, ensure_cu_cutover

revision = "0007_cu_ledger"
down_revision = "0006_creation_api"
branch_labels = None
depends_on = None


def upgrade():
    connection = op.get_bind()
    GpuCcuLedgerRow.__table__.create(connection, checkfirst=True)
    ensure_cu_cutover(connection)


def downgrade():
    raise RuntimeError("CU 余额流水需保留；请回退应用或恢复备份")
