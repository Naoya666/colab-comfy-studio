"""Add Telegram notification outbox table."""

from alembic import op
from app.db import TelegramNotificationRow

revision = "0008_telegram_notifications"
down_revision = "0007_cu_ledger"
branch_labels = None
depends_on = None


def upgrade():
    connection = op.get_bind()
    TelegramNotificationRow.__table__.create(connection, checkfirst=True)


def downgrade():
    connection = op.get_bind()
    TelegramNotificationRow.__table__.drop(connection, checkfirst=True)
