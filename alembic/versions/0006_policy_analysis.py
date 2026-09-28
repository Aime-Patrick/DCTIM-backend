from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "policy_artifacts",
        sa.Column("analysis", sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("policy_artifacts", "analysis")
