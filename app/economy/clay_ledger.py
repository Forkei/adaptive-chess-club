"""$CLAY balance ledger.

V1: SQLite-backed internal ledger. All wagering happens within Chess Club's DB.
Future: when Bhaven's Metropolis backend exposes an internal balance system, OR
when on-chain wagering is built, the implementation behind this interface swaps
without changing call sites.

All call sites import via:
    from app.economy.clay_ledger import get_ledger, InsufficientFunds
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import select

from app.models.clay_balance import ClayBalance
from app.models.clay_transaction import ClayTransaction

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class InsufficientFunds(Exception):
    """Raised by ClayLedger.debit / transfer when balance < amount."""


class ClayLedger:
    """Abstract interface — concrete impl is SqliteClayLedger for now."""

    def get_balance(self, player_id: str) -> int:
        raise NotImplementedError

    def debit(
        self, player_id: str, amount: int, reason: str, match_id: str | None = None
    ) -> ClayTransaction:
        raise NotImplementedError

    def credit(
        self, player_id: str, amount: int, reason: str, match_id: str | None = None
    ) -> ClayTransaction:
        raise NotImplementedError

    def transfer(
        self,
        from_id: str,
        to_id: str,
        amount: int,
        reason: str,
        match_id: str | None = None,
    ) -> tuple[ClayTransaction, ClayTransaction]:
        raise NotImplementedError

    def transactions_for_player(
        self,
        player_id: str,
        limit: int = 50,
        reason: str | None = None,
    ) -> list[ClayTransaction]:
        raise NotImplementedError


class SqliteClayLedger(ClayLedger):
    """Internal SQLite-backed implementation.

    Each public method opens its own session and commits (or rolls back on
    error). All balance + transaction inserts happen in one DB transaction
    so the audit log is always consistent with the running total.
    """

    def __init__(self, session_factory) -> None:
        self._factory = session_factory

    # --- Internal helpers --------------------------------------------------

    def _ensure_balance(self, session, player_id: str) -> ClayBalance:
        """Return the ClayBalance row, creating it (balance=0) if absent."""
        row = session.get(ClayBalance, player_id)
        if row is None:
            row = ClayBalance(
                player_id=player_id,
                balance=0,
                updated_at=datetime.utcnow(),
            )
            session.add(row)
            session.flush()
        return row

    # --- Public interface --------------------------------------------------

    def get_balance(self, player_id: str) -> int:
        with self._factory() as session:
            row = session.get(ClayBalance, player_id)
            return row.balance if row else 0

    def debit(
        self, player_id: str, amount: int, reason: str, match_id: str | None = None
    ) -> ClayTransaction:
        if amount <= 0:
            raise ValueError(f"debit amount must be positive, got {amount}")
        with self._factory() as session:
            bal = self._ensure_balance(session, player_id)
            if bal.balance < amount:
                raise InsufficientFunds(
                    f"Balance {bal.balance} < required {amount} for player {player_id}"
                )
            bal.balance -= amount
            bal.updated_at = datetime.utcnow()
            txn = ClayTransaction(
                player_id=player_id,
                amount=-amount,
                balance_after=bal.balance,
                reason=reason,
                related_match_id=match_id,
                created_at=datetime.utcnow(),
            )
            session.add(txn)
            session.commit()
            session.refresh(txn)
            return txn

    def credit(
        self, player_id: str, amount: int, reason: str, match_id: str | None = None
    ) -> ClayTransaction:
        if amount < 0:
            raise ValueError(f"credit amount must be non-negative, got {amount}")
        with self._factory() as session:
            bal = self._ensure_balance(session, player_id)
            bal.balance += amount
            bal.updated_at = datetime.utcnow()
            txn = ClayTransaction(
                player_id=player_id,
                amount=amount,
                balance_after=bal.balance,
                reason=reason,
                related_match_id=match_id,
                created_at=datetime.utcnow(),
            )
            session.add(txn)
            session.commit()
            session.refresh(txn)
            return txn

    def transfer(
        self,
        from_id: str,
        to_id: str,
        amount: int,
        reason: str,
        match_id: str | None = None,
    ) -> tuple[ClayTransaction, ClayTransaction]:
        """Atomic two-sided transfer: debit from_id, credit to_id.

        Either both succeed or neither does (single DB transaction).
        """
        if amount <= 0:
            raise ValueError(f"transfer amount must be positive, got {amount}")
        with self._factory() as session:
            from_bal = self._ensure_balance(session, from_id)
            to_bal = self._ensure_balance(session, to_id)
            if from_bal.balance < amount:
                raise InsufficientFunds(
                    f"Balance {from_bal.balance} < required {amount} for player {from_id}"
                )
            from_bal.balance -= amount
            from_bal.updated_at = datetime.utcnow()
            to_bal.balance += amount
            to_bal.updated_at = datetime.utcnow()

            now = datetime.utcnow()
            debit_txn = ClayTransaction(
                player_id=from_id,
                amount=-amount,
                balance_after=from_bal.balance,
                reason=reason,
                related_match_id=match_id,
                created_at=now,
            )
            credit_txn = ClayTransaction(
                player_id=to_id,
                amount=amount,
                balance_after=to_bal.balance,
                reason=reason,
                related_match_id=match_id,
                created_at=now,
            )
            session.add(debit_txn)
            session.add(credit_txn)
            session.commit()
            session.refresh(debit_txn)
            session.refresh(credit_txn)
            return debit_txn, credit_txn

    def transactions_for_player(
        self,
        player_id: str,
        limit: int = 50,
        reason: str | None = None,
    ) -> list[ClayTransaction]:
        with self._factory() as session:
            stmt = select(ClayTransaction).where(
                ClayTransaction.player_id == player_id
            )
            if reason is not None:
                stmt = stmt.where(ClayTransaction.reason == reason)
            stmt = stmt.order_by(ClayTransaction.created_at.desc()).limit(limit)
            return list(session.execute(stmt).scalars())


# --- Module-level singleton -----------------------------------------------
# Initialized lazily on first call. Startup code in main.py calls get_ledger()
# once so any misconfiguration surfaces early.

_ledger: SqliteClayLedger | None = None


def get_ledger() -> SqliteClayLedger:
    global _ledger
    if _ledger is None:
        from app.db import SessionLocal

        _ledger = SqliteClayLedger(SessionLocal)
    return _ledger
