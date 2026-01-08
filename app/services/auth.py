"""Auth service for managing users in Supabase."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

from loguru import logger

from app.db.supabase import get_supabase_client, SupabaseError


class AuthError(RuntimeError):
    """Raised when authentication-related operations fail."""


class UserAlreadyExistsError(AuthError):
    """Raised when attempting to create a user with an existing email."""


class UserNotFoundError(AuthError):
    """Raised when a user cannot be found."""


AUTH_SETUP_SQL = """
-- ===========================================
-- GCE English Backend - Auth Users Table
-- Run this ONCE in Supabase Dashboard → SQL Editor
-- ===========================================

CREATE TABLE IF NOT EXISTS app_users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    full_name TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS app_users_email_idx ON app_users (email);
"""


def get_auth_setup_sql() -> str:
    """Return the SQL needed to set up the auth users table."""
    return AUTH_SETUP_SQL


@dataclass
class AppUser:
    """Internal representation of an application user."""

    id: str
    email: str
    full_name: Optional[str] = None

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "AppUser":
        return cls(
            id=row["id"],
            email=row["email"],
            full_name=row.get("full_name"),
        )


def _get_users_table():
    """Return the Supabase table reference for app users."""
    client = get_supabase_client()
    return client.table("app_users")


def get_user_by_email(email: str) -> Optional[AppUser]:
    """Fetch a user by email, or return None if not found."""
    try:
        table = _get_users_table()
        result = table.select("*").eq("email", email).limit(1).execute()
        rows = result.data or []
        if not rows:
            return None
        return AppUser.from_row(rows[0])
    except Exception as exc:  # pragma: no cover - defensive
        msg = str(exc).lower()
        if "does not exist" in msg:
            logger.error(
                "Supabase table 'app_users' not found. "
                "Run the SQL from /auth/setup-sql in Supabase SQL Editor."
            )
            raise SupabaseError(
                "Auth users table not found. "
                "Please run the SQL from /auth/setup-sql in Supabase Dashboard."
            ) from exc
        raise SupabaseError(f"Failed to fetch user by email: {exc}") from exc


def get_user_by_id(user_id: str) -> Optional[AppUser]:
    """Fetch a user by id, or return None if not found."""
    try:
        table = _get_users_table()
        result = table.select("*").eq("id", user_id).limit(1).execute()
        rows = result.data or []
        if not rows:
            return None
        return AppUser.from_row(rows[0])
    except Exception as exc:  # pragma: no cover - defensive
        msg = str(exc).lower()
        if "does not exist" in msg:
            logger.error(
                "Supabase table 'app_users' not found. "
                "Run the SQL from /auth/setup-sql in Supabase SQL Editor."
            )
            raise SupabaseError(
                "Auth users table not found. "
                "Please run the SQL from /auth/setup-sql in Supabase Dashboard."
            ) from exc
        raise SupabaseError(f"Failed to fetch user by id: {exc}") from exc


def create_user(
    *,
    email: str,
    password_hash: str,
    full_name: Optional[str] = None,
) -> AppUser:
    """Create a new user in Supabase.

    Raises:
        UserAlreadyExistsError: if a user with this email already exists.
        SupabaseError: on Supabase-related failures.
    """
    existing = get_user_by_email(email)
    if existing:
        raise UserAlreadyExistsError(f"User with email {email} already exists")

    try:
        table = _get_users_table()
        payload: Dict[str, Any] = {
            "email": email,
            "password_hash": password_hash,
        }
        if full_name:
            payload["full_name"] = full_name

        # Supabase Python client returns inserted rows directly from insert().execute()
        result = table.insert(payload).execute()
        rows = result.data or []
        if not rows:
            raise SupabaseError("Failed to create user: no data returned from Supabase insert")
        user = AppUser.from_row(rows[0])
        logger.info(f"Created new app user {user.email} ({user.id})")
        return user
    except SupabaseError:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        msg = str(exc).lower()
        if "duplicate key value" in msg or "already exists" in msg:
            raise UserAlreadyExistsError(f"User with email {email} already exists") from exc
        if "does not exist" in msg:
            logger.error(
                "Supabase table 'app_users' not found. "
                "Run the SQL from /auth/setup-sql in Supabase SQL Editor."
            )
            raise SupabaseError(
                "Auth users table not found. "
                "Please run the SQL from /auth/setup-sql in Supabase Dashboard."
            ) from exc
        raise SupabaseError(f"Failed to create user: {exc}") from exc


def authenticate_user(email: str, password_verifier) -> Optional[AppUser]:
    """Authenticate a user using a provided password verifier callback.

    Args:
        email: User email.
        password_verifier: Callable that takes (email) and returns AppUser if password matches,
            otherwise None. Implemented in the API layer where the plain password is available.
    """
    # The actual authentication is handled in the API layer where we have the plain password.
    # This function is kept for consistency and future extension.
    return password_verifier(email)


