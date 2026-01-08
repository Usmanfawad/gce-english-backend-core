"""Authentication router providing signup, login, and current user endpoints."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, status

from app.db.supabase import SupabaseError, get_supabase_client
from app.api.auth.schemas import SignupRequest, LoginRequest, TokenResponse, UserResponse
from app.api.auth.security import (
    get_password_hash,
    verify_password,
    create_access_token,
    get_current_user,
)
from app.services.auth import (
    get_auth_setup_sql,
    get_user_by_email,
    create_user,
    UserAlreadyExistsError,
    AppUser,
)


router = APIRouter(prefix="/auth", tags=["auth"])


@router.post(
    "/signup",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new user account",
)
async def signup(payload: SignupRequest) -> UserResponse:
    """Create a new user in the Supabase `app_users` table."""
    try:
        password_hash = get_password_hash(payload.password)
        user = create_user(
            email=payload.email,
            password_hash=password_hash,
            full_name=payload.full_name,
        )
        # We don't currently fetch created_at from Supabase; it may be None.
        return UserResponse(
            id=user.id,
            email=user.email,
            full_name=user.full_name,
            created_at=None,
        )
    except UserAlreadyExistsError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A user with this email already exists",
        ) from None
    except SupabaseError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Log in and receive a JWT access token",
)
async def login(payload: LoginRequest) -> TokenResponse:
    """Authenticate a user by email and password and return a JWT access token."""
    try:
        user = get_user_by_email(payload.email)
    except SupabaseError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect email or password",
        )

    # Fetch the password_hash for this user directly from Supabase
    try:
        client = get_supabase_client()
        result = (
            client.table("app_users")
            .select("password_hash")
            .eq("id", user.id)
            .limit(1)
            .execute()
        )
        rows = result.data or []
        if not rows:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Incorrect email or password",
            )
        stored_hash = rows[0]["password_hash"]
    except HTTPException:
        # Re-raise HTTPException as is
        raise
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to verify credentials",
        ) from None

    if not verify_password(payload.password, stored_hash):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect email or password",
        )

    token_data: Dict[str, Any] = {"sub": user.id}
    access_token = create_access_token(token_data)
    return TokenResponse(access_token=access_token, token_type="bearer")


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get the current authenticated user",
)
async def read_current_user(current_user: AppUser = Depends(get_current_user)) -> UserResponse:
    """Return the currently authenticated user based on the JWT access token."""
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        full_name=current_user.full_name,
        created_at=None,
    )


@router.get(
    "/setup-sql",
    summary="Get SQL for creating the auth users table in Supabase",
)
async def auth_setup_sql() -> Dict[str, Any]:
    """Return SQL that must be run once in Supabase to create the `app_users` table."""
    return {
        "instructions": (
            "Run this SQL in Supabase Dashboard → SQL Editor to create the app_users table."
        ),
        "sql": get_auth_setup_sql(),
    }


