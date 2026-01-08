"""Schemas for authentication endpoints."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field


class SignupRequest(BaseModel):
    """Request body for user signup."""

    email: EmailStr = Field(..., description="User email address")
    password: str = Field(
        ...,
        min_length=8,
        max_length=72,
        description="User password (8-72 characters; bcrypt limitation)",
    )
    full_name: Optional[str] = Field(
        default=None,
        description="Optional full name for the user",
    )


class LoginRequest(BaseModel):
    """Request body for user login."""

    email: EmailStr = Field(..., description="User email address")
    password: str = Field(
        ...,
        min_length=8,
        max_length=72,
        description="User password (8-72 characters; bcrypt limitation)",
    )


class TokenResponse(BaseModel):
    """Response containing JWT access token."""

    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type (always 'bearer')")


class UserResponse(BaseModel):
    """Public representation of a user."""

    id: str = Field(..., description="User ID (UUID)")
    email: EmailStr = Field(..., description="User email address")
    full_name: Optional[str] = Field(default=None, description="Full name of the user")
    created_at: Optional[datetime] = Field(
        default=None,
        description="User creation timestamp, if available",
    )


