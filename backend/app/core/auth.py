from __future__ import annotations

import hmac

from fastapi import Depends, Header, HTTPException

from app.core.config import Settings, get_settings


async def verify_api_key(
    x_api_key: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> None:
    if settings.api_key is None:
        return
    if x_api_key is None:
        raise HTTPException(status_code=401, detail="Missing X-Api-Key header")
    if not hmac.compare_digest(x_api_key, settings.api_key):
        raise HTTPException(status_code=403, detail="Invalid API key")
