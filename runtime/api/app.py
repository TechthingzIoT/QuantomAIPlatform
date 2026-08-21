"""
QAIR API Application
"""

from __future__ import annotations

from fastapi import FastAPI

from runtime.api.routes import router


app = FastAPI(
    title="QAIR API",
    version="0.6.0",
    description="Quantom AI Runtime API",
)

app.include_router(router)