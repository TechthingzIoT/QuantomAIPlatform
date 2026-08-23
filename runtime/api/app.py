"""
QAIR API Application
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from runtime.api import routes


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage the QAIR runtime lifecycle.

    The API process owns runtime startup and shutdown:
    - startup loads and prepares the active model
    - shutdown releases inference resources
    """
    runtime = routes.runtime

    runtime.start()

    try:
        yield
    finally:
        runtime.stop()


app = FastAPI(
    title="QAIR API",
    version="0.6.0",
    description="Quantom AI Runtime API",
    lifespan=lifespan,
)

app.include_router(routes.router)
