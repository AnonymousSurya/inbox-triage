from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

# Load .env for local development; docker-compose passes vars directly.
_env = Path(__file__).resolve().parent.parent / ".env"
if _env.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env, override=False)
    except ImportError:
        pass

from app.api.routes import router
from app.services.triage import build_triage_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.triage = build_triage_service()
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="Inbox triage", lifespan=lifespan)
    app.include_router(router)
    return app


app = create_app()
