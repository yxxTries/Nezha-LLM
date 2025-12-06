from __future__ import annotations
import logging
from logging.config import dictConfig
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .routes import router
from .deps import get_asr_service, get_llm_service

LOG_CONFIG = {
  "version": 1,
  "disable_existing_loggers": False,
  "formatters": {"default": {"format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s"}},
  "handlers": {"default": {"class": "logging.StreamHandler", "formatter": "default"}},
  "root": {"level": settings.log_level.upper(), "handlers": ["default"]},
}

def create_app() -> FastAPI:
    dictConfig(LOG_CONFIG)
    app = FastAPI(title="Nezha-LLM Local API")
    app.add_middleware(CORSMiddleware,
                       allow_origins=settings.cors_allow_origins,
                       allow_credentials=settings.cors_allow_credentials,
                       allow_methods=settings.cors_allow_methods,
                       allow_headers=settings.cors_allow_headers)

    @app.on_event("startup")
    async def preload_models() -> None:
        # Preload ASR and LLM so the first request is fast; services are cached in deps
        get_asr_service()
        get_llm_service()

    app.include_router(router)
    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=settings.host, port=settings.port, reload=True)
