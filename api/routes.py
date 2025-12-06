from __future__ import annotations
import logging, tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, status

from asr.service import ASRService
from llm.service import LLMService
from llm.types import LLMRequest

from .deps import get_asr_service, get_llm_service
from .schemas import TextRequest, LLMReply, ASRLLMReply

logger = logging.getLogger(__name__)

router = APIRouter()

def utcnow() -> datetime:
    return datetime.now(timezone.utc)

@router.post("/api/llm", response_model=LLMReply)
async def run_llm(payload: TextRequest, llm: Annotated[LLMService, Depends(get_llm_service)]) -> LLMReply:
    try:
        req = LLMRequest(text=payload.message,
                         max_new_tokens=payload.max_new_tokens,
                         temperature=payload.temperature)
        rep = llm.generate(req)
        return LLMReply(text=rep.text, timestamp=utcnow())
    except Exception as e:
        logger.exception("LLM failed")
        raise HTTPException(500, "LLM failure") from e

@router.post("/api/asr-llm", response_model=ASRLLMReply)
async def asr_llm(
    asr: Annotated[ASRService, Depends(get_asr_service)],
    llm: Annotated[LLMService, Depends(get_llm_service)],
    audio: UploadFile = File(...),
) -> ASRLLMReply:
    tmp = None
    try:
        suffix = Path(audio.filename).suffix or ".webm"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            tmp = Path(f.name)
            f.write(await audio.read())

        asr_res = asr.transcribe_file(str(tmp))
        transcript = asr_res.text.strip()

        # Guard: empty or too-short audio -> skip LLM
        min_duration_s = 0.3
        if not transcript or (getattr(asr_res, "duration", 0.0) or 0.0) < min_duration_s:   
            msg = "No speech detected; please try again."
            return ASRLLMReply(text=msg, transcript=transcript, timestamp=utcnow())

        req = LLMRequest(text=transcript)
        llm_res = llm.generate(req)

        return ASRLLMReply(text=llm_res.text, transcript=transcript, timestamp=utcnow())
    except Exception as e:
        logger.exception("ASR+LLM error")
        raise HTTPException(500, "ASR+LLM pipeline failed") from e
    finally:
        if tmp:
            tmp.unlink(missing_ok=True)
