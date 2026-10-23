# """
# Enhanced FastAPI application with comprehensive error handling and logging.
# """

# import os
# import logging
# import time
# import traceback
# import json
# from pathlib import Path
# from dotenv import load_dotenv
# from datetime import datetime

# from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# from pydantic import BaseModel

# from chat.chat import router as chat_router
# from chat.knowledge_base import router as kb_router
# from chat.knowledge_base import process_kb_entry
# from dummy_chat.dummy_kb_ingest import ingest_kb
# from dummy_chat.dummy_kb_retrieve import retrieve_kb_chunks
# from dummy_chat.dummy_kb_answer import answer_from_kb

# # ═══════════════════════════════════════════════════════════════════
# # LOAD ENVIRONMENT
# # ═══════════════════════════════════════════════════════════════════
# env_path = Path(__file__).parent / ".env"
# load_dotenv(env_path, override=True)

# # ═══════════════════════════════════════════════════════════════════
# # LOGGING SETUP
# # ═══════════════════════════════════════════════════════════════════
# LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# logging.basicConfig(
#     level=getattr(logging, LOG_LEVEL, logging.INFO),
#     format="%(asctime)s | %(levelname)-8s | %(name)-15s | %(message)s",
#     datefmt="%Y-%m-%d %H:%M:%S",
# )

# log = logging.getLogger("app")
# log.info("=" * 80)
# log.info("🚀 APPLICATION STARTUP")
# log.info("=" * 80)
# log.info("Environment file: %s", env_path)
# log.info("Log level: %s", LOG_LEVEL)

# # ═══════════════════════════════════════════════════════════════════
# # APP INITIALIZATION
# # ═══════════════════════════════════════════════════════════════════
# app = FastAPI(
#     title="AI Calling API with RAG",
#     version="2.0.0",
#     description="Enhanced chatbot with vector search knowledge base",
# )

# # CORS Configuration
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ═══════════════════════════════════════════════════════════════════
# # REQUEST LOGGING MIDDLEWARE (Detailed + Safe Async Response)
# # ═══════════════════════════════════════════════════════════════════


# @app.middleware("http")
# async def log_requests(request: Request, call_next):
#     """Logs all HTTP requests and responses with detailed trace output."""
#     request_id = f"{int(time.time() * 1000)}"
#     client_ip = request.client.host
#     path = request.url.path
#     method = request.method

#     log.info("→ REQUEST | id=%s | method=%s | path=%s | client=%s",
#              request_id, method, path, client_ip)

#     start_time = time.perf_counter()

#     try:
#         response = await call_next(request)
#         duration_ms = (time.perf_counter() - start_time) * 1000

#         # Capture and log the raw response body safely
#         body_text = ""
#         if hasattr(response, "body_iterator"):
#             body_content = b""
#             async for chunk in response.body_iterator:
#                 body_content += chunk
#             body_text = body_content.decode("utf-8", errors="ignore").strip()

#             # ✅ FIXED: restore async body iterator properly
#             async def new_body_iterator():
#                 yield body_content
#             response.body_iterator = new_body_iterator()
#             response.headers["content-length"] = str(len(body_content))

#         else:
#             body_content = getattr(response, "body", b"")
#             if isinstance(body_content, str):
#                 body_text = body_content
#                 body_content = body_content.encode("utf-8")
#             else:
#                 body_text = body_content.decode(
#                     "utf-8", errors="ignore").strip()
#             response.headers["content-length"] = str(len(body_content))

#         log.info("← RESPONSE | id=%s | status=%d | time=%.1fms",
#                  request_id, response.status_code, duration_ms)
#         log.info("  ▸ Raw Response Body:")
#         for line in body_text.splitlines():
#             log.info("    %s", line)

#         return response

#     except Exception as e:
#         duration_ms = (time.perf_counter() - start_time) * 1000
#         log.error("✖ REQUEST FAILED | id=%s | time=%.1fms | error=%s",
#                   request_id, duration_ms, str(e))
#         raise

# # ═══════════════════════════════════════════════════════════════════
# # GLOBAL EXCEPTION HANDLER
# # ═══════════════════════════════════════════════════════════════════


# @app.exception_handler(Exception)
# async def global_exception_handler(request: Request, exc: Exception):
#     """Catch-all exception handler with detailed logging."""
#     log.error("=" * 80)
#     log.error("UNHANDLED EXCEPTION")
#     log.error("=" * 80)
#     log.error("Path: %s %s", request.method, request.url.path)
#     log.error("Error: %s", str(exc))
#     log.error("Traceback:\n%s", traceback.format_exc())
#     log.error("=" * 80)

#     return JSONResponse(
#         status_code=500,
#         content={
#             "ok": False,
#             "error": "Internal server error",
#             "message": str(exc),
#             "path": str(request.url.path),
#             "timestamp": datetime.utcnow().isoformat() + "Z",
#         }
#     )

# # ═══════════════════════════════════════════════════════════════════
# # INCLUDE ROUTERS
# # ═══════════════════════════════════════════════════════════════════
# log.info("Loading routers...")

# try:
#     app.include_router(chat_router, prefix="/chat", tags=["Chat"])
#     app.include_router(kb_router)
#     log.info("✓ Chat router loaded")
# except Exception as e:
#     log.error("✗ Failed to load chat router: %s", e)
#     raise

# # Optional: Voice router (if available)
# try:
#     from voice.voice_router import router as voice_router
#     app.include_router(voice_router, tags=["Voice"])
#     log.info("✓ Voice router loaded")
# except ImportError:
#     log.warning("⚠ Voice router not available (optional)")
# except Exception as e:
#     log.error("✗ Failed to load voice router: %s", e)

# # ═══════════════════════════════════════════════════════════════════
# # MODELS
# # ═══════════════════════════════════════════════════════════════════


# class KBRequest(BaseModel):
#     format: str
#     status: str | None = None
#     data: dict | str | None = None
#     filename: str | None = None

# # ═══════════════════════════════════════════════════════════════════
# # KNOWLEDGE BASE ENDPOINT
# # ═══════════════════════════════════════════════════════════════════


# @app.post("/kb/train", tags=["Knowledge Base"])
# async def kb_train(
#     request: Request,
#     format: str | None = Form(None),
#     status: str | None = Form(None),
#     file: UploadFile | None = None,
#     user_id: str | None = Form(None),
#     workspace_id: str | None = Form(None),
# ):
#     """Upload knowledge base content in various formats."""
#     log.info("=" * 80)
#     log.info("📚 KB TRAINING REQUEST")
#     log.info("=" * 80)

#     ctype = request.headers.get("content-type", "")
#     log.info("Content-Type: %s", ctype)

#     try:
#         # Handle multipart/form-data
#         if ctype and "multipart/form-data" in ctype:
#             if not format:
#                 form = await request.form()
#                 format = (form.get("format") or "").strip()
#                 status = status or form.get("status")
#                 # data might be in form field 'data'
#                 data_field = form.get("data")
#                 # if user_id/workspace_id were not in Form(...) signature, try form
#                 user_id = user_id or form.get("user_id")
#                 workspace_id = workspace_id or form.get("workspace_id")
#             else:
#                 # signature-supplied form values exist; still read `data` from form for safety
#                 form = await request.form()
#                 data_field = form.get("data") if form else None

#             fmt = format.lower().strip()
#             log.info("Format: %s (multipart)", fmt)

#             if fmt not in {"qna", "file", "link"}:
#                 raise HTTPException(
#                     400, "Invalid format: must be qna, file, or link")

#             if not user_id:
#                 raise HTTPException(400, "user_id is required")

#             # ⚙️ Simulate role (user_id/workspace_id come from request)
#             user_role = "admin"  # will come from JWT/session in future

#             if fmt == "file":
#                 if not file:
#                     raise HTTPException(
#                         400, "No file uploaded for format=file")
#                 log.info("Processing file: %s", file.filename)
#                 result = await process_kb_entry(
#                     "file",
#                     status=status,
#                     file=file,
#                     user_id=user_id,
#                     workspace_id=workspace_id,
#                     user_role=user_role,
#                 )
#             else:
#                 log.info("Processing %s with data: %s",
#                          fmt, str(data_field)[:100])
#                 result = await process_kb_entry(
#                     fmt,
#                     data=data_field,
#                     status=status,
#                     user_id=user_id,
#                     workspace_id=workspace_id,
#                     user_role=user_role,
#                 )

#             log.info("✓ KB training completed successfully")
#             return result

#         # Handle JSON body
#         raw = await request.body()
#         log.debug("Raw request body bytes: %r", raw)

#         if not raw:
#             log.error("Empty request body for JSON branch")
#             raise HTTPException(400, "Empty request body: expected JSON")

#         try:
#             body_text = raw.decode("utf-8")
#         except Exception as e:
#             log.error("Failed to decode request body: %s", e)
#             raise HTTPException(400, "Unable to decode request body as UTF-8")

#         try:
#             body = json.loads(body_text)
#         except Exception as e:
#             log.error("Invalid JSON body: %s", e)
#             # include a short hint for Windows cmd users who may need to escape pipes
#             raise HTTPException(
#                 400, "Invalid JSON body: ensure Content-Type: application/json and valid JSON (on Windows cmd, escape '|' as ^| or use a payload file)")

#         log.info("Processing JSON body")
#         fmt = (body.get("format") or "").lower().strip()

#         if fmt not in {"qna", "file", "link"}:
#             raise HTTPException(400, "Invalid or missing 'format' in JSON")

#         status = body.get("status")
#         data = body.get("data")
#         user_id = body.get("user_id") or user_id
#         workspace_id = body.get("workspace_id") or workspace_id

#         if not user_id:
#             raise HTTPException(400, "user_id is required")

#         if fmt == "file":
#             raise HTTPException(
#                 400,
#                 "format=file requires multipart/form-data with a 'file' field"
#             )

#         log.info("Processing %s with data: %s", fmt, str(data)[:100])
#         user_role = "admin"

#         result = await process_kb_entry(
#             fmt,
#             data=data,
#             status=status,
#             user_id=user_id,
#             workspace_id=workspace_id,
#             user_role=user_role,
#         )

#         log.info("✓ KB training completed successfully")
#         return result

#     except HTTPException:
#         raise
#     except Exception as e:
#         log.error("KB training failed: %s\n%s", e, traceback.format_exc())
#         raise HTTPException(500, f"Training error: {e}")


# @app.post("/kb/v2/train")
# async def kb_v2_train(
#     workspace_id: str = Form(...),
#     source_type: str = Form(...),  # file | link | qna
#     file: UploadFile | None = File(None),
#     link: str | None = Form(None),
#     question: str | None = Form(None),
#     answer: str | None = Form(None),
# ):
#     """
#     Knowledge Base ingestion v2
#     Supports:
#     - file (pdf, txt, docx)
#     - link
#     - qna
#     """
#     return await ingest_kb(
#         workspace_id=workspace_id,
#         source_type=source_type,
#         file=file,
#         link=link,
#         question=question,
#         answer=answer,
#     )


# @app.post("/kb/v2/query")
# async def kb_v2_query(
#     workspace_id: str = Form(...),
#     query: str = Form(...),
# ):
#     chunks = retrieve_kb_chunks(
#         workspace_id=workspace_id,
#         query=query,
#     )

#     if not chunks:
#         return {
#             "ok": True,
#             "answer": "This information is not available in the knowledge base.",
#             "sources": [],
#         }

#     answer = answer_from_kb(
#         query=query,
#         chunks=[c["text"] for c in chunks],
#     )

#     return {
#         "ok": True,
#         "answer": answer,
#         "sources": [c["public_id"] for c in chunks],
#     }


# # ═══════════════════════════════════════════════════════════════════
# # ROOT & HEALTH ENDPOINTS
# # ═══════════════════════════════════════════════════════════════════


# @app.get("/", tags=["System"])
# def root():
#     return {
#         "ok": True,
#         "service": "AI Calling API with RAG",
#         "version": "2.0.0",
#         "routes": {
#             "chat": "/chat",
#             "chat_stream": "/chat/stream",
#             "chat_history": "/chat/history",
#             "chat_reset": "/chat/reset",
#             "kb_train": "/kb/train",
#             "diagnostics": {
#                 "ping": "/chat/diag/ping",
#                 "env": "/chat/diag/env",
#                 "db_check": "/chat/diag/db-check",
#                 "index_check": "/chat/diag/index-check",
#                 "search_test": "/chat/diag/search-test",
#                 "full_flow": "/chat/diag/full-test",
#             },
#             "health": "/health",
#         },
#         "docs": "/docs",
#         "timestamp": datetime.utcnow().isoformat() + "Z",
#     }


# @app.get("/health", tags=["System"])
# def health():
#     openai_key = bool(os.getenv("OPENAI_API_KEY"))
#     mongodb_uri = bool(os.getenv("MONGODB_URI"))

#     all_ok = openai_key and mongodb_uri

#     status = {
#         "ok": all_ok,
#         "status": "healthy" if all_ok else "degraded",
#         "checks": {
#             "openai_key": "✓ configured" if openai_key else "✗ missing",
#             "mongodb_uri": "✓ configured" if mongodb_uri else "✗ missing",
#         },
#         "timestamp": datetime.utcnow().isoformat() + "Z",
#     }

#     if not all_ok:
#         log.warning("Health check failed: %s", status)

#     return status

# # ═══════════════════════════════════════════════════════════════════
# # STARTUP / SHUTDOWN EVENTS
# # ═══════════════════════════════════════════════════════════════════


# @app.on_event("startup")
# async def startup_event():
#     log.info("=" * 80)
#     log.info("✓ APPLICATION READY")
#     log.info("=" * 80)
#     log.info("Service: AI Calling API with RAG v2.0.0")
#     log.info("Environment:")
#     log.info("  - OPENAI_API_KEY: %s",
#              "✓ set" if os.getenv("OPENAI_API_KEY") else "✗ MISSING")
#     log.info("  - MONGODB_URI: %s",
#              "✓ set" if os.getenv("MONGODB_URI") else "✗ MISSING")
#     log.info("  - KB_DB_NAME: %s", os.getenv("KB_DB_NAME", "skylix_kb"))
#     log.info("  - KB_COL_NAME: %s",
#              os.getenv("KB_COL_NAME", "knowledge_entries"))
#     log.info("  - KB_INDEX_NAME: %s", os.getenv("KB_INDEX_NAME", "kb_index"))
#     log.info("  - LOG_LEVEL: %s", LOG_LEVEL)
#     log.info("Routes:")
#     log.info("  - Chat: /chat")
#     log.info("  - KB Training: /kb/train")
#     log.info("  - Diagnostics: /chat/diag/*")
#     log.info("  - API Docs: /docs")
#     log.info("=" * 80)
#     log.info("Ready to accept requests!")
#     log.info("=" * 80)


# @app.on_event("shutdown")
# async def shutdown_event():
#     log.info("=" * 80)
#     log.info("🛑 APPLICATION SHUTDOWN")
#     log.info("=" * 80)
from dotenv import load_dotenv
import os
import json
import asyncio
import certifi
from openai import OpenAI
import logging
import time
import traceback
from pathlib import Path
from datetime import datetime
from typing import Any
from pymongo import MongoClient
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from dummy_chat.dummy_kb_ingest import ingest_kb
from dummy_chat.dummy_kb_retrieve import retrieve_kb_chunks, get_kb_topics
from dummy_chat.dummy_kb_answer import answer_from_kb, calculate_token_cost

# ═══════════════════════════════════════════════════════════════════
# SETUP
# ═══════════════════════════════════════════════════════════════════
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path, override=True)

_openai_client = None
_mongo_client = None

_preferred_name_store: dict[str, str] = {}

# Conversation Working Memory support: persists extracted_data across turns so
# previously collected field values can be fed back into the next LLM call.
_extracted_data_store: dict[str, dict] = {}

# Hard kill switch for the debug.conversation_working_memory block in /chat
# responses. Set to False to disable debug output everywhere, regardless of
# what the client sends in the request payload.
DEBUG_WORKING_MEMORY_ENABLED = False


def _merge_extracted_data(previous: dict | None, new: dict | None) -> dict:
    """
    Additive merge of newly extracted_data into previously stored extracted_data.
    Only non-null/non-empty new values overwrite previous ones, so fields the
    model didn't mention this turn are preserved. Generic - no field names hardcoded.

    Exception: reserved "__"-prefixed meta keys (e.g. __current_focus_field__)
    always take the latest value, including null - they track transient
    conversation state (like "what was just asked"), not accumulated facts,
    so they must be clearable turn to turn.
    """
    merged = dict(previous or {})
    for key, value in (new or {}).items():
        if key.startswith("__"):
            merged[key] = value
            continue
        if value is not None and value != "":
            merged[key] = value
    return merged


def get_openai_client():
    """Lazy initialization of OpenAI client"""
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable not set")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def get_mongo_client():
    """Lazy initialization of MongoDB client"""
    global _mongo_client
    if _mongo_client is None:
        mongodb_uri = os.getenv("MONGODB_URI")
        if not mongodb_uri:
            raise RuntimeError("MONGODB_URI environment variable not set")
        _mongo_client = MongoClient(
            mongodb_uri,
            tls=True,
            tlsCAFile=certifi.where()
        )
    return _mongo_client


def get_kb_meta_collection():
    mongo = get_mongo_client()
    db = mongo[os.getenv("KB_DB_NAME", "skylix_rag")]
    return db["knowledge_bases"]


print("OPENAI_API_KEY loaded:", bool(os.getenv("OPENAI_API_KEY")))

logging.basicConfig(
    level=logging.DEBUG if os.getenv("RAG_DEBUG") else logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-15s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

log = logging.getLogger("app")
logging.getLogger("pymongo").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# ═══════════════════════════════════════════════════════════════════
# APP
# ═══════════════════════════════════════════════════════════════════
app = FastAPI(
    title="Simple RAG API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# REQUEST LOGGING MIDDLEWARE (Detailed + Safe Async Response)
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Logs all HTTP requests and responses with detailed trace output."""
    request_id = f"{int(time.time() * 1000)}"
    client_ip = request.client.host
    path = request.url.path
    method = request.method

    log.info(" REQUEST | id=%s | method=%s | path=%s | client=%s",
             request_id, method, path, client_ip)

    start_time = time.perf_counter()

    # Streaming endpoints (e.g. /chat_sse) must bypass this middleware's body
    # logging entirely - BaseHTTPMiddleware (which @app.middleware("http")
    # uses under the hood) always hands call_next()'s result back as its own
    # internal _StreamingResponse wrapper, regardless of what the endpoint
    # actually returned - so isinstance/media_type checks on the response
    # object never reliably identify it as streaming. Checking the request
    # path directly, before call_next() even runs, is the only way that
    # actually works here.
    if path.startswith("/chat_sse"):
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start_time) * 1000
        log.info(" RESPONSE | id=%s | status=%d | time=%.1fms (streaming - body not buffered/logged)",
                  request_id, response.status_code, duration_ms)
        return response

    try:
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start_time) * 1000

        # Capture and log the raw response body safely
        body_text = ""
        if hasattr(response, "body_iterator"):
            body_content = b""
            async for chunk in response.body_iterator:
                body_content += chunk
            body_text = body_content.decode("utf-8", errors="ignore").strip()

            # FIXED: restore async body iterator properly
            async def new_body_iterator():
                yield body_content
            response.body_iterator = new_body_iterator()
            response.headers["content-length"] = str(len(body_content))

        else:
            body_content = getattr(response, "body", b"")
            if isinstance(body_content, str):
                body_text = body_content
                body_content = body_content.encode("utf-8")
            else:
                body_text = body_content.decode(
                    "utf-8", errors="ignore").strip()
            response.headers["content-length"] = str(len(body_content))

        log.info(" RESPONSE | id=%s | status=%d | time=%.1fms",
                 request_id, response.status_code, duration_ms)
        #log.info("   Raw Response Body:")
        for line in body_text.splitlines():
            log.info("    %s", line)

        return response

    except Exception as e:
        duration_ms = (time.perf_counter() - start_time) * 1000
        log.error(" REQUEST FAILED | id=%s | time=%.1fms | error=%s",
                  request_id, duration_ms, str(e))
        raise


# GLOBAL EXCEPTION HANDLER
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all exception handler with detailed logging."""
    log.error("=" * 80)
    log.error("UNHANDLED EXCEPTION")
    log.error("=" * 80)
    log.error("Path: %s %s", request.method, request.url.path)
    log.error("Error: %s", str(exc))
    log.error("Traceback:\n%s", traceback.format_exc())
    log.error("=" * 80)

    return JSONResponse(
        status_code=500,
        content={
            "ok": False,
            "error": "Internal server error",
            "message": str(exc),
            "path": str(request.url.path),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
    )




# ═══════════════════════════════════════════════════════════════════
# SINGLE UPLOAD ENDPOINT (Like your original /kb/v2/train)
# ═══════════════════════════════════════════════════════════════════


@app.post("/kb/v2/train")
async def kb_train(
    request: Request,                                   # ADD THIS
    workspace_id: str = Form(...),
    knowledge_base_id: str | None = Form(None),
    folder_name: str | None = Form(None),
    user_id: str = Form(...),
    status: str = Form(...),
    source_type: str = Form(...),
    file: list[UploadFile] | None = File(None),
    link: list[str] | None = Form(None),
    paragraphs: list[str] | None = Form(None),
):
    """
    SINGLE ENDPOINT to upload ANY type of content

    Examples:

    1. FILE:
       curl -X POST http://localhost:8000/kb/v2/train \
            -F "workspace_id=ws_test" \
            -F "source_type=file" \
            -F "file=@document.pdf"

    2. LINK:
       curl -X POST http://localhost:8000/kb/v2/train \
            -F "workspace_id=ws_test" \
            -F "source_type=link" \
            -F "link=https://example.com/article"

    3. PARAGRAPH:
       curl -X POST http://localhost:8000/kb/v2/train \
            -F "workspace_id=ws_test" \
            -F "source_type=paragraph" \
            -F "paragraphs=RAG combines retrieval with generation..."
    """
    log.info("=" * 80)
    log.info(f"📥 KB TRAIN REQUEST")
    log.info(f"   Workspace: {workspace_id}")
    log.info(f"   Type: {source_type}")
    log.info("=" * 80)

    def _ensure_list(value):
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]

    try:
        if not knowledge_base_id or not knowledge_base_id.strip():
            raise HTTPException(400, "knowledge_base_id is required")
        
        # Extract Google access token from header (set by app backend after OAuth)
        # Header name is configurable via env so both backends can agree on it
        token_header_name = os.getenv("GOOGLE_TOKEN_HEADER", "X-Google-Access-Token")
        google_access_token: str | None = request.headers.get(token_header_name)

        

        items = []
        src_type = (source_type or "").strip().lower()

        if src_type not in {"file", "link", "paragraph", "mixed", "spreadsheet"}:
            raise HTTPException(
                400, "Invalid source_type. Must be 'file', 'link', 'paragraph', 'spreadsheet', or 'mixed'")

        files = _ensure_list(file)
        links = _ensure_list(link)
        paragraph_list = _ensure_list(paragraphs)

        if src_type in {"file", "mixed"}:
            for f in files:
                items.append(("file", {"file": f}))

        if src_type in {"link", "mixed"}:
            for l in links:
                if l:
                    items.append(("link", {"link": l}))

        if src_type == "spreadsheet":
            for l in links:
                if l:
                    items.append(("spreadsheet", {"link": l, "google_access_token": google_access_token}))

        if src_type in {"paragraph", "mixed"}:
            normalized_paragraphs = [p.strip() for p in paragraph_list if p and p.strip()]
            if src_type == "paragraph" and not normalized_paragraphs:
                raise HTTPException(400, "paragraphs required for source_type=paragraph")
            if normalized_paragraphs:
                items.append(("paragraph", {"paragraphs": normalized_paragraphs}))

        if not items:
            raise HTTPException(400, "No content provided for ingestion")

        results = []
        errors = []
        for item_type, payload in items:
            try:
                result = await ingest_kb(
                    workspace_id=workspace_id,
                    knowledge_base_id=knowledge_base_id.strip(),
                    folder_name=(folder_name.strip() if folder_name else None),
                    user_id=user_id,
                    status=status,
                    source_type=item_type,
                    file=payload.get("file"),
                    link=payload.get("link"),
                    paragraphs=payload.get("paragraphs"),
                    google_access_token=payload.get("google_access_token"),  # ADD THIS
                )
                results.append(result)
            except HTTPException as e:
                errors.append({
                    "ok": False,
                    "source_type": item_type,
                    "detail": e.detail,
                })
            except Exception as e:
                errors.append({
                    "ok": False,
                    "source_type": item_type,
                    "detail": str(e),
                })

        log.info("OK Upload processed!")
        if len(results) == 1 and not errors:
            return {
                "ok": True,
                "workspace_id": workspace_id,
                "total_items": 1,
                "item": results[0],
                "items": results,
            }
        return {
            "ok": len(errors) == 0,
            "workspace_id": workspace_id,
            "total_items": len(results) + len(errors),
            "succeeded": len(results),
            "failed": len(errors),
            "items": results,
            "errors": errors,
        }

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Upload failed: {e}", exc_info=True)
        raise HTTPException(500, str(e))


@app.post("/kb/v2/query")
async def kb_v2_query(
    workspace_id: str = Form(...),
    query: str = Form(...),
    knowledge_base_ids: str | None = Form(None),
):
    parsed_kb_ids = None
    if knowledge_base_ids:
        parsed = [kb_id.strip()
                  for kb_id in knowledge_base_ids.split(",") if kb_id.strip()]
        if parsed:
            parsed_kb_ids = parsed

    chunks = retrieve_kb_chunks(
        workspace_id=workspace_id,
        query=query,
        knowledge_base_ids=parsed_kb_ids,
    )

    if not chunks:
        return {
            "ok": True,
            "answer": "This information is not available in the knowledge base.",
            "sources": [],
        }

    answer = answer_from_kb(
        query=query,
        chunks=[c["text"] for c in chunks],
    )

    return {
        "ok": True,
        "answer": answer,
        "sources": [c["public_id"] for c in chunks],
    }


# INCLUDE ROUTERS
log.info("Loading routers...")

# Only local endpoints in this file should handle /chat and KB APIs.
# No imports from chat/ are used for routing.
log.info(" No external chat routers loaded")

# Optional: Voice router (if available)
# Optional: Voice router (if available)
try:
    from voice.voice_router import router as voice_router
    app.include_router(voice_router, tags=["Voice"])
    log.info("✓ Voice router loaded")
except ImportError as e:
    log.warning("⚠ Voice router not available (optional): %s", str(e))
except Exception as e:
    log.error("✗ Failed to load voice router: %s", e)

# Standalone, stateless POST /extract endpoint (separate from /chat).
from extraction.router import router as extraction_router
app.include_router(extraction_router)
log.info("✓ Extraction router loaded")


# ═══════════════════════════════════════════════════════════════════
# QUERY ENDPOINT
# ═══════════════════════════════════════════════════════════════════

class ChatQueryRequest(BaseModel):
    query: str
    conversation_id: str | None = None
    user_id: str | None = None
    workspace_ids: list[str] = []
    knowledge_base_ids: list[str] | None = None
    model: str | None = None
    temperature: float = 0
    max_tokens: int | None = None
    context: dict[str, Any] | str | None = None
    user_prompt: str | None = None
    status: str = "published"
    top_k: int = 5
    include_raw: bool = True
    fallback_text: bool = True
    extraction_fields: list[dict] | None = None
    debug: bool = False


@app.post("/chat")
async def kb_query(payload: ChatQueryRequest):
    """
    Query the knowledge base

    Example:
       curl -X POST http://localhost:8000/chat \
            -H "Content-Type: application/json" \
            -d "{\"workspace_ids\":[\"ws_test\"],\"query\":\"Who is Karthik Kailash?\"}"
    """
    if not payload.workspace_ids:
        raise HTTPException(
            400, "workspace_ids is required and cannot be empty")

    workspace_id = payload.workspace_ids[0]
    user_id = payload.user_id
    conversation_id = payload.conversation_id
    query = payload.query
    user_prompt = payload.user_prompt
    status = payload.status
    model = payload.model
    temperature = payload.temperature
    max_tokens = payload.max_tokens
    # Server-side floor, independent of what the caller sends. This
    # extraction schema's required output (every extraction field plus
    # its _refused/_needs_clarification companions, plus the natural
    # language "answer") can legitimately need more than a small caller-
    # supplied cap. Rather than trusting every client (widget, Postman,
    # future integrations) to pass an adequate value, guarantee a safe
    # minimum here so a too-low or missing value can never truncate the
    # JSON response and crash the request.
    MIN_EXTRACTION_MAX_TOKENS = 3000
    if max_tokens is None or max_tokens < MIN_EXTRACTION_MAX_TOKENS:
        max_tokens = MIN_EXTRACTION_MAX_TOKENS
    context = payload.context
    top_k = payload.top_k
    knowledge_base_ids = payload.knowledge_base_ids
    include_raw = payload.include_raw
    fallback_text = payload.fallback_text
    extraction_fields = payload.extraction_fields
    debug_enabled = bool(payload.debug) and DEBUG_WORKING_MEMORY_ENABLED
    log.info("=" * 80)
    log.info(f"🔍 KB QUERY REQUEST")
    log.info(f"   Workspace: {workspace_id}")
    log.info(f"   Query: {query}")
    log.info("=" * 80)

    try:
        status_norm = (status or 'published').strip().lower()
        if status_norm != 'published':
            return {
                'ok': True,
                'answer': 'This information is not available in the knowledge base.',
                'sources': [],
                'chunks_found': 0,
            }

        user_persona = None
        conversation_history = None
        contact_name = None
        preferred_name = _preferred_name_store.get(conversation_id) if conversation_id else None
        previous_extracted_data = _extracted_data_store.get(conversation_id) if conversation_id else None
        is_first_message = False
        if context:
            if isinstance(context, dict):
                user_persona = context.get('user_persona')
                conversation_history = context.get('conversation_history')
                contact_name = context.get('contact_name')
                preferred_name = preferred_name or context.get('preferred_name')
                is_first_message = bool(context.get('is_first_message', False))
            else:
                try:
                    import json
                    context_obj = json.loads(context)
                    if isinstance(context_obj, dict):
                        user_persona = context_obj.get('user_persona')
                        conversation_history = context_obj.get('conversation_history')
                        contact_name = context_obj.get('contact_name')
                        preferred_name = preferred_name or context.get('preferred_name')
                        is_first_message = bool(context_obj.get('is_first_message', False))
                except Exception:
                    user_persona = None

        min_score = 0.3
        # Retrieve chunks
        retrieval = retrieve_kb_chunks(
            workspace_id=workspace_id,
            query=query,
            knowledge_base_ids=knowledge_base_ids,
            top_k=top_k,
            min_score=min_score,
            include_raw=include_raw,
            fallback_text=fallback_text,
        )
        if include_raw:
            chunks, raw_results, diagnostics = retrieval
        else:
            chunks = retrieval
            raw_results = None
            diagnostics = None

        if not chunks:
            log.info("No relevant chunks found")
            kb_topics = get_kb_topics(workspace_id, limit=3, knowledge_base_ids=knowledge_base_ids)
            answer = answer_from_kb(
                query=query,
                chunks=[],
                user_prompt=user_prompt,
                user_persona=user_persona,
                conversation_history=conversation_history,
                kb_topics=kb_topics,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                contact_name=contact_name,
                preferred_name=preferred_name,
                is_first_message=is_first_message,
                extraction_fields=extraction_fields,
                previous_extracted_data=previous_extracted_data,
                debug=debug_enabled,
            )
            extracted_data = None
            debug_info = None

            if isinstance(answer, dict) and "answer" in answer and "extracted_data" in answer:
                extracted_data = answer.get("extracted_data")
                debug_info = answer.get("debug")
                answer = answer.get("answer")
            if conversation_id and extracted_data:
                _extracted_data_store[conversation_id] = _merge_extracted_data(
                    _extracted_data_store.get(conversation_id), extracted_data
                )
            # Check if name change was detected in this turn and persist it
            if conversation_id and isinstance(answer, dict) and answer.get("preferred_name"):
                _preferred_name_store[conversation_id] = answer.get("preferred_name")
                preferred_name = answer.get("preferred_name")
                answer = answer.get("answer") if "answer" in answer else answer
            total_cost = None
            response = {
                "ok": True,
                "answer": answer,
                "preferred_name": preferred_name,
                "sources": [],
                "chunks_found": 0,
                "total_cost": total_cost,
            }
            if extracted_data is not None:
                response["extracted_data"] = extracted_data
            if debug_info:
                response["debug"] = debug_info
            if include_raw:
                response["raw"] = {
                    "vector_results": raw_results,
                    "min_score": min_score,
                    "top_k": top_k,
                    "diagnostics": diagnostics,
                    "fallback_text": fallback_text,
                }
            return response

        # Generate answer
        log.info(f"Generating answer from {len(chunks)} chunks...")

        extracted_data = None   
        raw_openai = None
        debug_info = None
        if include_raw:
            answer, raw_openai = answer_from_kb(
                query=query,
                chunks=[c["text"] for c in chunks],
                extraction_fields=extraction_fields,  # ← pass dynamically
                include_raw=True,
                user_prompt=user_prompt,
                user_persona=user_persona,
                conversation_history=conversation_history,
                kb_topics=None,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                contact_name=contact_name,
                preferred_name=preferred_name,
                is_first_message=is_first_message,
                previous_extracted_data=previous_extracted_data,
                debug=debug_enabled,
            )
            extracted_data = None

            if isinstance(answer, dict) and "answer" in answer and "extracted_data" in answer:
                extracted_data = answer.get("extracted_data")
                debug_info = answer.get("debug")
                answer = answer.get("answer")
        else:
            answer = answer_from_kb(
                query=query,
                chunks=[c["text"] for c in chunks],
                extraction_fields=extraction_fields,  # ← pass dynamically
                user_prompt=user_prompt,
                user_persona=user_persona,
                conversation_history=conversation_history,
                kb_topics=None,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                contact_name=contact_name,
                preferred_name=preferred_name,
                is_first_message=is_first_message,
                previous_extracted_data=previous_extracted_data,
                debug=debug_enabled,
            )
            if isinstance(answer, dict) and "answer" in answer and "extracted_data" in answer:
                extracted_data = answer.get("extracted_data")
                debug_info = answer.get("debug")

        if conversation_id and extracted_data:
            _extracted_data_store[conversation_id] = _merge_extracted_data(
                _extracted_data_store.get(conversation_id), extracted_data
            )
        # Update preferred name store if AI detected a name change this turn


        if conversation_id and isinstance(answer, dict) and answer.get("preferred_name"):
            _preferred_name_store[conversation_id] = answer.get("preferred_name")
            preferred_name = answer.get("preferred_name")
            answer = answer.get("answer") if "answer" in answer else answer
        elif conversation_id and isinstance(answer, str):
            pass  # no name change detected this turn

        log.info("✅ Query successful!")

        cost_details = None
        total_cost = None
        if include_raw and raw_openai:
            cost_details = calculate_token_cost(
                raw_openai.get("usage"), raw_openai.get("model")
            )
            if cost_details:
                total_cost = cost_details.get("total_cost")

        response = {
            "ok": True,
            "answer": answer,
            "preferred_name": preferred_name,
            "sources": [
                {
                    "public_id": c["public_id"],
                    "source_name": c["source_name"],
                    "score": round(c["score"], 3),
                }
                for c in chunks
            ],
            "chunks_found": len(chunks),
            "total_cost": total_cost,
        }
        if extracted_data is not None:
            response["extracted_data"] = extracted_data
        if debug_info:
            response["debug"] = debug_info
        if include_raw:
            response["raw"] = {
                "vector_results": raw_results,
                "openai": raw_openai,
                "min_score": min_score,
                "top_k": top_k,
                "diagnostics": diagnostics,
                "fallback_text": fallback_text,
                "cost": cost_details,
            }
        return response

    except Exception as e:
        log.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(500, str(e))


@app.post("/chat_sse")
async def kb_query_sse(payload: ChatQueryRequest):
    """
    DEMO endpoint: same inputs as /chat, but streams the response as two
    Server-Sent Events over one connection instead of one JSON blob -
    "answer" as soon as the fast conversational reply is ready, then
    "extracted_data" a moment later once the separate extraction call
    finishes. Lets you see the answer/extraction split happening live.
    """

    async def event_stream():
        start_time = time.perf_counter()

        def sse(event: str, data: dict) -> str:
            data["elapsed_ms"] = round((time.perf_counter() - start_time) * 1000, 1)
            return f"event: {event}\ndata: {json.dumps(data)}\n\n"

        if not payload.workspace_ids:
            yield sse("error", {"detail": "workspace_ids is required and cannot be empty"})
            return

        log.info("SSE /chat_sse: INPUT QUERY -> %r", payload.query)

        workspace_id = payload.workspace_ids[0]
        conversation_id = payload.conversation_id
        query = payload.query
        user_prompt = payload.user_prompt
        model = payload.model
        temperature = payload.temperature
        extraction_fields = payload.extraction_fields

        MIN_EXTRACTION_MAX_TOKENS = 3000
        effective_max_tokens = payload.max_tokens
        if effective_max_tokens is None or effective_max_tokens < MIN_EXTRACTION_MAX_TOKENS:
            effective_max_tokens = MIN_EXTRACTION_MAX_TOKENS

        context = payload.context
        user_persona = None
        conversation_history = None
        contact_name = None
        preferred_name = _preferred_name_store.get(conversation_id) if conversation_id else None
        previous_extracted_data = _extracted_data_store.get(conversation_id) if conversation_id else None
        is_first_message = False
        if context:
            if isinstance(context, dict):
                user_persona = context.get('user_persona')
                conversation_history = context.get('conversation_history')
                contact_name = context.get('contact_name')
                preferred_name = preferred_name or context.get('preferred_name')
                is_first_message = bool(context.get('is_first_message', False))
            else:
                try:
                    context_obj = json.loads(context)
                    if isinstance(context_obj, dict):
                        user_persona = context_obj.get('user_persona')
                        conversation_history = context_obj.get('conversation_history')
                        contact_name = context_obj.get('contact_name')
                        preferred_name = preferred_name or context_obj.get('preferred_name')
                        is_first_message = bool(context_obj.get('is_first_message', False))
                except Exception:
                    user_persona = None

        min_score = 0.3
        chunks, raw_results, diagnostics = await asyncio.to_thread(
            retrieve_kb_chunks,
            workspace_id=workspace_id,
            query=query,
            knowledge_base_ids=payload.knowledge_base_ids,
            top_k=payload.top_k,
            min_score=min_score,
            include_raw=True,
            fallback_text=payload.fallback_text,
        )
        chunk_texts = [c["text"] for c in chunks] if chunks else []

        log.info("SSE /chat_sse: calling OpenAI #1 (fast reply, Working Memory aware, no extraction schema)")
        try:
            reply, raw_openai_answer = await asyncio.to_thread(
                answer_from_kb,
                query=query,
                chunks=chunk_texts,
                user_prompt=user_prompt,
                user_persona=user_persona,
                conversation_history=conversation_history,
                model=model,
                temperature=temperature,
                max_tokens=effective_max_tokens,
                contact_name=contact_name,
                preferred_name=preferred_name,
                is_first_message=is_first_message,
                extraction_fields=extraction_fields,
                previous_extracted_data=previous_extracted_data,
                include_raw=True,
                force_extraction_mode=False,
            )
        except Exception as e:
            log.error("SSE /chat_sse: call #1 (answer) failed: %s", e, exc_info=True)
            yield sse("error", {"stage": "answer", "detail": str(e)})
            return

        if isinstance(reply, dict):
            answer_text = reply.get("answer", "")
            if reply.get("preferred_name") and conversation_id:
                _preferred_name_store[conversation_id] = reply.get("preferred_name")
                preferred_name = reply.get("preferred_name")
        else:
            answer_text = reply

        log.info("SSE /chat_sse: OpenAI #1 done (answer) - elapsed=%.1fms",
                  (time.perf_counter() - start_time) * 1000)
        log.info("SSE /chat_sse: GENERATED ANSWER -> %r", answer_text)

        yield sse("answer", {
            "ok": True,
            "answer": answer_text,
            "preferred_name": preferred_name,
            "sources": [
                {
                    "public_id": c["public_id"],
                    "source_name": c["source_name"],
                    "score": round(c["score"], 3),
                }
                for c in chunks
            ] if chunks else [],
            "chunks_found": len(chunks) if chunks else 0,
        })

        # ─────────────────────────────────────────────────────────────────
        # DISABLED (not deleted): per-turn extraction call #2 + its
        # "extracted_data" event. Extraction now has its own dedicated,
        # stateless, debounced endpoint - POST /extract (see extraction/
        # folder + extraction/README.md) - which reconciles the FULL
        # conversation transcript on the calling backend's own schedule
        # instead of running on every single /chat_sse turn. Keeping this
        # running here too would mean paying for and running extraction
        # TWICE per message (once here, once via /extract) for no benefit.
        # Left commented out, not removed, in case we ever want to revert
        # to inline per-turn extraction instead of the separate endpoint.
        # ─────────────────────────────────────────────────────────────────
        # extracted_data = None
        # raw_openai_extraction = None
        # if extraction_fields:
        #     log.info("SSE /chat_sse: calling OpenAI #2 (extraction only)")
        #     try:
        #         extraction_result, raw_openai_extraction = await asyncio.to_thread(
        #             answer_from_kb,
        #             query=query,
        #             chunks=chunk_texts,
        #             user_prompt=user_prompt,
        #             user_persona=user_persona,
        #             conversation_history=conversation_history,
        #             model=model,
        #             temperature=temperature,
        #             max_tokens=effective_max_tokens,
        #             contact_name=contact_name,
        #             preferred_name=preferred_name,
        #             is_first_message=is_first_message,
        #             extraction_fields=extraction_fields,
        #             previous_extracted_data=previous_extracted_data,
        #             include_raw=True,
        #         )
        #     except Exception as e:
        #         log.error("SSE /chat_sse: call #2 (extraction) failed: %s", e, exc_info=True)
        #         yield sse("error", {"stage": "extracted_data", "detail": str(e)})
        #         return
        #
        #     if isinstance(extraction_result, dict):
        #         extracted_data = extraction_result.get("extracted_data")
        #
        #     if conversation_id and extracted_data:
        #         _extracted_data_store[conversation_id] = _merge_extracted_data(
        #             _extracted_data_store.get(conversation_id), extracted_data
        #         )
        #
        #     log.info("SSE /chat_sse: OpenAI #2 done (extraction) - elapsed=%.1fms",
        #               (time.perf_counter() - start_time) * 1000)
        #
        # # Combined cost of BOTH OpenAI calls (fast reply + extraction) - only
        # # knowable now, since call #2 has just finished.
        # cost_answer = None
        # cost_extraction = None
        # if raw_openai_answer:
        #     cost_answer = calculate_token_cost(
        #         raw_openai_answer.get("usage"), raw_openai_answer.get("model")
        #     )
        # if raw_openai_extraction:
        #     cost_extraction = calculate_token_cost(
        #         raw_openai_extraction.get("usage"), raw_openai_extraction.get("model")
        #     )
        #
        # total_cost = None
        # if cost_answer or cost_extraction:
        #     total_cost = (cost_answer.get("total_cost") if cost_answer else 0) + \
        #         (cost_extraction.get("total_cost") if cost_extraction else 0)
        #
        # yield sse("extracted_data", {
        #     "ok": True,
        #     "extracted_data": extracted_data,
        #     "total_cost": total_cost,
        #     "cost": {
        #         "answer_call": cost_answer,
        #         "extraction_call": cost_extraction,
        #     },
        #     "raw": {
        #         "vector_results": raw_results,
        #         "min_score": min_score,
        #         "top_k": payload.top_k,
        #         "diagnostics": diagnostics,
        #         "fallback_text": payload.fallback_text,
        #     },
        # })

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ═══════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════

@app.get("/kb/folders")
async def list_kb_folders(workspace_id: str):
    collection = get_kb_meta_collection()
    docs = list(collection.find(
        {
            "workspace_id": workspace_id,
            "is_deleted": {"$ne": True},
        },
        {
            "_id": 0,
            "knowledge_base_id": 1,
            "folder_name": 1,
            "paragraph": 1,
            "file": 1,
            "link": 1,
            "spreadsheet": 1,
            "created_at": 1,
        },
    ).sort("created_at", 1))
    return docs


@app.patch("/kb/folder/rename")
async def rename_kb_folder(
    workspace_id: str = Form(...),
    knowledge_base_id: str = Form(...),
    new_folder_name: str = Form(...),
):
    collection = get_kb_meta_collection()
    name = (new_folder_name or "").strip()
    if not name:
        raise HTTPException(400, "new_folder_name is required")

    duplicate = collection.find_one({
        "workspace_id": workspace_id,
        "folder_name": name,
        "is_deleted": {"$ne": True},
        "knowledge_base_id": {"$ne": knowledge_base_id},
    })
    if duplicate:
        raise HTTPException(400, "Folder name already exists in this workspace")

    result = collection.update_one(
        {
            "workspace_id": workspace_id,
            "knowledge_base_id": knowledge_base_id,
            "is_deleted": {"$ne": True},
        },
        {
            "$set": {
                "folder_name": name,
                "updated_at": datetime.utcnow(),
            }
        },
    )
    if result.matched_count == 0:
        raise HTTPException(404, "Folder not found")
    return {"ok": True}


@app.delete("/kb/folder")
async def delete_kb_folder(
    workspace_id: str = Form(...),
    knowledge_base_id: str = Form(...),
):
    collection = get_kb_meta_collection()
    result = collection.update_one(
        {
            "workspace_id": workspace_id,
            "knowledge_base_id": knowledge_base_id,
            "is_deleted": {"$ne": True},
        },
        {
            "$set": {
                "is_deleted": True,
                "updated_at": datetime.utcnow(),
            }
        },
    )
    if result.matched_count == 0:
        raise HTTPException(404, "Folder not found")
    return {"ok": True}


@app.get("/kb/stats/{workspace_id}")
async def get_stats(workspace_id: str):
    """Get workspace statistics"""
    from dummy_chat.dummy_kb_retrieve import collection

    try:
        total = collection.count_documents({
            "workspace_id": workspace_id,
            "status": "published"
        })

        # Count by source type
        pipeline = [
            {"$match": {"workspace_id": workspace_id, "status": "published"}},
            {"$group": {"_id": "$source_type", "count": {"$sum": 1}}}
        ]
        by_type = {doc["_id"]: doc["count"]
                   for doc in collection.aggregate(pipeline)}

        # Count unique sources
        pipeline = [
            {"$match": {"workspace_id": workspace_id, "status": "published"}},
            {"$group": {"_id": "$source_name"}},
            {"$count": "total"}
        ]
        result = list(collection.aggregate(pipeline))
        unique_sources = result[0]["total"] if result else 0

        return {
            "ok": True,
            "workspace_id": workspace_id,
            "total_chunks": total,
            "unique_sources": unique_sources,
            "by_type": by_type,
        }
    except Exception as e:
        log.error(f"Stats failed: {e}")
        raise HTTPException(500, str(e))


@app.get("/")
def root():
    return {
        "service": "Simple RAG API",
        "version": "1.0.0",
        "endpoints": {
            "upload": "POST /kb/v2/train",
            "query": "POST /chat",
            "stats": "GET /kb/stats/{workspace_id}",
        },
        "database": {
            "name": os.getenv("KB_DB_NAME", "skylix_rag"),
            "collection": os.getenv("KB_COLLECTION_NAME", "kb_chunks"),
            "index": os.getenv("KB_VECTOR_INDEX", "kb_chunks_vector"),
        },
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "openai": "✓" if os.getenv("OPENAI_API_KEY") else "✗",
        "mongodb": "✓" if os.getenv("MONGODB_URI") else "✗",
    }


@app.on_event("startup")
async def startup():
    log.info("=" * 80)
    log.info("🚀 SIMPLE RAG API READY")
    log.info("=" * 80)
    log.info(f"Database: {os.getenv('KB_DB_NAME', 'skylix_rag')}")
    log.info(f"Collection: {os.getenv('KB_COLLECTION_NAME', 'kb_chunks')}")
    log.info(f"Index: {os.getenv('KB_VECTOR_INDEX', 'kb_chunks_vector')}")
    log.info("=" * 80)
    log.info("Endpoints:")
    log.info("  POST /kb/v2/train  - Upload (file/link/paragraph)")
    log.info("  POST /chat        - Query knowledge base")
    log.info("  GET  /kb/stats/{workspace_id} - Get stats")
    log.info("=" * 80)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)