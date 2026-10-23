import os
import certifi
from datetime import datetime
import re
from pymongo import MongoClient
from fastapi import UploadFile, HTTPException
from openai import OpenAI
import logging

from dummy_chat.dummy_utils import clean_text, chunk_text, generate_public_id
from dummy_chat.dummy_extractors import (
    extract_pdf,
    extract_docx,
    extract_txt,
    extract_link,
    LinkExtractionError,
    probe_sheet_access,
    extract_google_sheet_open,
    extract_google_sheet_locked,
)
from google.oauth2.credentials import Credentials as _GoogleCredentials

logger = logging.getLogger(__name__)

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536

# Database configuration
DB_NAME = os.getenv("KB_DB_NAME", "rag_db")
COLLECTION_NAME = os.getenv("KB_COLLECTION_NAME", "kb_chunks")
VECTOR_INDEX_NAME = os.getenv("KB_VECTOR_INDEX", "kb_chunks_vector")

_mongo_client = None
_openai_client = None


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
        logger.info(f"MongoDB connected: {DB_NAME}.{COLLECTION_NAME}")
    return _mongo_client


def get_openai_client():
    """Lazy initialization of OpenAI client"""
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable not set")

        _openai_client = OpenAI(api_key=api_key)
        logger.info("OpenAI client initialized")
    return _openai_client

def _build_credentials_from_token(access_token: str) -> _GoogleCredentials:
    """
    Build a minimal Google Credentials object from a raw access token.
    The app backend already did OAuth; we just wrap the token so the
    Google Sheets API client can use it.
    No client_id/secret needed for a single API call with a valid token.
    """
    return _GoogleCredentials(token=access_token)

def get_collection():
    """Get MongoDB collection"""
    mongo = get_mongo_client()
    db = mongo[DB_NAME]
    return db[COLLECTION_NAME]


def get_kb_registry_collection():
    """Get MongoDB knowledge base registry collection"""
    mongo = get_mongo_client()
    db = mongo[DB_NAME]
    return db["knowledge_bases"]


def ensure_kb_registry(
    workspace_id: str,
    knowledge_base_id: str,
    folder_name: str,
    user_id: str,
):
    """Ensure knowledge_bases registry document exists without overwriting existing values."""
    collection = get_kb_registry_collection()
    now = datetime.utcnow()
    collection.update_one(
        {
            "workspace_id": workspace_id,
            "knowledge_base_id": knowledge_base_id,
        },
        {
            "$setOnInsert": {
                "workspace_id": workspace_id,
                "knowledge_base_id": knowledge_base_id,
                "folder_name": folder_name,
                "paragraph": {},
                "file": {},
                "link": {},
                "spreadsheet": {},
                "created_by": user_id,
                "created_at": now,
                "updated_at": now,
                "is_deleted": False,
            }
        },
        upsert=True,
    )


def _next_paragraph_index(paragraph_map: dict) -> int:
    max_idx = 0
    for key in paragraph_map.keys():
        match = re.fullmatch(r"paragraph_(\d+)", str(key))
        if match:
            max_idx = max(max_idx, int(match.group(1)))
    return max_idx + 1


def _folder_source_type(paragraph_map: dict, file_map: dict, link_map: dict, spreadsheet_map: dict | None = None) -> str:
    spreadsheet_map = spreadsheet_map or {}
    populated = int(bool(paragraph_map)) + int(bool(file_map)) + int(bool(link_map)) + int(bool(spreadsheet_map))
    if populated > 1:
        return "mixed"
    if paragraph_map:
        return "paragraph"
    if file_map:
        return "file"
    if spreadsheet_map:
        return "spreadsheet"
    if link_map:
        return "link"
    return "paragraph"

def update_kb_registry_content(
    workspace_id: str,
    knowledge_base_id: str,
    source_type: str,
    *,
    paragraph_text: str | None = None,
    filename: str | None = None,
    url: str | None = None,
) -> str:
    collection = get_kb_registry_collection()
    query = {
        "workspace_id": workspace_id,
        "knowledge_base_id": knowledge_base_id,
    }
    existing = collection.find_one(
        query, {"_id": 0, "paragraph": 1, "file": 1, "link": 1, "spreadsheet": 1}
    ) or {}
    paragraph_map = dict(existing.get("paragraph") or {})
    file_map = dict(existing.get("file") or {})
    link_map = dict(existing.get("link") or {})
    spreadsheet_map = dict(existing.get("spreadsheet") or {})

    if source_type == "paragraph":
        text = (paragraph_text or "").strip()
        if text:
            next_idx = _next_paragraph_index(paragraph_map)
            paragraph_map[f"paragraph_{next_idx}"] = text
    elif source_type == "file":
        key = (filename or "").strip()
        if key:
            file_map[key] = True
    elif source_type == "link":
        key = (url or "").strip()
        if key:
            link_map[key] = True
    elif source_type == "spreadsheet":
        key = (url or "").strip()
        if key:
            spreadsheet_map[key] = True

    folder_type = _folder_source_type(paragraph_map, file_map, link_map, spreadsheet_map)
    now = datetime.utcnow()
    collection.update_one(
        query,
        {
            "$set": {
                "paragraph": paragraph_map,
                "file": file_map,
                "link": link_map,
                "spreadsheet": spreadsheet_map,
                "folder_source_type": folder_type,
                "updated_at": now,
            },
        }
    )
    return folder_type


def embed_chunks(chunks):
    """Generate embeddings for chunks"""
    logger.info(f"Generating embeddings for {len(chunks)} chunks...")

    openai_client = get_openai_client()

    res = openai_client.embeddings.create(
        model=EMBED_MODEL,
        input=chunks,
        encoding_format="float",
    )
    vectors = [d.embedding for d in res.data]

    for v in vectors:
        if len(v) != EMBED_DIM:
            raise RuntimeError("Embedding dimension mismatch")

    logger.info(f"Generated {len(vectors)} embeddings")
    return vectors


async def ingest_kb(
    workspace_id: str,
    knowledge_base_id: str,
    folder_name: str | None,
    user_id: str,
    status: str,
    source_type: str,
    file: UploadFile | None = None,
    link: str | None = None,
    paragraphs: list[str] | None = None,
    google_access_token: str | None = None,
):
    """
    Single unified ingestion function
    Handles: file, link, paragraph
    """
    if not workspace_id:
        raise HTTPException(400, "workspace_id required")
    if not knowledge_base_id:
        raise HTTPException(400, "knowledge_base_id required")
    if not folder_name or not folder_name.strip():
        folder_name = knowledge_base_id
    if not user_id:
        raise HTTPException(400, "user_id required")

    status_norm = (status or "published").strip().lower()
    if status_norm not in {"draft", "published"}:
        raise HTTPException(400, "Invalid status. Must be 'draft' or 'published'")

    src_type = (source_type or "").strip().lower()
    if src_type not in {"file", "link", "paragraph","spreadsheet"}:
        raise HTTPException(400, "Invalid source_type. Must be 'file', 'link', 'paragraph', or 'spreadsheet'")

    collection = get_collection()
    try:
        ensure_kb_registry(
            workspace_id=workspace_id,
            knowledge_base_id=knowledge_base_id,
            folder_name=folder_name,
            user_id=user_id,
        )
    except Exception as e:
        logger.error(f"Knowledge base registry upsert failed: {e}", exc_info=True)
        raise HTTPException(500, f"Failed to ensure knowledge base registry: {str(e)}")

    total_chunks_inserted = 0
    total_characters_extracted = 0
    items_processed = 0
    last_source_name = ""
    first_raw_preview = ""
    first_chunk_preview = ""
    folder_source_type = "paragraph"

    def _process_text_item(raw_text: str, item_source_type: str, item_source_name: str):
        nonlocal total_chunks_inserted
        nonlocal total_characters_extracted
        nonlocal items_processed
        nonlocal last_source_name
        nonlocal first_raw_preview
        nonlocal first_chunk_preview
        nonlocal folder_source_type

        cleaned = clean_text(raw_text)
        if len(cleaned) < 100:
            raise HTTPException(
                400,
                f"Insufficient text ({len(cleaned)} chars). Minimum: 100 characters. "
                f"The source might be empty, behind paywall, or contain mostly images."
            )

        try:
            folder_source_type = update_kb_registry_content(
                workspace_id=workspace_id,
                knowledge_base_id=knowledge_base_id,
                source_type=item_source_type,
                paragraph_text=cleaned if item_source_type == "paragraph" else None,
                filename=item_source_name if item_source_type == "file" else None,
                url=item_source_name if item_source_type in {"link", "spreadsheet"} else None,
            )
        except Exception as e:
            logger.error(f"Registry metadata update error: {e}", exc_info=True)
            raise HTTPException(500, f"Failed to update knowledge base metadata: {str(e)}")

        try:
            chunks = chunk_text(cleaned)
        except Exception as e:
            logger.error(f"Chunking error: {e}", exc_info=True)
            raise HTTPException(500, f"Failed to chunk text: {str(e)}")

        try:
            embeddings = embed_chunks(chunks)
        except Exception as e:
            logger.error(f"Embedding error: {e}", exc_info=True)
            raise HTTPException(500, f"Failed to generate embeddings: {str(e)}")

        docs = []
        total = len(chunks)
        for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            docs.append({
                "public_id": generate_public_id(),
                "workspace_id": workspace_id,
                "knowledge_base_id": knowledge_base_id,
                "folder_name": folder_name,
                "folder_source_type": folder_source_type,
                "user_id": user_id,
                "source_type": item_source_type,
                "source_name": item_source_name,
                "text": chunk,
                "embedding": emb,
                "chunk_index": idx,
                "total_chunks": total,
                "status": status_norm,
                "created_at": datetime.utcnow()
            })

        try:
            collection.insert_many(docs)
        except Exception as e:
            logger.error(f"Database insertion error: {e}", exc_info=True)
            raise HTTPException(500, f"Failed to store in MongoDB: {str(e)}")

        total_chunks_inserted += total
        total_characters_extracted += len(cleaned)
        items_processed += 1
        last_source_name = item_source_name
        if not first_raw_preview:
            first_raw_preview = cleaned[:500]
        if not first_chunk_preview and chunks:
            first_chunk_preview = chunks[0][:200]

    try:
        if src_type == "file":
            if not file:
                raise HTTPException(400, "file required")

            filename = file.filename
            if not filename:
                raise HTTPException(400, "invalid file name")

            try:
                file.file.seek(0)
            except Exception:
                pass

            name = filename.lower()
            if name.endswith(".pdf"):
                raw = extract_pdf(file)
            elif name.endswith(".docx"):
                raw = extract_docx(file)
            elif name.endswith(".txt"):
                raw = extract_txt(file)
            else:
                raise HTTPException(400, "Unsupported file type. Supported: PDF, DOCX, TXT")

            _process_text_item(raw, "file", filename)

        elif src_type == "link":
            if not link:
                raise HTTPException(400, "link required")

            try:
                raw = extract_link(link)
            except LinkExtractionError:
                raise HTTPException(400, "Extracted content is too short. Source is empty")

            _process_text_item(raw, "link", link)

        elif src_type == "spreadsheet":
            if not link:
                raise HTTPException(400, "link (Google Sheet URL) required for source_type=spreadsheet")

            sheet_url = link.strip()
            access_type = probe_sheet_access(sheet_url)
            logger.info(f"Sheet access type detected: {access_type}")

            if access_type == "open":
                try:
                    raw = extract_google_sheet_open(sheet_url)
                except Exception as e:
                    raise HTTPException(400, f"Failed to extract open Google Sheet: {str(e)}")

            else:  # locked
                # Authentication is handled by the app backend.
                # It passes the Google access token via a request header.
                # Our job: check it exists, build credentials, extract sheet.
                if not google_access_token or not google_access_token.strip():
                    raise HTTPException(
                        401,
                        "This Google Sheet is private and requires authentication. "
                        "Please authenticate via the app and retry."
                    )
                creds = _build_credentials_from_token(google_access_token.strip())
                try:
                    raw = extract_google_sheet_locked(sheet_url, creds)
                except PermissionError as e:
                    raise HTTPException(403, str(e))
                except Exception as e:
                    raise HTTPException(400, f"Failed to extract locked Google Sheet: {str(e)}")

            _process_text_item(raw, "spreadsheet", sheet_url)

        elif src_type == "paragraph":
            normalized = [p.strip() for p in (paragraphs or []) if p and p.strip()]
            if not normalized:
                raise HTTPException(400, "paragraphs required for source_type=paragraph")

            for idx, text in enumerate(normalized, start=1):
                _process_text_item(text, "paragraph", f"paragraph_{idx}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        raise HTTPException(500, f"Ingestion failed: {str(e)}")

    verify_count = collection.count_documents({"workspace_id": workspace_id})

    return {
        "ok": True,
        "workspace_id": workspace_id,
        "knowledge_base_id": knowledge_base_id,
        "folder_name": folder_name,
        "folder_source_type": folder_source_type,
        "user_id": user_id,
        "status": status_norm,
        "source_type": src_type,
        "source_name": last_source_name,
        "items_processed": items_processed,
        "chunks_inserted": total_chunks_inserted,
        "characters_extracted": total_characters_extracted,
        "raw_text_preview": first_raw_preview,
        "chunk_preview": first_chunk_preview,
        "database": DB_NAME,
        "collection": COLLECTION_NAME,
        "total_in_workspace": verify_count,
    }
