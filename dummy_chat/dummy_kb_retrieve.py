import os
import certifi
import base64
import math
from array import array
from typing import List
import logging
from pymongo import MongoClient
from openai import OpenAI

logger = logging.getLogger(__name__)

# Configuration
DB_NAME = os.getenv("KB_DB_NAME", "skylix_rag")
COLLECTION_NAME = os.getenv("KB_COLLECTION_NAME", "kb_chunks")
VECTOR_INDEX_NAME = os.getenv("KB_VECTOR_INDEX", "kb_chunks_vector")
KB_META_COLLECTION_NAME = os.getenv("KB_META_COLLECTION_NAME", "knowledge_bases")

# ═══════════════════════════════════════════════════════════════════
# LAZY INITIALIZATION - Fixes the loading issue
# ═══════════════════════════════════════════════════════════════════

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
        logger.info(f"✓ Retrieval connected to: {DB_NAME}.{COLLECTION_NAME}")
    return _mongo_client


def get_openai_client():
    """Lazy initialization of OpenAI client"""
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable not set")

        _openai_client = OpenAI(api_key=api_key)
        logger.info("✓ OpenAI client initialized for retrieval")
    return _openai_client


def get_collection():
    """Get MongoDB collection"""
    mongo = get_mongo_client()
    db = mongo[DB_NAME]
    return db[COLLECTION_NAME]


def _resolve_active_kb_ids(
    workspace_id: str,
    requested_ids: list[str] | None,
) -> list[str] | None:
    """
    Return KB ids allowed for retrieval:
    - Excludes soft-deleted folders from knowledge_bases when metadata exists.
    - Keeps existing behavior if metadata collection has no records for workspace.
    """
    mongo = get_mongo_client()
    db = mongo[DB_NAME]
    kb_meta = db[KB_META_COLLECTION_NAME]
    try:
        base_filter = {"workspace_id": workspace_id}
        workspace_total = kb_meta.count_documents(base_filter)
        if workspace_total == 0:
            return requested_ids

        active_ids = kb_meta.distinct(
            "knowledge_base_id",
            {
                "workspace_id": workspace_id,
                "is_deleted": {"$ne": True},
            },
        )
        active_ids = [x for x in active_ids if isinstance(x, str) and x.strip()]
    except Exception as e:
        logger.warning("KB metadata lookup failed; skipping soft-delete filter: %s", str(e))
        return requested_ids

    if requested_ids is None:
        return active_ids

    requested = [x for x in requested_ids if isinstance(x, str) and x.strip()]
    active_set = set(active_ids)
    return [x for x in requested if x in active_set]


def get_kb_topics(
    workspace_id: str,
    limit: int = 3,
    knowledge_base_ids: list[str] | None = None,
) -> List[str]:
    """Return up to N topic names from published KB sources within the selected KBs."""
    collection = get_collection()
    topic_filter: dict = {"workspace_id": workspace_id, "status": "published"}
    if knowledge_base_ids is not None:
        topic_filter["knowledge_base_id"] = {"$in": knowledge_base_ids}
    names = collection.distinct("source_name", topic_filter)
    topics = [n for n in names if isinstance(n, str) and n.strip()]
    return topics[:max(int(limit), 0)]


def _decode_embedding(embedding) -> List[float] | None:
    if isinstance(embedding, list):
        if embedding and isinstance(embedding[0], (int, float)):
            return embedding
        return None
    if isinstance(embedding, (str, bytes, bytearray)):
        try:
            data = base64.b64decode(embedding)
            arr = array('f')
            arr.frombytes(data)
            return list(arr)
        except Exception:
            return None
    return None


def _local_vector_search(
    collection,
    workspace_id: str,
    query_embedding: List[float],
    top_k: int,
    min_score: float,
    knowledge_base_ids: list[str] | None = None,
    max_docs: int = 5000,
):
    local_match = {
        'workspace_id': workspace_id,
        'status': 'published',
    }
    if knowledge_base_ids is not None:
        local_match["knowledge_base_id"] = {"$in": knowledge_base_ids}

    docs = list(collection.find(
        local_match,
        {
            '_id': 0,
            'public_id': 1,
            'text': 1,
            'source_name': 1,
            'source_type': 1,
            'chunk_index': 1,
            'embedding': 1,
        },
    ).limit(int(max_docs)))

    q_norm = math.sqrt(sum(v * v for v in query_embedding)) or 1.0
    scored = []
    for doc in docs:
        emb = _decode_embedding(doc.get('embedding'))
        if not emb or len(emb) != len(query_embedding):
            continue
        dot = 0.0
        for a, b in zip(query_embedding, emb):
            dot += a * b
        d_norm = math.sqrt(sum(v * v for v in emb)) or 1.0
        score = dot / (q_norm * d_norm)
        doc.pop('embedding', None)
        doc['score'] = score
        scored.append(doc)

    scored.sort(key=lambda d: d['score'], reverse=True)
    filtered = [d for d in scored if d['score'] >= min_score]
    # Do NOT fall back to low-score results here.
    # If nothing meets min_score, return empty so the caller's out-of-context
    # logic fires correctly instead of returning irrelevant chunks.
    return filtered[:int(top_k)], len(docs)


def embed_query(text: str) -> List[float]:
    """Generate embedding for a query"""
    logger.info(f"Generating query embedding for: {text[:50]}...")

    openai = get_openai_client()

    resp = openai.embeddings.create(
        model="text-embedding-3-small",
        input=text,
        encoding_format="float",
    )

    embedding = resp.data[0].embedding
    logger.info(f"✓ Generated embedding ({len(embedding)} dimensions)")

    return embedding


def retrieve_kb_chunks(
    workspace_id: str,
    query: str,
    knowledge_base_ids: list[str] | None = None,
    top_k: int = 5,
    min_score: float = 0.3,
    include_raw: bool = False,
    fallback_text: bool = False,
):
    """
    Retrieve relevant chunks using vector search

    Args:
        workspace_id: The workspace to search in
        query: Search query text
        top_k: Number of results (default: 5)
        min_score: Minimum similarity score (default: 0.75)

    Returns:
        List of matching chunks with scores
    """
    logger.info("=" * 80)
    logger.info(f"🔍 VECTOR SEARCH")
    logger.info(f"   Workspace: {workspace_id}")
    logger.info(f"   Query: {query}")
    logger.info(f"   Top-K: {top_k}, Min Score: {min_score}")
    logger.info("=" * 80)

    # Get collection (lazy initialization)
    collection = get_collection()
    effective_kb_ids = _resolve_active_kb_ids(workspace_id, knowledge_base_ids)

    # If caller requested specific KB IDs but none are valid/active, return empty immediately.
    # This prevents fallback logic from widening the search to unselected knowledge bases.
    if isinstance(effective_kb_ids, list) and len(effective_kb_ids) == 0:
        logger.warning(
            "All requested knowledge_base_ids were filtered out (deleted or invalid). "
            "Returning empty results to enforce KB scope."
        )
        if include_raw:
            return [], [], {
                "workspace_id": workspace_id,
                "knowledge_base_ids": effective_kb_ids,
                "reason": "no_active_kb_ids",
            }
        return []

    # Generate query embedding

    # Generate query embedding
    query_embedding = embed_query(query)

    # MongoDB vector search pipeline
    # NOTE: filter fields must be in the vector index. To avoid empty results when
    # workspace_id/status are not indexed as filterable, apply match after search.
    match_filter = {
        "workspace_id": workspace_id,
        "status": "published",
    }
    if effective_kb_ids is not None:
        match_filter["knowledge_base_id"] = {"$in": effective_kb_ids}

    vector_limit = max(int(top_k) * 5, 50)
    pipeline = [
        {
            "$vectorSearch": {
                "index": "kb_chunks_vector",
                "queryVector": query_embedding,
                "path": "embedding",
                "numCandidates": 100,
                "limit": int(vector_limit),
            }
        },
        {
            "$match": match_filter
        },
        {
            "$limit": int(top_k),
        },
        {
            "$project": {
                "_id": 0,
                "public_id": 1,
                "text": 1,
                "source_name": 1,
                "source_type": 1,
                "chunk_index": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]

    logger.info(f"Executing vector search on index: {VECTOR_INDEX_NAME}")

    try:
        results = list(collection.aggregate(pipeline))
        logger.info(f"✓ Vector search returned {len(results)} results")
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        logger.error(
            f"Make sure index '{VECTOR_INDEX_NAME}' exists in MongoDB Atlas!")
        raise

    # Apply score filter
    filtered = [r for r in results if r["score"] >= min_score]
    logger.info(
        f"OK {len(filtered)} results above score threshold {min_score}")
    used_low_score = False
    used_fallback = False
    used_local_vector = False
    local_vector_docs = 0

    if not filtered and results and knowledge_base_ids is None:
        # Only bypass min_score when no specific KB is requested (broad search).
        # When KB scope is explicitly set, respect min_score strictly so the
        # out-of-context handler fires instead of returning low-confidence chunks.
        logger.info("No results above min_score; using top vector results (unscoped search)")
        filtered = results
        used_low_score = True

    if filtered:
        logger.info("Top results:")
        for i, r in enumerate(filtered[:3]):
            logger.info(
                f"  {i+1}. {r['source_name']} (score: {r['score']:.3f})")
            logger.info(f"     {r['text'][:100]}...")

    # Optional local vector fallback if vector search returns nothing
    if not filtered and os.getenv('KB_LOCAL_VECTOR_FALLBACK', '1') == '1':
        logger.info('Vector search empty; running local vector fallback')
        filtered, local_vector_docs = _local_vector_search(
            collection=collection,
            workspace_id=workspace_id,
            query_embedding=query_embedding,
            top_k=top_k,
            min_score=min_score,
            knowledge_base_ids=effective_kb_ids,
        )
        used_local_vector = True

    # Optional text fallback if vector search returns nothing
    if not filtered and fallback_text:
        logger.info("Vector search empty; running text fallback")
        words = [w for w in query.split() if len(w) >= 4]
        regex = "|".join(words) if words else query
        text_match = {
            "workspace_id": workspace_id,
            "status": "published",
            "text": {"$regex": regex, "$options": "i"},
        }
        if effective_kb_ids is not None:
            text_match["knowledge_base_id"] = {"$in": effective_kb_ids}

        text_results = list(collection.find(
            text_match,
            {"_id": 0, "public_id": 1, "text": 1,
                "source_name": 1, "source_type": 1, "chunk_index": 1},
        ).limit(int(top_k)))
        for r in text_results:
            r["score"] = 0.0
        filtered = text_results
        used_fallback = True

    if include_raw:
        diagnostics = {
            "workspace_id": workspace_id,
            "db": DB_NAME,
            "collection": COLLECTION_NAME,
            "vector_index": VECTOR_INDEX_NAME,
            "total_docs": collection.count_documents(
                {"workspace_id": workspace_id} if effective_kb_ids is None else {
                    "workspace_id": workspace_id,
                    "knowledge_base_id": {"$in": effective_kb_ids},
                }
            ),
            "published_docs": collection.count_documents(match_filter),
            "knowledge_base_ids": effective_kb_ids,
            "used_low_score": used_low_score,
            "used_local_vector": used_local_vector,
            "local_vector_docs": local_vector_docs,
            "used_fallback": used_fallback,
        }
        return filtered, results, diagnostics
    return filtered
