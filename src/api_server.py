"""
Flask API Server — AI-Based Smart File Assistant
=================================================
Serves the static SPA frontend (static/index.html) and exposes
JSON API endpoints consumed by the frontend.

Run:
    venv\\Scripts\\activate
    python api_server.py

Frontend: http://127.0.0.1:5000
"""

import os
import sys
import logging
from datetime import datetime
from functools import wraps

from flask import Flask, request, jsonify, session, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))
load_dotenv()

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger("api_server")

# ── Flask App ───────────────────────────────────────────────────────────────
app = Flask(
    __name__,
    static_folder=os.path.join(_PROJECT_ROOT, "static"),
    static_url_path=""
)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "fileai-dev-secret-2026")
CORS(app, supports_credentials=True)

# ── Backend Init ────────────────────────────────────────────────────────────
_db = None
_qa = None
_qa_init_error = None


def _get_db():
    global _db
    if _db is None:
        try:
            try:
                from src.modules.chromadb_handler import ChromaDBHandler
            except ImportError:
                from modules.chromadb_handler import ChromaDBHandler
            db_dir = os.path.join(_PROJECT_ROOT, "data", "chroma")
            
            # Vercel Serverless File System Workaround
            if os.environ.get("VERCEL"):
                import shutil
                tmp_dir = "/tmp/chroma"
                if not os.path.exists(tmp_dir):
                    try:
                        shutil.copytree(db_dir, tmp_dir)
                        logger.info(f"Copied ChromaDB to ephemeral {tmp_dir}")
                    except Exception as ce:
                        logger.warning(f"Failed to copy ChromaDB to {tmp_dir}: {ce}")
                db_dir = tmp_dir

            os.makedirs(db_dir, exist_ok=True)
            _db = ChromaDBHandler(persist_directory=db_dir, collection_name="document_chunks")
            logger.info(f"ChromaDB initialised ({_db.get_count()} vectors)")
        except Exception as e:
            logger.error(f"ChromaDB init failed: {e}")
            _db = None
    return _db


def _use_langchain_pipeline():
    return os.getenv("USE_LANGCHAIN_PIPELINE", "false").strip().lower() in {"1", "true", "yes", "on"}


def _get_qa():
    global _qa, _qa_init_error
    if _qa is None:
        db = _get_db()
        if db is None:
            _qa_init_error = "Vector database is not available"
            return None
        try:
            use_langchain = _use_langchain_pipeline()

            if use_langchain:
                try:
                    from src.modules.langchain_qa_system import LangChainQASystem
                    qa_cls = LangChainQASystem
                except ImportError:
                    from modules.langchain_qa_system import LangChainQASystem
                    qa_cls = LangChainQASystem
            else:
                try:
                    from src.modules.qa_system import QASystem
                    qa_cls = QASystem
                except ImportError:
                    from modules.qa_system import QASystem
                    qa_cls = QASystem

            try:
                from src.modules.openai_handler import OpenAIHandler
            except ImportError:
                from modules.openai_handler import OpenAIHandler

            _qa = qa_cls(
                vector_db=db,
                openai_handler=OpenAIHandler(),
                top_k=int(os.getenv("TOP_K", 8))
            )
            mode = "LangChainQASystem" if use_langchain else "QASystem"
            logger.info(f"{mode} initialised")
            _qa_init_error = None
        except Exception as e:
            _qa_init_error = str(e)
            logger.error(f"QASystem init failed: {e}")
            _qa = None
    return _qa


# ── Session Defaults ─────────────────────────────────────────────────────────

_user_data_store = {}

def _get_session_data():
    u = session.get("user")
    if not u:
        return None
    email = u.get("email")
    if email not in _user_data_store:
        _user_data_store[email] = {
            "chat_history": [],
            "query_history": [],
            "settings": {
                "model": os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                "temperature": 0.3,
                "max_tokens": 800,
                "top_k": 8
            }
        }
    return _user_data_store[email]



def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("user"):
            return jsonify({"error": "Not authenticated"}), 401
        return f(*args, **kwargs)
    return decorated


# ── Static SPA ───────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


# ── Auth Endpoints ────────────────────────────────────────────────────────────
@app.route("/api/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    email = data.get("email", "").strip()
    password = data.get("password", "").strip()

    # Simple mock auth — accept any non-empty email/password
    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    # Derive display name from email
    name_part = email.split("@")[0].replace(".", " ").replace("_", " ").title()
    initials = "".join(w[0].upper() for w in name_part.split()[:2])

    session["user"] = {
        "email": email,
        "name": name_part,
        "initials": initials,
        "role": "RESEARCHER",
        "created": "2026-02-15"
    }
    return jsonify({"success": True, "user": session["user"]})


@app.route("/api/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"success": True})


@app.route("/api/me")
def me():
    if not session.get("user"):
        return jsonify({"authenticated": False}), 200
    return jsonify({"authenticated": True, "user": session["user"]})


# ── Chat Endpoints ─────────────────────────────────────────────────────────────
@app.route("/api/chat", methods=["POST"])
@login_required
def chat():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    stream = data.get("stream", False)

    if not question:
        return jsonify({"error": "Question is required"}), 400

    qa = _get_qa()
    if qa is None:
        if not os.getenv("OPENAI_API_KEY"):
            return jsonify({"error": "AI backend is unavailable: OPENAI_API_KEY is missing. Add it to your project .env file."}), 503
        if _use_langchain_pipeline():
            return jsonify({"error": "AI backend is unavailable in LangChain mode. Ensure langchain-openai is installed and configuration is valid.", "details": _qa_init_error}), 503
        return jsonify({"error": "AI backend is unavailable. Check ChromaDB initialization and backend logs.", "details": _qa_init_error}), 503

    # Apply user settings to QA system
    user_data = _get_session_data()
    settings = user_data.get("settings", {})
    qa.model = settings.get("model")
    qa.query_processor.default_top_k = int(settings.get("top_k", 5))

    if not stream:
        result = qa.answer_with_followup(question)

        # Format sources
        sources = []
        for src in result.get("sources", []):
            pages = sorted(src.get("pages", []))
            sources.append({
                "file": src.get("source_file", "Unknown"),
                "pages": pages,
                "score": round(src.get("avg_score", 0), 2)
            })

        answer_obj = {
            "question": question,
            "answer": result.get("answer", ""),
            "sources": sources,
            "confidence": round(result.get("confidence", 0) * 100),
            "time_seconds": result.get("time_seconds", 0),
            "error": result.get("error")
        }

        # Append to session chat history
        history = user_data["chat_history"]
        history.append({"role": "user", "content": question, "ts": datetime.now().strftime("%H:%M")})
        history.append({
            "role": "assistant",
            "content": result.get("answer", ""),
            "sources": sources,
            "confidence": answer_obj["confidence"],
            "ts": datetime.now().strftime("%H:%M")
        })
        user_data["chat_history"] = history[-100:]  # cap at last 100 messages

        # Append to query history
        qh = user_data["query_history"]
        qh.append({
            "query": question,
            "date": datetime.now().strftime("Today · %H:%M"),
            "sources": ", ".join(s["file"] for s in sources)
        })
        user_data["query_history"] = qh[-50:]

        return jsonify(answer_obj)

    # Streaming mode
    def generate():
        import json
        gen = qa.answer_with_followup_stream(question)
        
        try:
            # First item is the metadata dict
            meta = next(gen)
            
            # Format sources for meta
            sources = []
            for src in meta.get("sources", []):
                pages = sorted(src.get("pages", []))
                sources.append({
                    "file": src.get("source_file", "Unknown"),
                    "pages": pages,
                    "score": round(src.get("avg_score", 0), 2)
                })
                
            meta_obj = {
                "type": "meta",
                "sources": sources,
                "confidence": round(meta.get("confidence", 0) * 100),
                "error": meta.get("error")
            }
            yield f"data: {json.dumps(meta_obj)}\n\n"
            
            if meta.get("error"):
                return
                
            # Next items are string chunks
            for chunk in gen:
                if isinstance(chunk, str):
                    yield f"data: {json.dumps({'type': 'chunk', 'text': chunk})}\n\n"
                elif isinstance(chunk, dict):
                    # Final result dict
                    final_result = chunk
                    
                    # Compute final confidence
                    final_confidence = round(final_result.get("confidence", 0) * 100)
                    
                    # Update session history
                    history = user_data["chat_history"]
                    history.append({"role": "user", "content": question, "ts": datetime.now().strftime("%H:%M")})
                    history.append({
                        "role": "assistant",
                        "content": final_result.get("answer", ""),
                        "sources": sources,
                        "confidence": final_confidence,
                        "ts": datetime.now().strftime("%H:%M")
                    })
                    user_data["chat_history"] = history[-100:]
                    
                    qh = user_data["query_history"]
                    qh.append({
                        "query": question,
                        "date": datetime.now().strftime("Today · %H:%M"),
                        "sources": ", ".join(s["file"] for s in sources)
                    })
                    user_data["query_history"] = qh[-50:]
                    
                    done_obj = {
                        "type": "done",
                        "confidence": final_confidence,
                        "time_seconds": final_result.get("time_seconds", 0)
                    }
                    yield f"data: {json.dumps(done_obj)}\n\n"
                    
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    from flask import Response, stream_with_context
    return Response(stream_with_context(generate()), mimetype='text/event-stream')


@app.route("/api/chat/history")
@login_required
def chat_history():
    user_data = _get_session_data()
    return jsonify({"history": user_data["chat_history"] if user_data else []})


@app.route("/api/chat/history", methods=["DELETE"])
@login_required
def clear_chat_history():
    user_data = _get_session_data()
    if user_data:
        user_data["chat_history"] = []
    qa = _get_qa()
    if qa:
        qa.reset_conversation()
    return jsonify({"success": True})


# ── Documents Endpoint ────────────────────────────────────────────────────────
@app.route("/api/documents")
@login_required
def documents():
    db = _get_db()
    if db is None:
        return jsonify({"documents": [], "total_vectors": 0})

    doc_names = db.list_documents()
    total_vectors = db.get_count()

    # Static metadata for known documents
    meta_map = {
        "document_pdf.pdf": {
            "title": "NIST Cybersecurity Framework",
            "summary": "Standards and guidelines for managing cybersecurity-related risks in critical infrastructure and other sectors.",
            "topics": ["GOVERNANCE", "RISK", "IDENTIFY"],
            "pages": 204,
            "category": "CYBERSECURITY"
        },
        "NIST.IR.8596.iprd.pdf": {
            "title": "NIST AI Security Profile",
            "summary": "AI risk management framework guidance for enterprise-grade AI systems and governance.",
            "topics": ["AI GOVERNANCE", "RISK", "POLICY"],
            "pages": 98,
            "category": "AI GOVERNANCE"
        },
        "NIST.SP.500-291r2.pdf": {
            "title": "NIST Cloud Computing Standards",
            "summary": "Roadmap for enterprise cloud computing standards adoption, interoperability, and data portability.",
            "topics": ["CLOUD", "STANDARDS", "INTEROP"],
            "pages": 212,
            "category": "CLOUD"
        },
        "IISF-Version-2.pdf": {
            "title": "Enterprise IoT Security Framework",
            "summary": "Standardized security architecture for Industrial IoT edge connectivity and data integrity protocols.",
            "topics": ["IOT SECURITY", "EDGE", "PROTOCOL"],
            "pages": 156,
            "category": "IOT SECURITY"
        }
    }

    result = []
    for name in doc_names:
        m = meta_map.get(name, {
            "title": name.replace("_", " ").replace(".pdf", ""),
            "summary": "Enterprise technology standards document indexed in the knowledge base.",
            "topics": ["STANDARDS"],
            "pages": "--",
            "category": "DOCUMENT"
        })
        result.append({"filename": name, **m})

    # If DB empty, show static placeholder docs
    if not result:
        result = list(meta_map.values())
        for i, (k, v) in enumerate(meta_map.items()):
            result[i] = {"filename": k, **v}

    return jsonify({"documents": result, "total_vectors": total_vectors})


@app.route("/api/documents/count")
@login_required
def documents_count():
    db = _get_db()
    count = db.get_count() if db else 0
    docs = len(db.list_documents()) if db else 0
    return jsonify({"vectors": count, "documents": docs})


@app.route("/api/raw_documents/<path:filename>")
@login_required
def raw_documents(filename):
    pdf_dir = os.path.join(os.path.dirname(app.root_path), "data", "pdfs")
    return send_from_directory(pdf_dir, filename)


# ── Insights Endpoint ─────────────────────────────────────────────────────────
@app.route("/api/insights")
@login_required
def insights():
    concepts = [
        {
            "title": "Cybersecurity Resilience",
            "description": "The infrastructure requires hardening against modern attack vectors. Existing protocols must align with NIST CSF guidance.",
            "tags": ["SECURITY", "PROTOCOL"]
        },
        {
            "title": "AI Governance",
            "description": "Autonomous systems decision-making must remain transparent. NIST AI RMF provides the governance foundation.",
            "tags": ["ETHICS", "POLICY"]
        },
        {
            "title": "Enterprise Security Architecture",
            "description": "Edge computing and cloud workloads require unified security architecture spanning on-prem and cloud boundaries.",
            "tags": ["INFRASTRUCTURE", "CLOUD"]
        },
        {
            "title": "Predictive Risk Management",
            "description": "Automated scaling and predictive analytics must incorporate cybersecurity risk models from NIST frameworks.",
            "tags": ["RISK", "ANALYTICS"]
        }
    ]

    highlights = [
        {
            "insight": "NIST defines 5 key cybersecurity functions: Identify, Protect, Detect, Respond, Recover.",
            "source": "NIST.CSWP.29 · Page 14",
            "tag": "FRAMEWORK"
        },
        {
            "insight": "Legacy encryption layers remain the primary vulnerability point in the majority of observed system breaches.",
            "source": "NIST.CSF · Section 3.2",
            "tag": "SECURITY"
        },
        {
            "insight": "Human-in-the-loop validation is a regulatory requirement for high-stakes AI decision-making systems.",
            "source": "NIST.IR.8596 · Page 42",
            "tag": "AI GOVERNANCE"
        }
    ]

    return jsonify({"concepts": concepts, "highlights": highlights})


# ── FAQ Endpoint ────────────────────────────────────────────────────────────────
@app.route("/api/faq")
def faq():
    items = [
        {
            "q": "What is the NIST Cybersecurity Framework?",
            "a": "The NIST Cybersecurity Framework (CSF) consists of standards, guidelines, and best practices to manage cybersecurity-related risk. It provides a prioritized, flexible, and cost-effective approach to help organizations protect and build resilience in critical infrastructure.",
            "open": True
        },
        {
            "q": "How does AI retrieve answers from documents?",
            "a": "The system uses a Retrieval-Augmented Generation (RAG) pipeline: your query is embedded into a vector, semantically searched against indexed document chunks in ChromaDB, and the top results are passed as context to an LLM (GPT) which generates a grounded, cited answer."
        },
        {
            "q": "Can I upload my own documents?",
            "a": "No. This is a read-only system. All domain documents are preloaded and managed by the backend. The system only answers based on the enterprise technology standards documents already indexed."
        },
        {
            "q": "Why are technology standards important?",
            "a": "Technology standards ensure interoperability, security, and governance consistency across enterprise systems. Standards like NIST CSF provide organizations with proven frameworks to reduce risk and meet regulatory requirements."
        },
        {
            "q": "What role does AI play in cybersecurity?",
            "a": "AI serves as the retrieval and synthesis layer. It uses embeddings and Large Language Models to interpret natural language queries and extract precise, cited answers from indexed enterprise documentation."
        },
        {
            "q": "What documents are available in this system?",
            "a": "The system includes: NIST Cybersecurity Framework (CSWP 29), NIST AI Security Profile (IR 8596), Cloud Computing Standards Roadmap (SP 500-291r2), and the Enterprise IoT Security Framework (IISF v2)."
        }
    ]
    return jsonify({"faq": items})


# ── Search History Endpoint ────────────────────────────────────────────────────
@app.route("/api/search-history")
@login_required
def search_history():
    user_data = _get_session_data()
    return jsonify({"history": user_data["query_history"] if user_data else []})


@app.route("/api/search-history", methods=["DELETE"])
@login_required
def clear_search_history():
    user_data = _get_session_data()
    if user_data:
        user_data["query_history"] = []
    return jsonify({"success": True})


# ── Settings Endpoints ────────────────────────────────────────────────────────
@app.route("/api/settings")
@login_required
def get_settings():
    user_data = _get_session_data()
    return jsonify(user_data["settings"] if user_data else {})


@app.route("/api/settings", methods=["POST"])
@login_required
def save_settings():
    data = request.get_json(silent=True) or {}
    user_data = _get_session_data()
    if not user_data:
        return jsonify({"error": "No valid session"}), 400
    
    current = user_data["settings"]
    current.update({
        "model": data.get("model", current.get("model")),
        "temperature": float(data.get("temperature", current.get("temperature", 0.3))),
        "max_tokens": int(data.get("max_tokens", current.get("max_tokens", 600))),
        "top_k": int(data.get("top_k", current.get("top_k", 5)))
    })
    user_data["settings"] = current

    # Apply to live QA system if running
    qa = _get_qa()
    if qa:
        qa.model = current.get("model")
        qa.query_processor.default_top_k = current["top_k"]

    return jsonify({"success": True, "settings": current})


# ── Profile Endpoint ──────────────────────────────────────────────────────────
@app.route("/api/profile")
@login_required
def profile():
    user = session.get("user", {})
    user_data = _get_session_data()
    
    chat_hist = user_data["chat_history"] if user_data else []
    query_hist = user_data["query_history"] if user_data else []

    user_msgs = [m for m in chat_hist if m.get("role") == "user"]
    assistant_msgs = [m for m in chat_hist if m.get("role") == "assistant"]
    avg_confidence = 0
    if assistant_msgs:
        confs = [m.get("confidence", 0) for m in assistant_msgs]
        avg_confidence = round(sum(confs) / len(confs))

    # Count unique sources referenced
    all_sources = set()
    for m in assistant_msgs:
        for s in m.get("sources", []):
            all_sources.add(s.get("file", ""))

    return jsonify({
        "user": user,
        "stats": {
            "queries_asked": len(user_msgs),
            "documents_referenced": len(all_sources),
            "search_history_count": len(query_hist),
            "avg_confidence": avg_confidence
        }
    })


# ── Run ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("Initialising backend (DB + QA)...")
    _get_db()
    _get_qa()
    port = int(os.getenv("PORT", "5050"))
    logger.info(f"Starting FileAI API server on http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=True, use_reloader=False)
