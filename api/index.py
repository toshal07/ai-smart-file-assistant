import os
import sys

# On Vercel, use pysqlite3-binary instead of the system sqlite3
# This is required for ChromaDB which needs SQLite >= 3.35.0
if os.environ.get("VERCEL"):
    try:
        __import__("pysqlite3")
        sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
    except ImportError:
        pass

# Hack to force Vercel's AST parser to bundle the data directory
try:
    import data
    import data.chroma
except ImportError:
    pass

# Import the pre-configured Flask app so Vercel can serve it
from src.api_server import app
