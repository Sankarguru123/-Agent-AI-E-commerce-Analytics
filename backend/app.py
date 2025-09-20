# backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import os, json, sqlite3, hashlib, time
from datetime import datetime
import re

# Use Groq client instead of OpenAI
from groq import Groq

load_env = False
try:
    from dotenv import load_dotenv
    load_dotenv()
    load_env = True
except Exception:
    pass

# Load GROQ key from environment
GROQ_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_KEY:
    raise RuntimeError("GROQ_API_KEY not set in environment (set backend/.env)")

# Instantiate Groq client
client = Groq(api_key=GROQ_KEY)

# Cache DB filename (local file)
CACHE_DB = os.getenv("LLM_CACHE_DB", "llm_cache.db")

# Model and system prompt
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# System prompt: instruct assistant to produce a single valid SQLite SELECT (or WITH..SELECT)
SYSTEM_PROMPT = (
    "You are an assistant that converts plain English into a single valid SQLite SQL SELECT or WITH ... SELECT query. "
    "Assume the table name is exactly 'orders'. Return ONLY the SQL statement (no explanation, no backticks). "
    "Use standard SQLite syntax. If ambiguous, choose sensible defaults and include LIMIT 1000."
)

# SQL safety regexes
SQL_ALLOW_PATTERN = re.compile(r'^\s*(WITH|SELECT)\b', re.IGNORECASE)
SQL_FORBIDDEN_PATTERN = re.compile(
    r'\b(INSERT|UPDATE|DELETE|DROP|ALTER|PRAGMA|CREATE|ATTACH|DETACH|REPLACE|EXEC|EXECUTE|TRUNCATE)\b',
    re.IGNORECASE
)

app = Flask(__name__)
CORS(app)

# --- cache helpers (sqlite-backed) ---
def _init_cache_db():
    conn = sqlite3.connect(CACHE_DB)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS llm_cache (
            request_hash TEXT PRIMARY KEY,
            nl_query TEXT,
            schema_json TEXT,
            sql_text TEXT,
            created_at INTEGER
        )
    """)
    conn.commit()
    conn.close()

def _make_request_hash(nl_query: str, schema: dict):
    key = json.dumps({"nl_query": nl_query, "schema": schema}, sort_keys=True)
    return hashlib.sha256(key.encode("utf-8")).hexdigest()

def cache_get(request_hash):
    conn = sqlite3.connect(CACHE_DB)
    cur = conn.cursor()
    cur.execute("SELECT sql_text, created_at FROM llm_cache WHERE request_hash = ?", (request_hash,))
    row = cur.fetchone()
    conn.close()
    if row:
        return row[0], row[1]
    return None, None

def cache_set(request_hash, nl_query, schema, sql_text):
    conn = sqlite3.connect(CACHE_DB)
    cur = conn.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO llm_cache (request_hash, nl_query, schema_json, sql_text, created_at) VALUES (?, ?, ?, ?, ?)",
        (request_hash, nl_query, json.dumps(schema), sql_text, int(time.time()))
    )
    conn.commit()
    conn.close()

_init_cache_db()

@app.route("/generate_sql", methods=["POST"])
def generate_sql():
    data = request.get_json(force=True)
    nl_query = (data.get("nl_query") or "").strip()
    schema = data.get("schema", {})

    if not nl_query:
        return jsonify({"ok": False, "error": "nl_query is required"}), 400

    request_hash = _make_request_hash(nl_query, schema)
    cached_sql, cached_ts = cache_get(request_hash)
    if cached_sql:
        return jsonify({"ok": True, "sql": cached_sql, "cached": True, "cached_at": cached_ts})

    schema_desc = ", ".join([f"{col} ({typ})" for col, typ in schema.items()]) if schema else "columns: unknown"

    # Keep system prompt focused on generating SQL only.
    # Use a minimal user prompt with the schema + natural language query.
    prompt_user = (
        f"Table name: orders.\nSchema: {schema_desc}\n\n"
        f"Natural language: {nl_query}\n\n"
        "Return a single valid SQLite SELECT query only. Include LIMIT 1000 if needed."
    )

    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_user}
            ],
            temperature=0.0,
            max_tokens=700,
            n=1
        )

        # Extract assistant reply (Groq uses OpenAI-compatible shape)
        sql_text = resp.choices[0].message.content.strip()
        print("Generated SQL:", sql_text)
    except Exception as exc:
        return jsonify({"ok": False, "error": f"Groq API error: {str(exc)}"}), 500

    # Basic sanitization
    if SQL_FORBIDDEN_PATTERN.search(sql_text):
        return jsonify({"ok": False, "error": "Generated SQL contains forbidden statements."}), 400
    if not SQL_ALLOW_PATTERN.search(sql_text):
        return jsonify({"ok": False, "error": "Generated SQL is not a SELECT / WITH ... SELECT statement."}), 400
    if not sql_text.endswith(";"):
        sql_text = sql_text + ";"

    # Save to cache
    try:
        cache_set(request_hash, nl_query, schema, sql_text)
    except Exception:
        pass

    return jsonify({"ok": True, "sql": sql_text, "cached": False})

@app.route("/cache_list", methods=["GET"])
def cache_list():
    try:
        conn = sqlite3.connect(CACHE_DB)
        cur = conn.cursor()
        cur.execute("SELECT request_hash, nl_query, datetime(created_at,'unixepoch') FROM llm_cache ORDER BY created_at DESC LIMIT 50")
        rows = cur.fetchall()
        conn.close()
        items = [{"hash": r[0], "nl_query": r[1], "created_at": r[2]} for r in rows]
        return jsonify({"ok": True, "items": items})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host=host, port=port, debug=debug)




# # backend/app.py
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import os, json, sqlite3, hashlib, time
# from datetime import datetime
# import re
#
# # OpenAI v1 client
# from openai import OpenAI
#
# load_env = False
# try:
#     from dotenv import load_dotenv
#     load_dotenv()
#     load_env = True
# except Exception:
#     pass
#
# # OPENAI_KEY = os.getenv("OPENAI_API_KEY")
# # if not OPENAI_KEY:
# #     raise RuntimeError("OPENAI_API_KEY not set in environment (set backend/.env)")
# #
# # # instantiate client (it reads OPENAI_API_KEY automatically, or you can pass api_key=OPENAI_KEY)
# # client = OpenAI(api_key=OPENAI_KEY)
# #
# # # Cache DB filename (local file)
# # CACHE_DB = os.getenv("LLM_CACHE_DB", "llm_cache.db")
# #
# # # Model and system prompt
# # OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
#
#
# import os
# from groq import Groq  # instead of openai
#
# # Load API Key from environment
# GROQ_KEY = os.getenv("GROQ_API_KEY")
# if not GROQ_KEY:
#     raise RuntimeError("GROQ_API_KEY not set in environment (set backend/.env)")
#
# # Instantiate Groq client
# client = Groq(api_key=GROQ_KEY)
#
# # Cache DB filename (local file)
# CACHE_DB = os.getenv("LLM_CACHE_DB", "llm_cache.db")
#
# # Model and system prompt
# # Example: "llama-3.1-8b-instant" (adjust as needed)
# GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
#
# SYSTEM_PROMPT = (
#
#     "You are an assistant that converts plain English into a single valid SQLite SQL SELECT or WITH ... SELECT query. "
#     "Assume the table name is exactly 'orders'. Return ONLY the SQL statement (no explanation, no backticks). "
#     "Use standard SQLite syntax. If ambiguous, choose sensible defaults and include LIMIT 1000."
# )
#
# SQL_ALLOW_PATTERN = re.compile(r'^\s*(WITH|SELECT)\b', re.IGNORECASE)
# SQL_FORBIDDEN_PATTERN = re.compile(
#     r'\b(INSERT|UPDATE|DELETE|DROP|ALTER|PRAGMA|CREATE|ATTACH|DETACH|REPLACE|EXEC|EXECUTE|TRUNCATE)\b',
#     re.IGNORECASE
# )
#
# app = Flask(__name__)
# CORS(app)
#
# # --- cache helpers (sqlite-backed) ---
# def _init_cache_db():
#     conn = sqlite3.connect(CACHE_DB)
#     cur = conn.cursor()
#     cur.execute("""
#         CREATE TABLE IF NOT EXISTS llm_cache (
#             request_hash TEXT PRIMARY KEY,
#             nl_query TEXT,
#             schema_json TEXT,
#             sql_text TEXT,
#             created_at INTEGER
#         )
#     """)
#     conn.commit()
#     conn.close()
#
# def _make_request_hash(nl_query: str, schema: dict):
#     key = json.dumps({"nl_query": nl_query, "schema": schema}, sort_keys=True)
#     return hashlib.sha256(key.encode("utf-8")).hexdigest()
#
# def cache_get(request_hash):
#     conn = sqlite3.connect(CACHE_DB)
#     cur = conn.cursor()
#     cur.execute("SELECT sql_text, created_at FROM llm_cache WHERE request_hash = ?", (request_hash,))
#     row = cur.fetchone()
#     conn.close()
#     if row:
#         return row[0], row[1]
#     return None, None
#
# def cache_set(request_hash, nl_query, schema, sql_text):
#     conn = sqlite3.connect(CACHE_DB)
#     cur = conn.cursor()
#     cur.execute(
#         "INSERT OR REPLACE INTO llm_cache (request_hash, nl_query, schema_json, sql_text, created_at) VALUES (?, ?, ?, ?, ?)",
#         (request_hash, nl_query, json.dumps(schema), sql_text, int(time.time()))
#     )
#     conn.commit()
#     conn.close()
#
# _init_cache_db()
#
# @app.route("/generate_sql", methods=["POST"])
# def generate_sql():
#     data = request.get_json(force=True)
#     nl_query = (data.get("nl_query") or "").strip()
#     schema = data.get("schema", {})
#
#     if not nl_query:
#         return jsonify({"ok": False, "error": "nl_query is required"}), 400
#
#     request_hash = _make_request_hash(nl_query, schema)
#     cached_sql, cached_ts = cache_get(request_hash)
#     if cached_sql:
#         return jsonify({"ok": True, "sql": cached_sql, "cached": True, "cached_at": cached_ts})
#
#     schema_desc = ", ".join([f"{col} ({typ})" for col, typ in schema.items()]) if schema else "columns: unknown"
#     prompt_user = (
#         f"Table name: orders.\nSchema: {schema_desc}\n\n"
#         f"Natural language: {nl_query}\n\n"
#         "Return a single valid SQLite SELECT query only."
#         """
# You will be given CSV data containing order transactions with fields like order_id, product, quantity, price, customer, region, and date.
#
# 1. Answer the user's question directly using the provided CSV data.
# 2. Support both summary-type questions (totals, averages, top products, trends)
#    AND detailed queries (e.g., "What did Sankar purchase on 2023-05-01?").
# 3. For customer-specific questions, filter by customer name and return matching orders
#    with order_id, product, quantity, price, and date.
# 4. For product-specific questions, show which customers purchased it and when.
# 5. Always calculate totals, averages, or other metrics when relevant.
# 6. Return results in clear, structured English with tables if the output is tabular.
# 7. If the dataset sample is small, analyze it carefully but still provide useful insights.
#
# """
#     )
#
#     # ==== NEW: use OpenAI v1 client ====
#     try:
#         resp = client.chat.completions.create(
#             model=GROQ_MODEL,
#             messages=[
#                 {"role": "system", "content": SYSTEM_PROMPT},
#                 {"role": "user", "content": prompt_user}
#             ],
#             temperature=0.0,
#             max_tokens=700,
#             n=1
#         )
#
#         # Extract assistant reply
#         # Access the generated text:
#         # In v1 client the shape is resp.choices[0].message.content
#         sql_text = resp.choices[0].message.content.strip()
#         print(sql_text,'sql')
#     except Exception as exc:
#         return jsonify({"ok": False, "error": f"OpenAI error: {str(exc)}"}), 500
#
#     # Basic sanitization
#     if SQL_FORBIDDEN_PATTERN.search(sql_text):
#         return jsonify({"ok": False, "error": "Generated SQL contains forbidden statements."}), 400
#     if not SQL_ALLOW_PATTERN.search(sql_text):
#         return jsonify({"ok": False, "error": "Generated SQL is not a SELECT / WITH ... SELECT statement."}), 400
#     if not sql_text.endswith(";"):
#         sql_text = sql_text + ";"
#
#     # Save to cache
#     try:
#         cache_set(request_hash, nl_query, schema, sql_text)
#     except Exception:
#         pass
#
#     return jsonify({"ok": True, "sql": sql_text, "cached": False})
#
# @app.route("/cache_list", methods=["GET"])
# def cache_list():
#     try:
#         conn = sqlite3.connect(CACHE_DB)
#         cur = conn.cursor()
#         cur.execute("SELECT request_hash, nl_query, datetime(created_at,'unixepoch') FROM llm_cache ORDER BY created_at DESC LIMIT 50")
#         rows = cur.fetchall()
#         conn.close()
#         items = [{"hash": r[0], "nl_query": r[1], "created_at": r[2]} for r in rows]
#         return jsonify({"ok": True, "items": items})
#     except Exception as e:
#         return jsonify({"ok": False, "error": str(e)}), 500
#
# if __name__ == "__main__":
#     host = os.getenv("HOST", "0.0.0.0")
#     port = int(os.getenv("PORT", 5000))
#     debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
#     app.run(host=host, port=port, debug=debug)
