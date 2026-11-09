"""Optional vec_episodes support for imported BEAM banks."""

import json
import re
import sqlite3


def legacy_vector_type(db):
    row = db.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='vec_episodes'").fetchone()
    if row is None:
        return None
    schema = row[0] or ""
    if not re.search(r"\bUSING\s+vec0\s*\(", schema, re.I):
        return "float32"
    try:
        db.execute("SELECT vec_version()")
    except sqlite3.OperationalError:
        try:
            import sqlite_vec

            db.enable_load_extension(True)
            try:
                sqlite_vec.load(db)
            finally:
                db.enable_load_extension(False)
        except (ImportError, AttributeError, OSError, sqlite3.Error):
            return None
    column = re.search(r'\bembedding["`\]]?\s+(bit|int8)\s*\[', schema, re.I)
    return column[1].lower() if column else "float32"


def insert_legacy_vector(db, vector_type, rowid, vector):
    expression = {"bit": "vec_quantize_binary(?)", "int8": "vec_quantize_int8(?, 'unit')"}.get(vector_type, "?")
    db.execute(f"INSERT INTO vec_episodes(rowid,embedding) VALUES (?,{expression})",
               (rowid, json.dumps(vector, separators=(",", ":"), allow_nan=False)))


def delete_legacy_vectors(db):
    if legacy_vector_type(db):
        db.execute("DELETE FROM vec_episodes")
