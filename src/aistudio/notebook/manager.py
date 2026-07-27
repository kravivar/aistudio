import sqlite3
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

DB_PATH = Path("./data/notebook.db").resolve()

def init_notebook_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS notes (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            tags TEXT,
            created_at INTEGER,
            updated_at INTEGER
        )
        """)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS sources (
            id TEXT PRIMARY KEY,
            note_id TEXT,
            filename TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at INTEGER
        )
        """)
        conn.commit()

class NotebookManager:
    def __init__(self):
        init_notebook_db()

    def create_note(self, title: str, content: str, tags: Optional[List[str]] = None) -> Dict[str, Any]:
        note_id = f"note_{int(time.time()*1000)}"
        now = int(time.time())
        tags_str = json.dumps(tags or [])
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO notes (id, title, content, tags, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                (note_id, title, content, tags_str, now, now)
            )
            conn.commit()
        return {
            "id": note_id,
            "title": title,
            "content": content,
            "tags": tags or [],
            "created_at": now,
            "updated_at": now
        }

    def get_notes(self) -> List[Dict[str, Any]]:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM notes ORDER BY updated_at DESC")
            rows = cursor.fetchall()
            return [
                {
                    "id": row["id"],
                    "title": row["title"],
                    "content": row["content"],
                    "tags": json.loads(row["tags"] or "[]"),
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"]
                }
                for row in rows
            ]

    def search_notes(self, query: str) -> List[Dict[str, Any]]:
        notes = self.get_notes()
        q = query.lower()
        return [
            n for n in notes
            if q in n["title"].lower() or q in n["content"].lower() or any(q in t.lower() for t in n["tags"])
        ]

notebook_manager = NotebookManager()
