import sqlite3
import threading
from datetime import datetime


class MemoryManager:

    def __init__(self, database_path="memory/robot_memory.db"):
        self.database_path = database_path
        self.lock = threading.Lock()

        self._initialize_database()

    # =====================================================
    # DATABASE
    # =====================================================

    def _connect(self):
        return sqlite3.connect(
            self.database_path,
            check_same_thread=False
        )

    def _initialize_database(self):

        with self.lock:

            db = self._connect()
            cursor = db.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,

                    memory_type TEXT NOT NULL,

                    memory_key TEXT NOT NULL,

                    memory_value TEXT NOT NULL,

                    created_at TEXT NOT NULL,

                    updated_at TEXT NOT NULL,

                    UNIQUE(memory_type, memory_key)
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,

                    started_at TEXT NOT NULL,

                    ended_at TEXT,

                    summary TEXT
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,

                    session_id INTEGER NOT NULL,

                    role TEXT NOT NULL,

                    content TEXT NOT NULL,

                    timestamp TEXT NOT NULL
                )
            """)

            db.commit()
            db.close()

    # =====================================================
    # SESSION
    # =====================================================

    def start_session(self):

        now = datetime.now().isoformat()

        with self.lock:

            db = self._connect()
            cursor = db.cursor()

            cursor.execute("""
                INSERT INTO sessions(started_at)
                VALUES(?)
            """, (now,))

            session_id = cursor.lastrowid

            db.commit()
            db.close()

        return session_id

    # =====================================================
    # MESSAGE
    # =====================================================

    def save_message(self,session_id,role,content):

        if not content:
            return

        now = datetime.now().isoformat()

        with self.lock:

            db = self._connect()
            cursor = db.cursor()

            cursor.execute("""
                INSERT INTO messages
                (
                    session_id,
                    role,
                    content,
                    timestamp
                )
                VALUES (?, ?, ?, ?)
            """, (
                session_id,
                role,
                content,
                now
            ))

            db.commit()
            db.close()

    # =====================================================
    # LONG-TERM MEMORY
    # =====================================================

    def save_memory(self,memory_type,key,value):

        if not key or not value:
            return

        now = datetime.now().isoformat()

        with self.lock:

            db = self._connect()
            cursor = db.cursor()

            cursor.execute("""
                INSERT INTO memories
                (
                    memory_type,
                    memory_key,
                    memory_value,
                    created_at,
                    updated_at
                )

                VALUES (?, ?, ?, ?, ?)

                ON CONFLICT(memory_type, memory_key)

                DO UPDATE SET

                    memory_value =
                        excluded.memory_value,

                    updated_at =
                        excluded.updated_at
            """, (
                memory_type,
                key,
                value,
                now,
                now
            ))

            db.commit()
            db.close()

        print(
            f"🧠 Memory saved: "
            f"[{memory_type}] {key} = {value}"
        )

    # =====================================================
    # GET MEMORIES
    # =====================================================

    def get_memories(self, limit=100):

        with self.lock:

            db = self._connect()
            cursor = db.cursor()

            cursor.execute("""
                SELECT
                    memory_type,
                    memory_key,
                    memory_value

                FROM memories

                ORDER BY updated_at DESC

                LIMIT ?
            """, (limit,))

            rows = cursor.fetchall()

            db.close()

        return rows

    # =====================================================
    # MEMORY CONTEXT FOR GEMINI
    # =====================================================

    def get_memory_context(self):

        memories = self.get_memories()

        if not memories:

            return (
                "No persistent memories are currently "
                "available."
            )

        lines = [
            "Persistent memories about the user:"
        ]

        for (
            memory_type,
            key,
            value
        ) in memories:

            lines.append(
                f"- {key}: {value}"
            )

        return "\n".join(lines)

    # =====================================================
    # SESSION MESSAGES
    # =====================================================

    def get_session_messages(
        self,
        session_id
    ):

        with self.lock:

            db = self._connect()
            cursor = db.cursor()

            cursor.execute("""
                SELECT
                    role,
                    content

                FROM messages

                WHERE session_id = ?

                ORDER BY id ASC
            """, (session_id,))

            rows = cursor.fetchall()

            db.close()

        return rows

    # =====================================================
    # END SESSION
    # =====================================================

    def end_session(
        self,
        session_id,
        summary=""
    ):

        if not session_id:
            return

        now = datetime.now().isoformat()

        with self.lock:

            db = self._connect()
            cursor = db.cursor()

            cursor.execute("""
                UPDATE sessions

                SET ended_at = ?,
                    summary = ?

                WHERE id = ?
            """, (
                now,
                summary,
                session_id
            ))

            db.commit()
            db.close()

    # =====================================================
    # DEBUG
    # =====================================================

    def print_memories(self):

        memories = self.get_memories()

        print("\n========== ROBOT MEMORY ==========")

        if not memories:

            print("No memories stored.")

        for (
            memory_type,
            key,
            value
        ) in memories:

            print(
                f"[{memory_type}] "
                f"{key} = {value}"
            )

        print("==================================\n")