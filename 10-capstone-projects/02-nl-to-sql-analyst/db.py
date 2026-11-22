"""
Capstone 02: SQLite Database Initialization & Seed Data.
"""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "analytics.db"

def init_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS orders (
        order_id INTEGER PRIMARY KEY,
        customer_name TEXT,
        category TEXT,
        amount REAL,
        status TEXT,
        created_at TEXT
    );
    """)

    cursor.execute("DELETE FROM orders;")
    sample_data = [
        (1, "Acme Corp", "Electronics", 1250.00, "COMPLETED", "2026-11-01"),
        (2, "Stark Industries", "Cloud", 8400.50, "COMPLETED", "2026-11-03"),
        (3, "Wayne Enterprise", "Security", 3200.00, "PENDING", "2026-11-05"),
        (4, "Cyberdyne", "Electronics", 450.00, "COMPLETED", "2026-11-10"),
        (5, "Initech", "Software", 950.00, "REFUNDED", "2026-11-12")
    ]
    cursor.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?);", sample_data)
    conn.commit()
    conn.close()
    print(f"Database initialized at {DB_PATH}")

def execute_query(sql: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute(sql)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        conn.close()
        return {"columns": columns, "rows": rows, "error": None}
    except Exception as e:
        conn.close()
        return {"columns": [], "rows": [], "error": str(e)}

if __name__ == "__main__":
    init_database()
