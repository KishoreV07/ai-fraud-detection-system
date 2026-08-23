"""
database.py
-------------
SQLite database setup for storing prediction history.
Uses Python's built-in sqlite3 — no extra install needed.
"""

import sqlite3
import os
from datetime import datetime

DB_PATH = "data/fraud_detection.db"


def init_db():
    """Create the database and predictions table if they don't exist."""
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            amount REAL NOT NULL,
            time_val REAL NOT NULL,
            fraud_probability REAL NOT NULL,
            is_fraud INTEGER NOT NULL,
            risk_level TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def save_prediction(amount: float, time_val: float, fraud_probability: float,
                     is_fraud: bool, risk_level: str):
    """Insert a new prediction record."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions (timestamp, amount, time_val, fraud_probability, is_fraud, risk_level)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        amount, time_val, fraud_probability, int(is_fraud), risk_level
    ))
    conn.commit()
    conn.close()


def get_history(limit: int = 100):
    """Fetch the most recent predictions."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM predictions ORDER BY timestamp DESC LIMIT ?
    """, (limit,))
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows