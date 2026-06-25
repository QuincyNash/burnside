#!/usr/bin/env python3

import sqlite3

DB_FILE = "tom_cache.sqlite"

conn = sqlite3.connect(DB_FILE)

# Add column if it doesn't already exist
cols = [row[1] for row in conn.execute("PRAGMA table_info(marks)")]

if "closed" not in cols:
    conn.execute("ALTER TABLE marks ADD COLUMN closed INTEGER NOT NULL DEFAULT 0")
    print("Added closed column.")
else:
    print("closed column already exists.")

# Ensure all existing rows are False
conn.execute("UPDATE marks SET closed = 0")
print("Initialized all rows to False.")

conn.commit()
conn.close()
