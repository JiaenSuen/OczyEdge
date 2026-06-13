import sqlite3
from config import DB_PATH


DEFAULT_ZERO_SHOT_LABELS = [
    "product",
    "retail product",
    "packaged product",
    "package",
    "box",
    "bottle",
    "drink bottle",
    "can",
    "carton",
    "container",
    "snack package",
    "food package",
    "merchandise",
]


def get_conn():
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = get_conn()
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS products (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        price REAL NOT NULL,
        image_path TEXT NOT NULL
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS zero_shot_labels (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL COLLATE NOCASE UNIQUE,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """)

    c.execute("SELECT COUNT(*) FROM zero_shot_labels")
    label_count = c.fetchone()[0]

    if label_count == 0:
        c.executemany(
            "INSERT OR IGNORE INTO zero_shot_labels (name) VALUES (?)",
            [(label,) for label in DEFAULT_ZERO_SHOT_LABELS],
        )

    conn.commit()
    conn.close()


def insert_product(name, price, image_path):
    conn = get_conn()
    c = conn.cursor()

    c.execute(
        "INSERT INTO products (name, price, image_path) VALUES (?, ?, ?)",
        (name, price, image_path)
    )

    pid = c.lastrowid
    conn.commit()
    conn.close()
    return pid


def get_all_products():
    conn = get_conn()
    c = conn.cursor()

    c.execute("SELECT * FROM products ORDER BY id DESC")
    rows = c.fetchall()

    conn.close()
    return rows


def get_product(pid):
    conn = get_conn()
    c = conn.cursor()

    c.execute("SELECT * FROM products WHERE id = ?", (pid,))
    row = c.fetchone()

    conn.close()
    return row


def update_product(pid, name, price):
    conn = get_conn()
    c = conn.cursor()

    c.execute(
        "UPDATE products SET name = ?, price = ? WHERE id = ?",
        (name, price, pid)
    )

    conn.commit()
    conn.close()


def delete_product(pid):
    conn = get_conn()
    c = conn.cursor()

    c.execute("DELETE FROM products WHERE id = ?", (pid,))
    conn.commit()
    conn.close()


def get_zero_shot_labels():
    conn = get_conn()
    c = conn.cursor()

    c.execute("SELECT id, name FROM zero_shot_labels ORDER BY id ASC")
    rows = c.fetchall()

    conn.close()
    return rows


def insert_zero_shot_label(name):
    label = (name or "").strip()

    if not label:
        raise ValueError("Detection target cannot be empty.")

    conn = get_conn()
    c = conn.cursor()

    created = True

    try:
        c.execute(
            "INSERT INTO zero_shot_labels (name) VALUES (?)",
            (label,),
        )
        label_id = c.lastrowid
        conn.commit()
    except sqlite3.IntegrityError:
        created = False
        c.execute(
            "SELECT id FROM zero_shot_labels WHERE name = ? COLLATE NOCASE",
            (label,),
        )
        row = c.fetchone()
        label_id = row[0] if row else None

    c.execute("SELECT id, name FROM zero_shot_labels WHERE id = ?", (label_id,))
    row = c.fetchone()
    conn.close()

    return row, created


def delete_zero_shot_label(label_id):
    conn = get_conn()
    c = conn.cursor()

    c.execute("DELETE FROM zero_shot_labels WHERE id = ?", (label_id,))
    deleted = c.rowcount > 0

    conn.commit()
    conn.close()
    return deleted
