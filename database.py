import sqlite3
import json
import numpy as np

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_NAME = os.path.join(BASE_DIR, "federated_audit.db")

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS fl_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            round INTEGER,
            client_id INTEGER,
            train_loss REAL,
            train_acc REAL,
            cosine_similarity REAL,
            kl_divergence REAL,
            shap_values TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS global_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            round INTEGER,
            avg_loss REAL,
            avg_acc REAL,
            mean_cosine_sim REAL,
            mean_kl_div REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def log_client_metrics(round_num, client_id, loss, acc, cos_sim, kl_div, shap_vals):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    shap_json = json.dumps(shap_vals.tolist() if isinstance(shap_vals, np.ndarray) else shap_vals)
    cursor.execute('''
        INSERT INTO fl_metrics (round, client_id, train_loss, train_acc, cosine_similarity, kl_divergence, shap_values)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (round_num, client_id, loss, acc, cos_sim, kl_div, shap_json))
    conn.commit()
    conn.close()

def log_global_metrics(round_num, avg_loss, avg_acc, mean_cos, mean_kl):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO global_metrics (round, avg_loss, avg_acc, mean_cosine_sim, mean_kl_div)
        VALUES (?, ?, ?, ?, ?)
    ''', (round_num, avg_loss, avg_acc, mean_cos, mean_kl))
    conn.commit()
    conn.close()
