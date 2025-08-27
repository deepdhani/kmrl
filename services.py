import pandas as pd
from pathlib import Path

# Import from DB-backed fitness_certificates_ai.py
from fitness_certificates_ai import (
    expiring_within as db_expiring_within,
    total_active as db_total_active,
    upsert_from_csv as db_upsert,
    list_certificates as db_list,
    add_certificate as db_add,
    update_certificate as db_update,
    delete_certificate as db_delete,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# --------- CSV helpers ---------
def _read_csv(path):
    for enc in ["utf-8", "utf-8-sig", "cp1252"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path)

def _norm_cols(df):
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
        .str.replace("/", "_")
    )
    return df

# --------- Jobcards (CSV-backed) ---------
def load_jobcards():
    df = _norm_cols(_read_csv(DATA_DIR / "jobcards.csv"))
    train = next((c for c in ["trainset_id","train_id","train","rake_id","code"] if c in df.columns), None)
    status = next((c for c in ["status","state"] if c in df.columns), None)
    severity = next((c for c in ["severity","priority","sev"] if c in df.columns), None)

    if status is None:
        df["status"] = "open"; status = "status"
    if severity is None:
        df["severity"] = "C"; severity = "severity"

    df[status] = df[status].astype(str).str.lower()
    df[severity] = df[severity].astype(str).str.upper().str[0]
    return df, train, status, severity

def job_status_summary(train_id:int):
    df, train, status, severity = load_jobcards()
    if train is None:
        return {"train_id":train_id,"total_jobs":0,"open_jobs":0,"closed_jobs":0,"critical_alert":"NO"}
    sub = df[df[train]==int(train_id)]
    total = int(sub.shape[0])
    open_jobs = int((sub[status]=="open").sum())
    closed = total - open_jobs
    critical = "YES" if ((sub[status]=="open") & (sub[severity]=="A")).any() else "NO"
    return {"train_id":int(train_id),"total_jobs":total,"open_jobs":open_jobs,"closed_jobs":closed,"critical_alert":critical}

# --------- Fitness (DB-backed) ---------
def expiring_within(days:int=7):
    return db_expiring_within(days)

def total_active():
    return db_total_active()

# --------- ML features for predicted failures ---------
def load_ml():
    df = _norm_cols(_read_csv(DATA_DIR / "ml_features.csv"))
    return df

def predicted_failures_count():
    ml = load_ml()
    if ml.empty:
        return {"count":0}
    pred_col = next((c for c in ["predicted_failure","prediction","pred"] if c in ml.columns), None)
    prob_col = next((c for c in ["failure_prob","prob_failure","risk_prob"] if c in ml.columns), None)
    if pred_col:
        return {"count": int(ml[pred_col].astype(str).str.lower().isin(["1","true","yes"]).sum())}
    if prob_col:
        return {"count": int((ml[prob_col].astype(float) >= 0.5).sum())}
    return {"count":0}

def predict_failure():
    ml = load_ml()
    if ml.empty:
        return {"prediction":"LOW","reason":"no ml_features rows"}
    prob_col = next((c for c in ["failure_prob","prob_failure","risk_prob"] if c in ml.columns), None)
    pred_col = next((c for c in ["predicted_failure","prediction","pred"] if c in ml.columns), None)
    if prob_col:
        mean = float(ml[prob_col].astype(float).mean())
        label = "HIGH" if mean>=0.5 else ("MEDIUM" if mean>=0.25 else "LOW")
        return {"prediction":label,"mean_failure_prob":round(mean,3)}
    if pred_col:
        rate = float((ml[pred_col].astype(str).str.lower().isin(["1","true","yes"]).mean()))
        label = "HIGH" if rate>=0.5 else ("MEDIUM" if rate>=0.25 else "LOW")
        return {"prediction":label,"positive_rate":round(rate,3)}
    return {"prediction":"LOW"}

# --------- Admin pass-through for DB ---------
def certs_import_from_csv():
    return db_upsert(str(DATA_DIR / "fitness_certificates_history.csv"))

def certs_list(limit:int=200, offset:int=0):
    return db_list(limit, offset)

def certs_add(record:dict):
    return db_add(record)

def certs_update(cert_id:int, patch:dict):
    return db_update(cert_id, patch)

def certs_delete(cert_id:int):
    return db_delete(cert_id)