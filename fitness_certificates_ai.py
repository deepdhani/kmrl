"""
Fitness Certificates DB layer (SQLAlchemy + CSV upsert) — robust & Flask-ready.

Default DB: SQLite at ./data/certificates.db
Override with: CERT_DB_URL="postgresql+psycopg2://user:pass@host/dbname"

Public API:
  init_db(db_url=None) -> Engine
  upsert_from_csv(path) -> dict
  expiring_within(days=7, path=None, df=None) -> list[dict]
  total_active(path=None, df=None) -> dict
  list_certificates(limit=200, offset=0) -> dict
  add_certificate(record: dict) -> dict
  update_certificate(cert_id: int, patch: dict) -> dict
  delete_certificate(cert_id: int) -> dict
"""

from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Iterable, Optional

import pandas as pd
import pytz
from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    String,
    UniqueConstraint,
    create_engine,
    func,
    select,
)
from sqlalchemy.orm import declarative_base, sessionmaker

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
IST = pytz.timezone("Asia/Kolkata")
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

def _default_db_url() -> str:
    return f"sqlite:///{(DATA_DIR / 'certificates.db').as_posix()}"

DB_URL = os.getenv("CERT_DB_URL", _default_db_url())
engine = create_engine(DB_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)
Base = declarative_base()

# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------
class Certificate(Base):
    __tablename__ = "certificates"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trainset_id = Column(String(32), index=True, nullable=False)
    dept = Column(String(8), index=True, nullable=False)  # RS / SIG / TEL
    status = Column(String(16), nullable=True)            # valid / active / ok / ...
    valid_from = Column(DateTime(timezone=True), nullable=True)
    valid_to   = Column(DateTime(timezone=True), nullable=False)
    source = Column(String(64), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())

    # avoid exact dupes but allow history across different expiry dates
    __table_args__ = (UniqueConstraint("trainset_id","dept","valid_to", name="uq_cert_train_dept_to"),)

# ---------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------
def init_db(db_url: Optional[str] = None):
    """Create tables if not exists. Rebinds engine/session if db_url is given."""
    global engine, SessionLocal
    if db_url:
        engine = create_engine(db_url, echo=False, future=True)
        SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    Base.metadata.create_all(engine)
    return engine

# Initialize on import
init_db(DB_URL)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
_VALID_MAP = {
    "valid":"valid","active":"valid","ok":"valid","cleared":"valid","1":"valid","true":"valid","v":"valid"
}

def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str).str.strip().str.lower()
        .str.replace(" ","_").str.replace("-","_").str.replace("/","_")
    )
    return df

def _parse_date_series(s: pd.Series) -> pd.Series:
    # robust parse for CSV columns
    for kw in (
        dict(errors="coerce"),
        dict(errors="coerce", dayfirst=True),
        dict(errors="coerce", format="%d-%m-%Y"),
        dict(errors="coerce", format="%Y-%m-%d"),
        dict(errors="coerce", format="%d/%m/%Y"),
        dict(errors="coerce", format="%m/%d/%Y"),
    ):
        out = pd.to_datetime(s, **kw)
        if out.notna().any():
            return out
    return pd.to_datetime(s, errors="coerce")

def _to_dt(value):
    """Return tz-aware IST datetime or None.
    Accepts: YYYY-MM-DD, DD/MM/YYYY, MM/DD/YYYY, DD-MM-YYYY, and datetime.
    """
    if value in (None, "", "null", "NaT"):
        return None
    s = str(value).strip()

    # explicit formats first
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y"):
        try:
            dt = datetime.strptime(s, fmt)
            return IST.localize(dt)
        except Exception:
            continue

    # already datetime (e.g., DB)
    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is None:
            dt = IST.localize(dt)
        return dt

    # fallback
    ts = pd.to_datetime(s, errors="coerce", dayfirst=True)
    if pd.isna(ts):
        return None
    dt = ts.to_pydatetime()
    if dt.tzinfo is None:
        dt = IST.localize(dt)
    return dt

def _dept_norm(x: str) -> str:
    if not x: return ""
    x = str(x).upper()
    if x.startswith("ROLLING"): return "RS"
    if x.startswith("SIG") or x.startswith("SIGNAL"): return "SIG"
    if x.startswith("TEL") or x.startswith("TELE"): return "TEL"
    if x in {"RS","SIG","TEL"}: return x
    return x[:8]

def _pick(cols: Iterable[str], *cands: str) -> Optional[str]:
    for c in cands:
        if c in cols: return c
    return None

def _read_csv_flex(path: str) -> pd.DataFrame:
    for enc in ("utf-8","utf-8-sig","cp1252"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path)

# ---------------------------------------------------------------------
# CSV → DB Upsert (idempotent)
# ---------------------------------------------------------------------
def upsert_from_csv(path: str) -> dict:
    init_db()
    p = Path(path)
    if not p.exists():
        return {"imported":0, "skipped":0, "error":f"file not found: {path}"}

    raw = _norm_cols(_read_csv_flex(path))
    if raw.empty:
        return {"imported":0, "skipped":0, "error":"csv empty"}

    train = _pick(raw.columns, "trainset_id","train_id","trainset","rake_id","code")
    dept  = _pick(raw.columns, "dept","department","division")
    vfrom = _pick(raw.columns, "valid_from","issue_date","start")
    vto   = _pick(raw.columns, "valid_to","valid_till","expiry","expires_on","to","valid_upto")
    stat  = _pick(raw.columns, "status","state","certificate_status")
    if not all([train, dept, vto]):
        return {"imported":0, "skipped":0, "error":"required columns missing"}

    # parse columns
    if vfrom: raw[vfrom] = _parse_date_series(raw[vfrom])
    raw[vto] = _parse_date_series(raw[vto])
    if stat: raw[stat] = raw[stat].astype(str).str.lower().map(_VALID_MAP).fillna(raw[stat])
    raw[dept] = raw[dept].map(_dept_norm)

    imported, skipped = 0, 0
    with SessionLocal() as db:
        for _, r in raw.iterrows():
            vt = _to_dt(r[vto])
            if vt is None:
                skipped += 1
                continue

            rec = Certificate(
                trainset_id=str(r[train]).strip(),
                dept=str(r[dept]).strip(),
                status=(str(r[stat]).lower().strip() if stat in raw.columns else None),
                valid_from=_to_dt(r[vfrom]) if vfrom in raw.columns else None,
                valid_to=vt,
                source="csv-import"
            )

            exists = db.execute(
                select(Certificate.id).where(
                    Certificate.trainset_id == rec.trainset_id,
                    Certificate.dept == rec.dept,
                    Certificate.valid_to == rec.valid_to
                )
            ).first()
            if exists:
                skipped += 1
                continue

            db.add(rec)
            imported += 1

        db.commit()

    return {"imported": imported, "skipped": skipped, "detail": "ok"}

# ---------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------
def _now() -> datetime:
    return datetime.now(tz=IST)

def _valid_status_clause(status: Optional[str]) -> bool:
    if status is None: return True
    return str(status).lower() in _VALID_MAP

def expiring_within(days: int = 7, path: str | None = None, df=None) -> list[dict]:
    """
    Returns list of dicts: {train_id, department, expiry_date, days_to_expiry}
    Uses calendar-day math so dates like 'tomorrow' are always counted as 1 day left.
    """
    init_db()
    # Optional idempotent upsert before reading
    if path:
        upsert_from_csv(path)
    elif df is not None and isinstance(df, pd.DataFrame):
        tmp = DATA_DIR / "_tmp_import.csv"
        df.to_csv(tmp, index=False)
        upsert_from_csv(str(tmp))
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass

    now = _now()
    today = now.date()
    results: list[dict] = []

    with SessionLocal() as db:
        # latest per (trainset_id, dept)
        subq = (
            select(
                Certificate.trainset_id,
                Certificate.dept,
                func.max(Certificate.valid_to).label("max_to"),
            )
            .group_by(Certificate.trainset_id, Certificate.dept)
            .subquery()
        )

        q = (
            select(Certificate.trainset_id, Certificate.dept, Certificate.valid_to, Certificate.status)
            .join(
                subq,
                (Certificate.trainset_id == subq.c.trainset_id)
                & (Certificate.dept == subq.c.dept)
                & (Certificate.valid_to == subq.c.max_to),
            )
        )

        for t_id, dpt, v_to, st in db.execute(q):
            v_to_dt = _to_dt(v_to)
            if v_to_dt is None:
                continue

            # Latest cert must be valid now (status allows or future-dated)
            if not (_valid_status_clause(st) or (st is None and v_to_dt >= now)):
                continue

            # Calendar-day delta avoids time-of-day floor/ceil traps
            delta_days = (v_to_dt.date() - today).days

            # Include anything expiring today .. within 'days'
            if 0 <= delta_days <= days:
                results.append(
                    {
                        "train_id": t_id,
                        "department": dpt,
                        # Return as DD/MM/YYYY to match your UI
                        "expiry_date": v_to_dt.astimezone(IST).strftime("%d/%m/%Y"),
                        "days_to_expiry": int(delta_days),
                    }
                )

    # Sort for a stable table
    results.sort(key=lambda r: (r["days_to_expiry"], r["train_id"], r["department"]))
    return results

def total_active(path: str | None = None, df=None) -> dict:
    """Counts certificates whose latest record per (train,dept) is currently valid."""
    init_db()
    if path:
        upsert_from_csv(path)
    elif df is not None and isinstance(df, pd.DataFrame):
        tmp = DATA_DIR / "_tmp_import.csv"
        df.to_csv(tmp, index=False)
        upsert_from_csv(str(tmp))
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass

    now = _now()
    with SessionLocal() as db:
        subq = (
            select(
                Certificate.trainset_id,
                Certificate.dept,
                func.max(Certificate.valid_to).label("max_to"),
            )
            .group_by(Certificate.trainset_id, Certificate.dept)
            .subquery()
        )
        q = (
            select(Certificate.status, Certificate.valid_to)
            .join(
                subq,
                (Certificate.trainset_id == subq.c.trainset_id)
                & (Certificate.dept == subq.c.dept)
                & (Certificate.valid_to == subq.c.max_to),
            )
        )
        count = 0
        for st, vt in db.execute(q):
            vt = _to_dt(vt)
            if vt is None:
                continue
            if (_valid_status_clause(st) or st is None) and vt >= now:
                count += 1
    return {"count": count}

# ---------------------------------------------------------------------
# Listing + CRUD (for Admin UI)
# ---------------------------------------------------------------------
def list_certificates(limit: int = 200, offset: int = 0) -> dict:
    init_db()
    with SessionLocal() as db:
        q = (
            select(Certificate)
            .order_by(Certificate.trainset_id, Certificate.dept, Certificate.valid_to.desc())
            .offset(offset).limit(limit)
        )
        rows = []
        for c in db.execute(q).scalars():
            rows.append({
                "id": c.id,
                "trainset_id": c.trainset_id,
                "department": c.dept,
                "status": c.status,
                # keep ISO here for admin table date inputs
                "valid_from": c.valid_from.isoformat() if c.valid_from else None,
                "valid_to": c.valid_to.isoformat() if c.valid_to else None,
                "source": c.source
            })
        return {"rows": rows, "limit": limit, "offset": offset, "count": len(rows)}

def add_certificate(record: dict) -> dict:
    init_db()
    trainset_id = str(record.get("trainset_id") or "").strip()
    dept = _dept_norm(record.get("dept") or "")
    valid_to = _to_dt(record.get("valid_to"))
    if not trainset_id or not dept or valid_to is None:
        return {"error": "trainset_id, dept, valid_to are required. Use YYYY-MM-DD / DD/MM/YYYY / MM/DD/YYYY."}

    c = Certificate(
        trainset_id=trainset_id,
        dept=dept,
        status=(str(record.get("status")).lower() if record.get("status") else None),
        valid_from=_to_dt(record.get("valid_from")),
        valid_to=valid_to,
        source=record.get("source") or "api"
    )
    with SessionLocal() as db:
        db.add(c); db.commit(); db.refresh(c)
        return {"id": c.id}

def update_certificate(cert_id: int, patch: dict) -> dict:
    init_db()
    with SessionLocal() as db:
        c = db.get(Certificate, cert_id)
        if not c:
            return {"updated": 0, "error": "not found"}

        if "trainset_id" in patch: c.trainset_id = str(patch["trainset_id"]).strip()
        if "dept" in patch: c.dept = _dept_norm(patch["dept"])
        if "status" in patch: c.status = (str(patch["status"]).lower() or None)

        if "valid_from" in patch:
            c.valid_from = _to_dt(patch["valid_from"])
        if "valid_to" in patch:
            vt = _to_dt(patch["valid_to"])
            if vt is None:
                return {"updated": 0, "error": "valid_to must be a valid date"}
            c.valid_to = vt

        db.commit()
        return {"updated": 1}

def delete_certificate(cert_id: int) -> dict:
    init_db()
    with SessionLocal() as db:
        c = db.get(Certificate, cert_id)
        if not c: return {"deleted": 0, "error": "not found"}
        db.delete(c); db.commit()
        return {"deleted": 1}