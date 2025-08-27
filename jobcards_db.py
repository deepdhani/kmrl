# jobcards_db.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Iterable, Optional
from datetime import datetime

import pandas as pd
import pytz
from sqlalchemy import Column, Integer, String, DateTime, Text, create_engine, func, select
from sqlalchemy.orm import declarative_base, sessionmaker

IST = pytz.timezone("Asia/Kolkata")
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

def _default_db_url():
    return f"sqlite:///{(DATA_DIR / 'jobcards.db').as_posix()}"

DB_URL = os.getenv("JOBCARDS_DB_URL", _default_db_url())
engine = create_engine(DB_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)
Base = declarative_base()

class JobCard(Base):
    __tablename__ = "jobcards"
    id = Column(Integer, primary_key=True, autoincrement=True)
    trainset_id = Column(String(32), index=True, nullable=False)
    status = Column(String(16), index=True, nullable=False)   # open/closed
    severity = Column(String(4), index=True, nullable=True)   # A/B/C
    title = Column(String(120), nullable=True)
    description = Column(Text, nullable=True)
    opened_at = Column(DateTime(timezone=True), nullable=True)
    closed_at = Column(DateTime(timezone=True), nullable=True)
    source = Column(String(64), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())

def init_db(db_url: Optional[str]=None):
    global engine, SessionLocal
    if db_url:
        engine = create_engine(db_url, echo=False, future=True)
        SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    Base.metadata.create_all(engine)
    return engine

init_db(DB_URL)

def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (df.columns.astype(str).str.strip().str.lower()
                  .str.replace(" ","_").str.replace("-","_").str.replace("/","_"))
    return df

def _read_csv_flex(path:str) -> pd.DataFrame:
    for enc in ("utf-8","utf-8-sig","cp1252"):
        try: return pd.read_csv(path, encoding=enc)
        except Exception: continue
    return pd.read_csv(path)

def _to_dt(value):
    if value in (None, "", "null", "NaT"): return None
    s = str(value).strip()
    for fmt in ("%Y-%m-%d","%d/%m/%Y","%m/%d/%Y","%d-%m-%Y","%Y-%m-%d %H:%M","%d/%m/%Y %H:%M"):
        try:
            d = datetime.strptime(s, fmt); 
            import pytz; return pytz.timezone("Asia/Kolkata").localize(d)
        except Exception:
            pass
    ts = pd.to_datetime(s, errors="coerce", dayfirst=True)
    if pd.isna(ts): return None
    d = ts.to_pydatetime()
    if d.tzinfo is None: d = IST.localize(d)
    return d

def _pick(cols: Iterable[str], *cands: str):
    for c in cands:
        if c in cols: return c
    return None

def upsert_from_csv(path:str) -> dict:
    init_db()
    p = Path(path)
    if not p.exists(): return {"imported":0,"skipped":0,"error":f"file not found: {path}"}
    raw = _norm_cols(_read_csv_flex(path))
    if raw.empty: return {"imported":0,"skipped":0,"error":"csv empty"}

    train = _pick(raw.columns,"trainset_id","train_id","train","rake","rake_id","code")
    stat  = _pick(raw.columns,"status","state")
    sev   = _pick(raw.columns,"severity","priority","sev")
    title = _pick(raw.columns,"title","summary","subject")
    desc  = _pick(raw.columns,"description","details","note","remark")
    open_ = _pick(raw.columns,"opened_at","open_date","created","created_at","start")
    close = _pick(raw.columns,"closed_at","close_date","completed","end","finished_at")
    if not (train and stat): return {"imported":0,"skipped":0,"error":"required columns missing (trainset_id/status)"}

    if sev and sev in raw.columns: raw[sev] = raw[sev].astype(str).str.upper().str[0]
    if stat in raw.columns: raw[stat] = raw[stat].astype(str).str.lower()

    imported=0; skipped=0
    with SessionLocal() as db:
        for _, r in raw.iterrows():
            jc = JobCard(
                trainset_id=str(r[train]).strip(),
                status=str(r[stat]).lower(),
                severity=(str(r[sev]).upper().strip()[0] if sev in raw.columns and pd.notna(r[sev]) else None),
                title=(str(r[title]).strip() if title and pd.notna(r[title]) else None),
                description=(str(r[desc]).strip() if desc and pd.notna(r[desc]) else None),
                opened_at=_to_dt(r[open_]) if open_ in raw.columns else None,
                closed_at=_to_dt(r[close]) if close in raw.columns else None,
                source="csv-import"
            )
            exists = db.execute(
                select(JobCard.id).where(
                    JobCard.trainset_id==jc.trainset_id,
                    JobCard.title==jc.title,
                    JobCard.opened_at==jc.opened_at
                )
            ).first()
            if exists: skipped+=1; continue
            db.add(jc); imported+=1
        db.commit()
    return {"imported":imported,"skipped":skipped}

def list_jobcards(limit:int=200, offset:int=0) -> dict:
    init_db()
    rows=[]
    with SessionLocal() as db:
        q = (select(JobCard)
             .order_by(JobCard.opened_at.desc().nullslast(), JobCard.id.desc())
             .offset(offset).limit(limit))
        for jc in db.execute(q).scalars():
            rows.append({
                "id": jc.id, "trainset_id": jc.trainset_id, "status": jc.status,
                "severity": jc.severity, "title": jc.title, "description": jc.description,
                "opened_at": jc.opened_at.isoformat() if jc.opened_at else None,
                "closed_at": jc.closed_at.isoformat() if jc.closed_at else None,
                "source": jc.source
            })
    return {"rows": rows, "limit": limit, "offset": offset, "count": len(rows)}

def add_jobcard(record:dict) -> dict:
    init_db()
    train = str(record.get("trainset_id") or "").strip()
    status = (record.get("status") or "open").lower()
    if not train: return {"error":"trainset_id is required"}
    jc = JobCard(
        trainset_id=train, status=status,
        severity=(str(record.get("severity")).upper().strip()[0] if record.get("severity") else None),
        title=(record.get("title") or None), description=(record.get("description") or None),
        opened_at=_to_dt(record.get("opened_at")), closed_at=_to_dt(record.get("closed_at")),
        source=record.get("source") or "api"
    )
    with SessionLocal() as db:
        db.add(jc); db.commit(); db.refresh(jc)
        return {"id": jc.id}

def update_jobcard(jc_id:int, patch:dict) -> dict:
    init_db()
    with SessionLocal() as db:
        jc = db.get(JobCard, jc_id)
        if not jc: return {"updated":0,"error":"not found"}
        if "trainset_id" in patch: jc.trainset_id = str(patch["trainset_id"]).strip()
        if "status" in patch: jc.status = str(patch["status"]).lower()
        if "severity" in patch: jc.severity = (str(patch["severity"]).upper().strip()[0] if patch["severity"] else None)
        if "title" in patch: jc.title = patch["title"] or None
        if "description" in patch: jc.description = patch["description"] or None
        if "opened_at" in patch: jc.opened_at = _to_dt(patch["opened_at"])
        if "closed_at" in patch: jc.closed_at = _to_dt(patch["closed_at"])
        db.commit()
        return {"updated":1}

def delete_jobcard(jc_id:int) -> dict:
    init_db()
    with SessionLocal() as db:
        jc = db.get(JobCard, jc_id)
        if not jc: return {"deleted":0,"error":"not found"}
        db.delete(jc); db.commit()
        return {"deleted":1}

def summary_for_train(train_id:int) -> dict:
    init_db()
    total=open_cnt=closed_cnt=0; critical=False
    with SessionLocal() as db:
        q = select(JobCard.status, JobCard.severity).where(JobCard.trainset_id==str(train_id))
        for st, sev in db.execute(q):
            total += 1
            if str(st).lower()=="open":
                open_cnt += 1
                if (sev or "").upper().startswith("A"): critical=True
            else:
                closed_cnt += 1
    return {"train_id": int(train_id), "total_jobs": total, "open_jobs": open_cnt,
            "closed_jobs": closed_cnt, "critical_alert": "YES" if critical else "NO"}