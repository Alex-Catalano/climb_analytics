import time
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import engine, SessionLocal
import models

models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Climb Analytics API")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class StartSessionIn(BaseModel):
    date: str
    location: Optional[str] = None
    sleep_hours: Optional[float] = None


class AttemptIn(BaseModel):
    problem_name: str
    outcome: str  # fail / send / flash
    grade: Optional[str] = None
    board_angle: Optional[int] = None
    attempt_number_on_problem: Optional[int] = None
    notes: Optional[str] = None
    video_path: Optional[str] = None


@app.get("/", response_class=HTMLResponse)
def ui():
    return HTMLResponse("<h2>Server running</h2><p>Go to <a href='/docs'>/docs</a></p>")


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/session/start")
def session_start(payload: StartSessionIn, db: Session = Depends(get_db)):
    s = models.Session(
        date=payload.date,
        location=payload.location,
        sleep_hours=payload.sleep_hours,
        start_ts=time.time(),
        warmup_end_ts=None,
        end_ts=None,
    )
    db.add(s)
    db.commit()
    db.refresh(s)
    return {"session_id": s.id, "start_ts": s.start_ts}


@app.post("/session/{session_id}/warmup_end")
def session_warmup_end(session_id: int, db: Session = Depends(get_db)):
    s = db.query(models.Session).filter(models.Session.id == session_id).first()
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")

    if s.warmup_end_ts is None:
        s.warmup_end_ts = time.time()
        db.commit()
        db.refresh(s)

    return {"session_id": s.id, "warmup_end_ts": s.warmup_end_ts}


@app.post("/session/{session_id}/attempt")
def add_attempt(session_id: int, payload: AttemptIn, db: Session = Depends(get_db)):
    s = db.query(models.Session).filter(models.Session.id == session_id).first()
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")

    now = time.time()
    session_elapsed = now - s.start_ts

    last = (
        db.query(models.Attempt)
        .filter(models.Attempt.session_id == session_id)
        .order_by(models.Attempt.timestamp.desc())
        .first()
    )
    rest_seconds = (now - last.timestamp) if last else None

    phase = "warmup" if s.warmup_end_ts is None else "main"

    outcome = payload.outcome.strip().lower()
    if outcome not in ("fail", "send", "flash"):
        raise HTTPException(status_code=400, detail="Outcome must be: fail, send, flash")

    a = models.Attempt(
        session_id=session_id,
        timestamp=now,
        session_elapsed_seconds=session_elapsed,
        rest_seconds=rest_seconds,
        phase=phase,
        problem_name=payload.problem_name,
        grade=payload.grade,
        board_angle=payload.board_angle,
        attempt_number_on_problem=payload.attempt_number_on_problem,
        outcome=outcome,
        notes=payload.notes,
        video_path=payload.video_path,
    )
    db.add(a)
    db.commit()
    db.refresh(a)

    return {
        "attempt_id": a.id,
        "phase": a.phase,
        "session_elapsed_seconds": round(a.session_elapsed_seconds, 2),
        "rest_seconds": round(a.rest_seconds, 2) if a.rest_seconds is not None else None,
    }


@app.post("/session/{session_id}/end")
def session_end(session_id: int, db: Session = Depends(get_db)):
    s = db.query(models.Session).filter(models.Session.id == session_id).first()
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")

    if s.end_ts is None:
        s.end_ts = time.time()
        db.commit()
        db.refresh(s)

    return {"session_id": s.id, "duration_seconds": round(s.end_ts - s.start_ts, 2)}