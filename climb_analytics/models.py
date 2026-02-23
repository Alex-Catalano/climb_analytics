from sqlalchemy import Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import relationship
from database import Base


class Session(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, index=True)

    date = Column(String, nullable=False)
    location = Column(String, nullable=True)
    sleep_hours = Column(Float, nullable=True)

    start_ts = Column(Float, nullable=False)
    warmup_end_ts = Column(Float, nullable=True)
    end_ts = Column(Float, nullable=True)

    attempts = relationship("Attempt", back_populates="session", cascade="all, delete")


class Attempt(Base):
    __tablename__ = "attempts"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)

    timestamp = Column(Float, nullable=False)
    session_elapsed_seconds = Column(Float, nullable=False)
    rest_seconds = Column(Float, nullable=True)

    phase = Column(String, nullable=False)  # "warmup" or "main"

    problem_name = Column(String, nullable=False)
    grade = Column(String, nullable=True)
    board_angle = Column(Integer, nullable=True)

    attempt_number_on_problem = Column(Integer, nullable=True)
    outcome = Column(String, nullable=False)  # "flash" / "send" / "fail"

    notes = Column(String, nullable=True)
    video_path = Column(String, nullable=True)

    session = relationship("Session", back_populates="attempts")