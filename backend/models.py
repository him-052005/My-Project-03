
"""
SQLite + SQLAlchemy models
"""
from __future__ import annotations
import os
from datetime import date, datetime
from sqlalchemy import (
    create_engine, Column, Integer, String, Date, DateTime, ForeignKey, UniqueConstraint
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "attendance.db")
engine = create_engine(f"sqlite:///{DB_PATH}", echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, future=True)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String(64), unique=True, nullable=False)
    password_hash = Column(String(256), nullable=False)

class Student(Base):
    __tablename__ = "students"
    id = Column(Integer, primary_key=True)
    roll_no = Column(String(32), unique=True, nullable=False)
    name = Column(String(128), nullable=False)
    email = Column(String(128))
    klass = Column(String(64))
    section = Column(String(16))
    image_path = Column(String(256))  # representative image used in UI

    attendance = relationship("Attendance", back_populates="student", cascade="all, delete-orphan")

class Attendance(Base):
    __tablename__ = "attendance"
    id = Column(Integer, primary_key=True)
    student_id = Column(Integer, ForeignKey("students.id"), nullable=False)
    day = Column(Date, nullable=False)
    status = Column(String(16), nullable=False, default="Absent")
    timestamp = Column(DateTime)  # when marked present

    student = relationship("Student", back_populates="attendance")
    __table_args__ = (UniqueConstraint('student_id', 'day', name='uq_student_day'),)


def init_db():
    Base.metadata.create_all(engine)


def ensure_day_rows(session, the_day: date):
    """Ensure each student has an Attendance row (default Absent) for the_day."""
    students = session.query(Student).all()
    for s in students:
        exists = session.query(Attendance).filter_by(student_id=s.id, day=the_day).one_or_none()
        if not exists:
            session.add(Attendance(student_id=s.id, day=the_day, status="Absent"))
    session.commit()