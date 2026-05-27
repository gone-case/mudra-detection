import os
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime

Base = declarative_base()

class Mudra(Base):
    __tablename__ = 'mudras'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(50), unique=True, nullable=False)
    meaning = Column(String(200), nullable=False)
    description = Column(Text, nullable=False)
    hand_formation = Column(Text, nullable=False)
    usage = Column(Text, nullable=False)
    common_mistakes = Column(Text, nullable=False)
    improvement_tips = Column(Text, nullable=False)
    difficulty = Column(String(20), nullable=False) # Easy, Medium, Hard
    
    # Relationship to progress
    progress_records = relationship("UserProgress", back_populates="mudra")

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    progress = relationship("UserProgress", back_populates="user")
    sessions = relationship("PracticeSession", back_populates="user")

class UserProgress(Base):
    __tablename__ = 'user_progress'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    mudra_id = Column(Integer, ForeignKey('mudras.id'))
    
    total_attempts = Column(Integer, default=0)
    correct_attempts = Column(Integer, default=0)
    avg_confidence = Column(Float, default=0.0)
    
    user = relationship("User", back_populates="progress")
    mudra = relationship("Mudra", back_populates="progress_records")

class PracticeSession(Base):
    __tablename__ = 'practice_sessions'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    start_time = Column(DateTime, default=datetime.utcnow)
    duration_minutes = Column(Float, default=0.0)
    total_mudras_practiced = Column(Integer, default=0)
    overall_accuracy = Column(Float, default=0.0)
    
    user = relationship("User", back_populates="sessions")

# SQLite Database connection
db_path = os.path.join(os.path.dirname(__file__), 'mudra_practice.db')
engine = create_engine(f'sqlite:///{db_path}', connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

def init_db():
    Base.metadata.create_all(bind=engine)

if __name__ == "__main__":
    init_db()
    print("Database initialized.")
