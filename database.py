"""
=====================================================
Database Configuration & Models
=====================================================
Defines SQLAlchemy models for the federated learning system.
Tables: Users, Hospitals, TrainingRounds, ModelMetrics
=====================================================
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os

# Database URL from environment or default
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./federated_learning.db")

# Create SQLAlchemy engine
engine = create_engine(
    DATABASE_URL, 
    connect_args={"check_same_thread": False}  # Needed for SQLite
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for all models
Base = declarative_base()


# =====================================================
# User Model - Stores admin and hospital user accounts
# =====================================================
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    username = Column(String(100), unique=True, index=True, nullable=False)
    email = Column(String(200), unique=True, index=True, nullable=False)
    hashed_password = Column(String(300), nullable=False)
    full_name = Column(String(200), nullable=True)
    role = Column(String(50), default="hospital")  # "admin" or "hospital"
    hospital_id = Column(Integer, ForeignKey("hospitals.id"), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship to hospital
    hospital = relationship("Hospital", back_populates="users")


# =====================================================
# Hospital Model - Represents each participating hospital
# =====================================================
class Hospital(Base):
    __tablename__ = "hospitals"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(200), unique=True, nullable=False)
    location = Column(String(200), nullable=True)
    description = Column(Text, nullable=True)
    dataset_type = Column(String(100), nullable=True)  # "heart", "diabetes", "kidney"
    num_samples = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)
    is_training = Column(Boolean, default=False)
    local_accuracy = Column(Float, default=0.0)
    local_loss = Column(Float, default=0.0)
    last_trained = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    users = relationship("User", back_populates="hospital")
    metrics = relationship("ModelMetrics", back_populates="hospital")


# =====================================================
# Training Round Model - Tracks federated learning rounds
# =====================================================
class TrainingRound(Base):
    __tablename__ = "training_rounds"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    round_number = Column(Integer, nullable=False)
    status = Column(String(50), default="pending")  # pending, in_progress, completed
    global_accuracy = Column(Float, default=0.0)
    global_loss = Column(Float, default=0.0)
    participating_hospitals = Column(Integer, default=0)
    aggregation_method = Column(String(50), default="FedAvg")
    privacy_budget = Column(Float, default=1.0)  # Differential privacy epsilon
    communication_cost = Column(Float, default=0.0)  # Simulated communication cost in MB
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


# =====================================================
# Model Metrics - Stores per-hospital per-round metrics
# =====================================================
class ModelMetrics(Base):
    __tablename__ = "model_metrics"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    hospital_id = Column(Integer, ForeignKey("hospitals.id"), nullable=False)
    round_number = Column(Integer, nullable=False)
    accuracy = Column(Float, default=0.0)
    loss = Column(Float, default=0.0)
    precision_score = Column(Float, default=0.0)
    recall_score = Column(Float, default=0.0)
    f1_score = Column(Float, default=0.0)
    auc_score = Column(Float, default=0.0)
    dataset_type = Column(String(100), nullable=True)
    num_samples_used = Column(Integer, default=0)
    training_time = Column(Float, default=0.0)  # In seconds
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship to hospital
    hospital = relationship("Hospital", back_populates="metrics")


# =====================================================
# Prediction Log - Stores patient risk predictions
# =====================================================
class PredictionLog(Base):
    __tablename__ = "prediction_logs"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    hospital_id = Column(Integer, ForeignKey("hospitals.id"), nullable=True)
    prediction_type = Column(String(100), nullable=False)  # heart, diabetes, kidney
    risk_score = Column(Float, default=0.0)
    risk_level = Column(String(50), nullable=True)  # low, medium, high
    input_features = Column(Text, nullable=True)  # JSON string of input features
    model_version = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


# =====================================================
# Create all tables in the database
# =====================================================
def init_db():
    """Initialize the database and create all tables."""
    Base.metadata.create_all(bind=engine)
    print("✅ Database initialized successfully!")


# =====================================================
# Get database session (dependency injection for FastAPI)
# =====================================================
def get_db():
    """
    Provides a database session for each request.
    Automatically closes the session when done.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
