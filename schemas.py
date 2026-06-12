"""
=====================================================
Pydantic Schemas (Request/Response Models)
=====================================================
Defines data validation schemas for API endpoints.
These ensure type safety and clean API documentation.
=====================================================
"""

from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from datetime import datetime


# =====================================================
# Authentication Schemas
# =====================================================
class UserRegister(BaseModel):
    """Schema for user registration."""
    username: str = Field(..., min_length=3, max_length=100, description="Unique username")
    email: str = Field(..., description="User email address")
    password: str = Field(..., min_length=6, description="Password (min 6 chars)")
    full_name: Optional[str] = Field(None, description="Full name")
    role: str = Field(default="hospital", description="User role: admin or hospital")
    hospital_id: Optional[int] = Field(None, description="Associated hospital ID")


class UserLogin(BaseModel):
    """Schema for user login."""
    username: str = Field(..., description="Username")
    password: str = Field(..., description="Password")


class TokenResponse(BaseModel):
    """Schema for JWT token response."""
    access_token: str
    token_type: str = "bearer"
    user: dict


class UserResponse(BaseModel):
    """Schema for user info response."""
    id: int
    username: str
    email: str
    full_name: Optional[str]
    role: str
    hospital_id: Optional[int]
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


# =====================================================
# Hospital Schemas
# =====================================================
class HospitalCreate(BaseModel):
    """Schema for creating a new hospital."""
    name: str = Field(..., min_length=2, max_length=200, description="Hospital name")
    location: Optional[str] = Field(None, description="Hospital location")
    description: Optional[str] = Field(None, description="Hospital description")
    dataset_type: Optional[str] = Field(None, description="Dataset type: heart, diabetes, kidney")


class HospitalResponse(BaseModel):
    """Schema for hospital info response."""
    id: int
    name: str
    location: Optional[str]
    description: Optional[str]
    dataset_type: Optional[str]
    num_samples: int
    is_active: bool
    is_training: bool
    local_accuracy: float
    local_loss: float
    last_trained: Optional[datetime]
    created_at: datetime

    class Config:
        from_attributes = True


class HospitalUpdate(BaseModel):
    """Schema for updating hospital info."""
    name: Optional[str] = None
    location: Optional[str] = None
    description: Optional[str] = None
    dataset_type: Optional[str] = None
    is_active: Optional[bool] = None


# =====================================================
# Training Schemas
# =====================================================
class TrainingConfig(BaseModel):
    """Schema for training configuration."""
    dataset_type: str = Field(..., description="Dataset: heart, diabetes, or kidney")
    epochs: int = Field(default=5, ge=1, le=50, description="Number of local epochs")
    learning_rate: float = Field(default=0.001, gt=0, description="Learning rate")
    batch_size: int = Field(default=32, ge=8, le=256, description="Batch size")


class FederatedConfig(BaseModel):
    """Schema for federated learning configuration."""
    num_rounds: int = Field(default=10, ge=1, le=100, description="Number of federated rounds")
    local_epochs: int = Field(default=5, ge=1, le=50, description="Local epochs per round")
    learning_rate: float = Field(default=0.001, gt=0, description="Learning rate")
    batch_size: int = Field(default=32, ge=8, le=256, description="Batch size")
    privacy_budget: float = Field(default=1.0, ge=0.1, description="Differential privacy epsilon")
    aggregation_method: str = Field(default="FedAvg", description="Aggregation method")
    hospital_ids: Optional[List[int]] = Field(None, description="Specific hospitals to include")


class TrainingRoundResponse(BaseModel):
    """Schema for training round info."""
    id: int
    round_number: int
    status: str
    global_accuracy: float
    global_loss: float
    participating_hospitals: int
    aggregation_method: str
    privacy_budget: float
    communication_cost: float
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    created_at: datetime

    class Config:
        from_attributes = True


# =====================================================
# Metrics Schemas
# =====================================================
class MetricsResponse(BaseModel):
    """Schema for model metrics response."""
    id: int
    hospital_id: int
    round_number: int
    accuracy: float
    loss: float
    precision_score: float
    recall_score: float
    f1_score: float
    auc_score: float
    dataset_type: Optional[str]
    num_samples_used: int
    training_time: float
    created_at: datetime

    class Config:
        from_attributes = True


# =====================================================
# Prediction Schemas
# =====================================================
class PredictionRequest(BaseModel):
    """Schema for making a risk prediction."""
    prediction_type: str = Field(..., description="Type: heart, diabetes, or kidney")
    features: dict = Field(..., description="Patient features as key-value pairs")


class PredictionResponse(BaseModel):
    """Schema for prediction result."""
    prediction_type: str
    risk_score: float
    risk_level: str  # low, medium, high
    confidence: float
    features_used: dict
    model_version: str


# =====================================================
# Dashboard Schemas
# =====================================================
class DashboardStats(BaseModel):
    """Schema for admin dashboard statistics."""
    total_hospitals: int
    active_hospitals: int
    total_rounds: int
    completed_rounds: int
    avg_global_accuracy: float
    best_global_accuracy: float
    total_predictions: int
    total_samples: int


class HospitalStats(BaseModel):
    """Schema for hospital-specific statistics."""
    hospital_id: int
    hospital_name: str
    local_accuracy: float
    local_loss: float
    total_rounds_participated: int
    num_samples: int
    dataset_type: Optional[str]
    latest_metrics: Optional[MetricsResponse]
