"""
=====================================================
FastAPI Main Application
=====================================================
Central server for the Federated Learning system.
Handles authentication, hospital management,
federated training orchestration, and predictions.
=====================================================
"""

import os
import sys

# Fix Windows console encoding for unicode
os.environ['PYTHONIOENCODING'] = 'utf-8'
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

import json
import copy
import threading
from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import FastAPI, Depends, HTTPException, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from dotenv import load_dotenv

# Add project root to path for ML imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import init_db, get_db, User, Hospital, TrainingRound, ModelMetrics, PredictionLog
from auth import (get_password_hash, verify_password, create_access_token,
                  get_current_user, get_admin_user)
from schemas import (UserRegister, UserLogin, TokenResponse, UserResponse,
                     HospitalCreate, HospitalResponse, HospitalUpdate,
                     TrainingConfig, FederatedConfig, TrainingRoundResponse,
                     MetricsResponse, PredictionRequest, PredictionResponse,
                     DashboardStats)

# Load environment variables
load_dotenv()

# =====================================================
# Initialize FastAPI App
# =====================================================
app = FastAPI(
    title="Federated Learning Healthcare System",
    description="Cross-Hospital Risk Prediction Without Data Sharing",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# Global State for FL Engine
# =====================================================
fl_server = None
training_status = {
    "is_training": False,
    "current_round": 0,
    "total_rounds": 0,
    "status": "idle",
    "progress": 0
}


# =====================================================
# Startup Event
# =====================================================
@app.on_event("startup")
async def startup():
    """Initialize database and seed data on startup."""
    init_db()
    db = next(get_db())
    
    # Create default admin if not exists
    admin = db.query(User).filter(User.username == "admin").first()
    if not admin:
        admin = User(
            username="admin", email="admin@fedlearn.com",
            hashed_password=get_password_hash("admin123"),
            full_name="System Administrator", role="admin", is_active=True
        )
        db.add(admin)
        db.commit()
        print("[OK] Default admin created (admin/admin123)")
    
    # Create default hospitals if none exist
    hospital_count = db.query(Hospital).count()
    if hospital_count == 0:
        hospitals = [
            Hospital(name="City General Hospital", location="New York, USA",
                     description="Large urban medical center with 1000+ beds",
                     dataset_type="heart", num_samples=400),
            Hospital(name="St. Mary's Medical Center", location="London, UK",
                     description="Teaching hospital specializing in cardiology",
                     dataset_type="heart", num_samples=350),
            Hospital(name="Tokyo University Hospital", location="Tokyo, Japan",
                     description="Leading research hospital in Asia Pacific",
                     dataset_type="diabetes", num_samples=450),
            Hospital(name="Berlin Health Institute", location="Berlin, Germany",
                     description="European center for chronic disease research",
                     dataset_type="kidney", num_samples=300),
            Hospital(name="Mumbai Care Hospital", location="Mumbai, India",
                     description="Multi-specialty hospital with advanced diagnostics",
                     dataset_type="diabetes", num_samples=500),
        ]
        db.add_all(hospitals)
        db.commit()
        
        # Create hospital users
        for i, h in enumerate(hospitals):
            user = User(
                username=f"hospital{i+1}",
                email=f"hospital{i+1}@fedlearn.com",
                hashed_password=get_password_hash(f"hospital{i+1}"),
                full_name=f"{h.name} Admin",
                role="hospital", hospital_id=h.id
            )
            db.add(user)
        db.commit()
        print("[OK] Default hospitals and users created")
    
    db.close()
    print("[OK] Server started successfully!")


# =====================================================
# Health Check
# =====================================================
@app.get("/")
async def root():
    return {"message": "Federated Learning Healthcare API", "status": "running", "version": "1.0.0"}


@app.get("/api/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


# =====================================================
# AUTH ENDPOINTS
# =====================================================
@app.post("/api/auth/register", response_model=UserResponse)
async def register(user_data: UserRegister, db: Session = Depends(get_db)):
    """Register a new user (admin or hospital)."""
    # Check if username exists
    if db.query(User).filter(User.username == user_data.username).first():
        raise HTTPException(status_code=400, detail="Username already taken")
    if db.query(User).filter(User.email == user_data.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    
    user = User(
        username=user_data.username, email=user_data.email,
        hashed_password=get_password_hash(user_data.password),
        full_name=user_data.full_name, role=user_data.role,
        hospital_id=user_data.hospital_id
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@app.post("/api/auth/login")
async def login(user_data: UserLogin, db: Session = Depends(get_db)):
    """Login and get JWT token."""
    user = db.query(User).filter(User.username == user_data.username).first()
    if not user or not verify_password(user_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = create_access_token(data={
        "sub": user.username, "role": user.role,
        "hospital_id": user.hospital_id
    })
    
    return {
        "access_token": token, "token_type": "bearer",
        "user": {
            "id": user.id, "username": user.username,
            "email": user.email, "full_name": user.full_name,
            "role": user.role, "hospital_id": user.hospital_id
        }
    }


@app.get("/api/auth/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user)):
    """Get current user info."""
    return current_user


# =====================================================
# HOSPITAL ENDPOINTS
# =====================================================
@app.get("/api/hospitals", response_model=List[HospitalResponse])
async def get_hospitals(db: Session = Depends(get_db)):
    """Get all hospitals."""
    return db.query(Hospital).all()


@app.get("/api/hospitals/{hospital_id}", response_model=HospitalResponse)
async def get_hospital(hospital_id: int, db: Session = Depends(get_db)):
    """Get a specific hospital."""
    hospital = db.query(Hospital).filter(Hospital.id == hospital_id).first()
    if not hospital:
        raise HTTPException(status_code=404, detail="Hospital not found")
    return hospital


@app.post("/api/hospitals", response_model=HospitalResponse)
async def create_hospital(data: HospitalCreate, db: Session = Depends(get_db),
                           current_user: User = Depends(get_admin_user)):
    """Create a new hospital (admin only)."""
    hospital = Hospital(**data.dict())
    db.add(hospital)
    db.commit()
    db.refresh(hospital)
    return hospital


@app.put("/api/hospitals/{hospital_id}", response_model=HospitalResponse)
async def update_hospital(hospital_id: int, data: HospitalUpdate,
                           db: Session = Depends(get_db),
                           current_user: User = Depends(get_admin_user)):
    """Update hospital info (admin only)."""
    hospital = db.query(Hospital).filter(Hospital.id == hospital_id).first()
    if not hospital:
        raise HTTPException(status_code=404, detail="Hospital not found")
    for key, value in data.dict(exclude_unset=True).items():
        setattr(hospital, key, value)
    db.commit()
    db.refresh(hospital)
    return hospital


# =====================================================
# FEDERATED LEARNING ENDPOINTS
# =====================================================
@app.post("/api/federated/start")
async def start_federated_training(config: FederatedConfig,
                                    background_tasks: BackgroundTasks,
                                    db: Session = Depends(get_db)):
    """Start federated learning training."""
    global fl_server, training_status
    
    if training_status["is_training"]:
        raise HTTPException(status_code=400, detail="Training already in progress")
    
    training_status = {
        "is_training": True, "current_round": 0,
        "total_rounds": config.num_rounds,
        "status": "initializing", "progress": 0
    }
    
    # Run training in background
    background_tasks.add_task(
        run_federated_training_task, config, db
    )
    
    return {"message": "Federated training started", "config": config.dict()}


def run_federated_training_task(config: FederatedConfig, db_session):
    """Background task for federated learning."""
    global fl_server, training_status
    
    try:
        from ml.federated_engine import FederatedServer
        
        # Determine dataset type from hospitals
        db = next(get_db())
        hospitals = db.query(Hospital).filter(Hospital.is_active == True).all()
        dataset_type = hospitals[0].dataset_type if hospitals else 'heart'
        num_hospitals = len(hospitals)
        
        training_status["status"] = "generating_data"
        
        # Initialize FL server
        fl_server = FederatedServer(
            dataset_type=dataset_type,
            num_hospitals=min(num_hospitals, 5)
        )
        fl_server.privacy_budget = config.privacy_budget
        fl_server.generate_and_distribute_data(total_samples=2000)
        
        training_status["status"] = "training"
        
        # Run federated rounds
        for round_num in range(1, config.num_rounds + 1):
            training_status["current_round"] = round_num
            training_status["progress"] = int((round_num / config.num_rounds) * 100)
            
            result = fl_server.train_single_round(
                round_num, config.local_epochs,
                config.learning_rate, config.batch_size
            )
            
            # Save round to database
            training_round = TrainingRound(
                round_number=round_num, status="completed",
                global_accuracy=result['global_accuracy'],
                global_loss=result['global_loss'],
                participating_hospitals=result['participating_hospitals'],
                aggregation_method=config.aggregation_method,
                privacy_budget=config.privacy_budget,
                communication_cost=result['communication_cost'],
                started_at=datetime.utcnow(),
                completed_at=datetime.utcnow()
            )
            db.add(training_round)
            
            # Save per-hospital metrics
            for h_id, metrics in result['hospital_metrics'].items():
                if h_id < len(hospitals):
                    hospital = hospitals[h_id]
                    hospital.local_accuracy = metrics['accuracy']
                    hospital.local_loss = metrics['loss']
                    hospital.is_training = False
                    hospital.last_trained = datetime.utcnow()
                    
                    metric = ModelMetrics(
                        hospital_id=hospital.id, round_number=round_num,
                        accuracy=metrics['accuracy'], loss=metrics['loss'],
                        precision_score=metrics['precision'],
                        recall_score=metrics['recall'],
                        f1_score=metrics['f1'], auc_score=metrics['auc'],
                        dataset_type=dataset_type,
                        num_samples_used=fl_server.hospital_data[h_id]['num_samples'],
                        training_time=metrics['training_time']
                    )
                    db.add(metric)
            
            db.commit()
        
        training_status["status"] = "completed"
        training_status["is_training"] = False
        training_status["progress"] = 100
        db.close()
        
    except Exception as e:
        training_status["status"] = f"error: {str(e)}"
        training_status["is_training"] = False
        print(f"[ERROR] Training error: {e}")
        import traceback
        traceback.print_exc()


@app.get("/api/federated/status")
async def get_training_status():
    """Get current federated training status."""
    global training_status, fl_server
    
    result = {**training_status}
    if fl_server and fl_server.round_history:
        result["latest_round"] = fl_server.round_history[-1]
        summary = fl_server.get_training_summary()
        if summary:
            result["summary"] = summary
    return result


@app.get("/api/federated/rounds", response_model=List[TrainingRoundResponse])
async def get_training_rounds(db: Session = Depends(get_db)):
    """Get all training rounds."""
    return db.query(TrainingRound).order_by(TrainingRound.round_number).all()


@app.get("/api/federated/metrics")
async def get_all_metrics(db: Session = Depends(get_db)):
    """Get all model metrics grouped by hospital and round."""
    metrics = db.query(ModelMetrics).order_by(
        ModelMetrics.hospital_id, ModelMetrics.round_number
    ).all()
    
    result = {}
    for m in metrics:
        h_id = str(m.hospital_id)
        if h_id not in result:
            result[h_id] = []
        result[h_id].append({
            "round": m.round_number, "accuracy": m.accuracy,
            "loss": m.loss, "precision": m.precision_score,
            "recall": m.recall_score, "f1": m.f1_score,
            "auc": m.auc_score, "training_time": m.training_time,
            "num_samples": m.num_samples_used
        })
    return result


@app.get("/api/federated/hospital/{hospital_id}/metrics")
async def get_hospital_metrics(hospital_id: int, db: Session = Depends(get_db)):
    """Get metrics for a specific hospital."""
    metrics = db.query(ModelMetrics).filter(
        ModelMetrics.hospital_id == hospital_id
    ).order_by(ModelMetrics.round_number).all()
    return [MetricsResponse.from_orm(m) for m in metrics]


# =====================================================
# PREDICTION ENDPOINTS
# =====================================================
@app.post("/api/predict")
async def predict_risk(request: PredictionRequest, db: Session = Depends(get_db)):
    """Make a health risk prediction using the global model."""
    global fl_server
    
    if fl_server is None:
        raise HTTPException(status_code=400, detail="No trained model available. Run federated training first.")
        
    if request.prediction_type != fl_server.dataset_type:
        raise HTTPException(
            status_code=400, 
            detail=f"The federated model is currently trained for '{fl_server.dataset_type}'. Cannot predict '{request.prediction_type}'."
        )
    
    from ml.model import predict_risk as make_prediction
    from ml.dataset_generator import HealthcareDataGenerator
    
    gen = HealthcareDataGenerator()
    feature_names = gen.get_feature_names(request.prediction_type)
    
    # Build feature vector
    features = []
    for fname in feature_names:
        features.append(float(request.features.get(fname, 0)))
    
    import numpy as np
    features = np.array(features, dtype=np.float32).reshape(1, -1)
    
    # Scale features
    if 0 in fl_server.hospital_scalers:
        features = fl_server.hospital_scalers[0].transform(features)
    
    result = make_prediction(fl_server.global_model, features[0])
    
    # Log prediction
    log = PredictionLog(
        prediction_type=request.prediction_type,
        risk_score=result['risk_score'],
        risk_level=result['risk_level'],
        input_features=json.dumps(request.features),
        model_version="federated_v1"
    )
    db.add(log)
    db.commit()
    
    return {
        "prediction_type": request.prediction_type,
        "risk_score": result['risk_score'],
        "risk_level": result['risk_level'],
        "confidence": result['confidence'],
        "features_used": request.features,
        "model_version": "federated_v1"
    }


# =====================================================
# DASHBOARD ENDPOINTS
# =====================================================
@app.get("/api/dashboard/stats")
async def get_dashboard_stats(db: Session = Depends(get_db)):
    """Get admin dashboard statistics."""
    total_hospitals = db.query(Hospital).count()
    active_hospitals = db.query(Hospital).filter(Hospital.is_active == True).count()
    total_rounds = db.query(TrainingRound).count()
    completed_rounds = db.query(TrainingRound).filter(TrainingRound.status == "completed").count()
    total_predictions = db.query(PredictionLog).count()
    
    from sqlalchemy import func
    avg_acc = db.query(func.avg(TrainingRound.global_accuracy)).filter(
        TrainingRound.status == "completed"
    ).scalar() or 0
    best_acc = db.query(func.max(TrainingRound.global_accuracy)).filter(
        TrainingRound.status == "completed"
    ).scalar() or 0
    total_samples = db.query(func.sum(Hospital.num_samples)).scalar() or 0
    
    return {
        "total_hospitals": total_hospitals,
        "active_hospitals": active_hospitals,
        "total_rounds": total_rounds,
        "completed_rounds": completed_rounds,
        "avg_global_accuracy": round(float(avg_acc), 4),
        "best_global_accuracy": round(float(best_acc), 4),
        "total_predictions": total_predictions,
        "total_samples": int(total_samples),
        "training_status": training_status
    }


@app.get("/api/dashboard/hospital-performance")
async def get_hospital_performance(db: Session = Depends(get_db)):
    """Get performance leaderboard for all hospitals."""
    hospitals = db.query(Hospital).all()
    performance = []
    for h in hospitals:
        latest_metric = db.query(ModelMetrics).filter(
            ModelMetrics.hospital_id == h.id
        ).order_by(ModelMetrics.round_number.desc()).first()
        
        performance.append({
            "id": h.id, "name": h.name, "location": h.location,
            "dataset_type": h.dataset_type, "num_samples": h.num_samples,
            "local_accuracy": h.local_accuracy, "local_loss": h.local_loss,
            "is_active": h.is_active,
            "latest_f1": latest_metric.f1_score if latest_metric else 0,
            "latest_auc": latest_metric.auc_score if latest_metric else 0,
            "last_trained": h.last_trained.isoformat() if h.last_trained else None
        })
    
    # Sort by accuracy descending
    performance.sort(key=lambda x: x["local_accuracy"], reverse=True)
    return performance


# =====================================================
# Run the application
# =====================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
