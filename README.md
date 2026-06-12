# 🏥 FedHealth AI - Federated Learning for Cross-Hospital Risk Prediction

> **"Federated Learning For Cross-Hospital Risk Prediction Without Data Sharing"**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python)](https://python.org)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react)](https://reactjs.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-EE4C2C?logo=pytorch)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?logo=fastapi)](https://fastapi.tiangolo.com)

A complete federated learning system where **multiple hospitals collaboratively train an AI model** for healthcare risk prediction **without sharing patient data**. Only model weights are exchanged, ensuring full data privacy.

---

## 🌟 Key Features

| Feature | Description |
|---------|------------|
| 🏥 **Multi-Hospital Simulation** | 5 hospitals with separate datasets |
| 🔄 **FedAvg Algorithm** | Federated Averaging implementation |
| 🫀 **Heart Disease Prediction** | 13-feature risk assessment |
| 🩸 **Diabetes Prediction** | 8-feature Pima-style analysis |
| 🫘 **Kidney Disease Prediction** | 19-feature CKD detection |
| 🔒 **Differential Privacy** | ε-DP noise injection |
| 📊 **Interactive Dashboard** | Real-time charts & analytics |
| 🔐 **JWT Authentication** | Role-based access control |
| 🐳 **Docker Support** | One-command deployment |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    FEDERATED SERVER                       │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │ Global Model │  │ FedAvg Agg.  │  │ Privacy Engine │  │
│  └──────┬──────┘  └──────┬───────┘  └────────┬───────┘  │
│         │               │                    │           │
│         └───────────────┼────────────────────┘           │
│                         │                                │
│         ┌───────────────┼───────────────┐                │
│         ▼               ▼               ▼                │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐           │
│  │Hospital 1│    │Hospital 2│    │Hospital N│           │
│  │Local Data│    │Local Data│    │Local Data│           │
│  │Local Model│   │Local Model│   │Local Model│          │
│  └──────────┘    └──────────┘    └──────────┘           │
│                                                          │
│  ⚠️ DATA NEVER LEAVES HOSPITALS - ONLY WEIGHTS SHARED   │
└──────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
federated-learning-project/
├── backend/                    # FastAPI Backend
│   ├── main.py                # Main application & API endpoints
│   ├── database.py            # SQLAlchemy models & DB config
│   ├── auth.py                # JWT authentication utilities
│   ├── schemas.py             # Pydantic request/response schemas
│   ├── requirements.txt       # Python dependencies
│   └── .env                   # Environment variables
│
├── frontend/                   # React Frontend
│   ├── src/
│   │   ├── App.jsx            # Main app with routing & auth
│   │   ├── main.jsx           # Entry point
│   │   ├── index.css          # Global styles & Tailwind
│   │   ├── components/
│   │   │   └── Sidebar.jsx    # Navigation sidebar
│   │   ├── pages/
│   │   │   ├── Login.jsx      # Auth page
│   │   │   ├── Dashboard.jsx  # Admin dashboard
│   │   │   ├── Hospitals.jsx  # Hospital management
│   │   │   ├── FederatedTraining.jsx  # FL training control
│   │   │   ├── Predictions.jsx # Risk prediction
│   │   │   └── Analytics.jsx  # Detailed analytics
│   │   └── services/
│   │       └── api.js         # API service layer
│   ├── package.json
│   ├── tailwind.config.js
│   └── vite.config.js
│
├── ml/                        # Machine Learning Module
│   ├── __init__.py
│   ├── model.py               # PyTorch neural network
│   ├── dataset_generator.py   # Synthetic healthcare data
│   └── federated_engine.py    # FedAvg implementation
│
├── docker/                    # Docker Configuration
│   ├── Dockerfile.backend
│   ├── Dockerfile.frontend
│   ├── docker-compose.yml
│   └── nginx.conf
│
└── README.md                  # This file
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+** with pip
- **Node.js 18+** with npm
- Git

### 1. Clone & Setup Backend

```bash
# Navigate to the project
cd federated-learning-project

# Create Python virtual environment
cd backend
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

# Install Python dependencies
pip install -r requirements.txt

# Start the backend server
python main.py
```

The backend starts at **http://localhost:8000**
API docs at **http://localhost:8000/docs**

### 2. Setup Frontend

```bash
# In a new terminal
cd frontend

# Install npm dependencies
npm install

# Start development server
npm run dev
```

The frontend starts at **http://localhost:5173**

### 3. Login & Use

Use these demo credentials:

| Role | Username | Password |
|------|----------|----------|
| Admin | `admin` | `admin123` |
| Hospital 1 | `hospital1` | `hospital1` |
| Hospital 2 | `hospital2` | `hospital2` |

---

## 📖 How It Works

### Federated Learning Workflow

1. **Initialize** - Server creates a global neural network model
2. **Distribute** - Global model weights are sent to each hospital
3. **Local Training** - Each hospital trains on its private data
4. **Privacy** - Differential privacy noise is added to local weights
5. **Upload** - Only model weights (not data) are sent back
6. **Aggregate** - Server performs FedAvg: `w_global = Σ(n_k/n_total) × w_k`
7. **Update** - New global model is distributed for next round
8. **Repeat** - Steps 2-7 repeat for multiple rounds

### FedAvg Algorithm

```python
# Pseudocode for Federated Averaging
for each round t:
    for each hospital k in parallel:
        w_k = LocalTrain(global_model, local_data_k, epochs)
        w_k = AddDPNoise(w_k, epsilon)  # Privacy
    
    # Weighted aggregation
    w_global = Σ (n_k / n_total) × w_k
```

### Neural Network Architecture

```
Input Layer (varies by disease) 
    → Dense(128) + BatchNorm + ReLU + Dropout(0.3)
    → Dense(64)  + BatchNorm + ReLU + Dropout(0.2)
    → Dense(32)  + ReLU + Dropout(0.1)
    → Dense(1)   + Sigmoid → Risk Score [0,1]
```

---

## 🔌 API Documentation

### Authentication
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/auth/register` | Register new user |
| POST | `/api/auth/login` | Login & get JWT token |
| GET | `/api/auth/me` | Get current user info |

### Hospitals
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/hospitals` | List all hospitals |
| POST | `/api/hospitals` | Create hospital (admin) |
| PUT | `/api/hospitals/{id}` | Update hospital (admin) |

### Federated Learning
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/federated/start` | Start FL training |
| GET | `/api/federated/status` | Get training status |
| GET | `/api/federated/rounds` | Get round history |
| GET | `/api/federated/metrics` | Get all metrics |

### Predictions
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/predict` | Make risk prediction |

### Dashboard
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/dashboard/stats` | Dashboard statistics |
| GET | `/api/dashboard/hospital-performance` | Performance leaderboard |

---

## 🔒 Privacy Features

- **Data Isolation**: Patient data never leaves the hospital node
- **Differential Privacy**: Calibrated Laplace noise (ε-DP) added to model weights
- **Secure Aggregation**: FedAvg ensures server only sees averaged weights
- **No Raw Data Sharing**: Only model parameters are communicated

---

## 🐳 Docker Deployment

```bash
cd docker
docker-compose up --build
```

- Frontend: http://localhost
- Backend: http://localhost:8000

---

## 📚 Technologies Used

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Frontend | React 18 | UI framework |
| Styling | Tailwind CSS | Utility-first CSS |
| Charts | Recharts | Data visualization |
| Backend | FastAPI | REST API server |
| Auth | JWT + bcrypt | Authentication |
| Database | SQLite + SQLAlchemy | Data persistence |
| ML | PyTorch | Neural network |
| FL | Custom FedAvg | Federated learning |
| Privacy | ε-DP | Differential privacy |

---

## 👨‍🎓 For Viva / Presentation

**Key Points to Explain:**
1. Why federated learning? → Privacy-preserving collaborative ML
2. How does FedAvg work? → Weighted average of model parameters
3. Why not just share data? → HIPAA, GDPR, patient privacy regulations
4. What is differential privacy? → Mathematical guarantee against data leakage
5. How is communication efficient? → Only model weights sent, not raw data
6. Real-world applications → Cross-hospital research, rare disease studies

---

## 📄 License

This project is created for educational and research purposes.

---

## 🙏 Acknowledgments

- McMahan et al. (2017) - "Communication-Efficient Learning of Deep Networks from Decentralized Data"
- UCI Machine Learning Repository - Healthcare Datasets
- Google AI - Federated Learning Research
