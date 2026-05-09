# Final AI Project - Crop Recommendation and Yield Prediction System

A full-stack agricultural decision-support system that combines machine learning, explainable AI, map-assisted field input, and realistic market-aware ranking to recommend suitable crops and estimate yield outcomes.

## Project Overview

This project is designed to help users choose the best crop for a given field condition by combining:

- crop recommendation using classification models
- yield prediction using regression
- rule-based seasonal and climate screening
- SHAP-based explainability for model interpretation
- map-assisted area and location input
- a FastAPI backend with an integrated web interface

The system takes soil nutrients, pH, temperature, humidity, rainfall, moisture, area, and regional context as inputs, then returns ranked crop recommendations, cost-profit estimates, risk signals, and explainable insights.

## Demo and Links
  
- Render Demo: `[(https://santhosh-23mis0616.onrender.com/)]`
- GitHub Repository: [final-ai-project](https://github.com/SanthoshS23MIS0616/final-ai-project)
- Screenshots PDF / Drive Link: `[https://drive.google.com/file/d/1YdznPG5hFJdTpdQkcMZzZ3FbCu6NdMK4/view?usp=drivesdk]`

## Architecture

User (Browser) -> FastAPI Web App -> Inference Engine -> Trained Models -> Reports / Visual Assets

Main flow:

1. User enters or auto-fills land and climate-related inputs
2. FastAPI receives the request through `/api/predict`
3. The inference layer preprocesses inputs and applies trained models
4. The system ranks crop options, estimates yield and profit, and returns explainable outputs
5. The frontend displays recommendation cards, tables, SHAP graphs, and analytics

## Models Used

### Classification Models

- LightGBM Classifier
- CatBoost Classifier
- Random Forest Classifier
- Logistic Regression Classifier
- Stacking Classifier

### Regression Model

- LightGBM Regressor for crop yield prediction

### Explainability

- SHAP summary plots for LightGBM and CatBoost
- feature-importance based explanation cards in the UI

## Key Highlights

- realistic crop recommendation workflow with seasonal filtering
- market-aware cost, revenue, and profit estimation
- SHAP-based explainability support
- FastAPI backend with browser-based interface
- map-based field interaction for area and location input
- Render-ready Docker deployment
- static frontend served directly by the backend

## Reported Performance

Based on the saved training artifacts included in the project:

- LightGBM classification accuracy: `0.9371`
- Stacking classification accuracy: `0.9339`
- Stacking top-3 accuracy: `0.9971`
- Yield regression R²: `0.9181`

These values come from the saved training report and may be updated if models are retrained.

## Project Structure

```text
final-ai-project/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── routes.py
│   │   ├── services/
│   │   │   └── predictor.py
│   │   ├── static/
│   │   │   ├── app.js
│   │   │   ├── index.html
│   │   │   └── styles.css
│   │   ├── config.py
│   │   ├── main.py
│   │   └── schemas.py
│   └── ml/
│       ├── config_realistic_v2.py
│       ├── crop_rules_realistic_v2.py
│       ├── inference_realistic_v2.py
│       ├── models.py
│       └── training_realistic_v2.py
├── data/
│   └── realistic_v2/
├── models/
│   └── realistic_v2/
├── reports/
│   └── realistic_v2/
├── scripts/
│   ├── run_api.py
│   └── train_realistic_v2.py
├── Dockerfile
├── render.yaml
├── requirements.txt
├── runtime.txt
└── README.md
```

## Installation

### 1. Clone the Repository

```powershell
git clone https://github.com/SanthoshS23MIS0616/final-ai-project.git
cd final-ai-project
```

### 2. Create and Activate a Virtual Environment

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

## Running the Project Locally

### Start the FastAPI Application

```powershell
python scripts/run_api.py
```

Runs on:

- App: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- Health API: [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)

## Training the Models

To retrain the realistic V2 pipeline:

```powershell
python scripts/train_realistic_v2.py
```

This regenerates:

- model artifacts under `models/realistic_v2/`
- training report JSON
- SHAP summary images under `reports/realistic_v2/`

## API Endpoints

### `GET /api/health`

Checks whether required model artifacts are available.

### `GET /api/metadata`

Returns:

- default input values
- training summary
- SHAP and explainability asset references

### `POST /api/predict`

Runs crop recommendation and yield estimation.

Example request body:

```json
{
  "nitrogen": 70,
  "phosphorous": 55,
  "potassium": 40,
  "ph": 6.5,
  "temperature_c": 28,
  "humidity": 75,
  "rainfall_mm": 120,
  "area": 1.5,
  "top_k": 5
}
```

### `POST /api/train`

Retrains the models from the included dataset pipeline.

## How It Works

### Input Processing

- handles numeric and categorical agricultural features
- applies imputation and scaling for numerical columns
- applies one-hot encoding for categorical columns

### Recommendation Logic

- predicts crop suitability with multiple classifiers
- combines model outputs using a stacking ensemble
- filters crops using sowing-season and climate rules
- ranks crops using yield, cost, profit, risk, and sustainability

### Yield Estimation

- uses a LightGBM regressor to estimate expected yield
- adjusts downstream profit calculations using market and cost logic

### Explainability

- generates SHAP summary plots for model interpretation
- exposes top important features in the web interface

## Frontend Features

- clean responsive web interface
- live prediction output cards
- recommendation table with crop-wise metrics
- SHAP summary graph display
- map interaction for land selection
- weather-assisted input filling

## Deployment

This project is configured for Render using:

- `Dockerfile`
- `render.yaml`

### Render Deployment Notes

- runtime: Docker
- health check path: `/api/health`
- main application entry: `backend.app.main:app`

## Future Improvements

- richer real-time weather integration
- mobile-first dashboard refinement
- multilingual interface support
- expanded dataset coverage across more regions
- additional explainability views for end users

## Author

**Santhosh S**  
**23MIS0616**

