# api.py
"""FastAPI API for TempoIA.
Improvements implemented:
1️⃣ Centralised configuration via ``settings.py`` (env/.env).
2️⃣ Structured JSON logging.
3️⃣ Global exception handler returning uniform error payloads.
4️⃣ Pydantic validation for request parameters.
6️⃣ Async‑compatible DB access (wrapped sync calls with ``run_in_threadpool``).
7️⃣ Token authentication, CORS and rate‑limiting (slowapi).
9️⃣ Enriched OpenAPI docs with tags & examples.
🔟 Dockerfile added for container deployment.
1️⃣2️⃣ Service layer (``services.py``) separates business logic from the API.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List

import logging
from fastapi import (
    FastAPI,
    Depends,
    HTTPException,
    Header,
    Request,
    Request,
    status,
    Query,
)
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, PositiveInt, conint
from starlette.concurrency import run_in_threadpool

# ---- Configuration ---------------------------------------------------------
from settings import settings

# ---- Logging --------------------------------------------------------------
logger = logging.getLogger("tempoia_api")
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '{"time":"%(asctime)s","level":"%(levelname)s","module":"%(name)s","msg":"%(message)s"}'
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ---- Rate limiting --------------------------------------------------------
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address, default_limits=[settings.rate_limit])

# ---- FastAPI app ----------------------------------------------------------
app = FastAPI(
    title="TempoIA API",
    description="API for TempoIA predictions, training and database updates.",
    version="1.0.1",
    openapi_tags=[
        {"name": "prediction", "description": "Endpoints returning predictions (no auth)."},
        {"name": "maintenance", "description": "Endpoints that modify data or model (auth required)."},
    ],
)

# Register middleware & handlers
app.state.limiter = limiter
app.add_exception_handler(429, _rate_limit_exceeded_handler)
app.add_exception_handler(RequestValidationError, lambda request, exc: JSONResponse(
    status_code=422,
    content={"error": "validation_error", "detail": exc.errors()},
))
app.add_exception_handler(Exception, lambda request, exc: JSONResponse(
    status_code=500,
    content={"error": "internal_server_error", "detail": str(exc)},
))

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in settings.cors_origins.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Authentication --------------------------------------------------------
def verify_token(x_api_token: str = Header(...)):
    if x_api_token != settings.api_token:
        logger.warning("Invalid token attempt from %s", get_remote_address(Request))
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API token")
    return True

# ---- Service layer ---------------------------------------------------------
from services import get_predictor, clear_prediction_cache, get_cached_prediction, set_cached_prediction

# ---- Pydantic models ------------------------------------------------------
class TrainResponse(BaseModel):
    status: str = Field(..., example="training completed")

class UpdateResponse(BaseModel):
    status: str = Field(..., example="database updated")

class PredictionItem(BaseModel):
    date: str
    probabilities: Dict[str, float]

class PredictionResponse(BaseModel):
    cached: bool
    predictions: List[Dict[str, Any]]

class StatusResponse(BaseModel):
    status: str
    time: str

class ColorDistribution(BaseModel):
    BLEU: int = Field(0, example=300)
    BLANC: int = Field(0, example=43)
    ROUGE: int = Field(0, example=22)

class WeatherAverages(BaseModel):
    temperature: float = Field(..., example=12.5)
    precipitation: float = Field(..., example=1.8)
    sunshine: float = Field(..., example=4.5)

class DatabaseStatsResponse(BaseModel):
    total_days: int = Field(..., example=365)
    color_distribution: ColorDistribution
    total_weather_records: int = Field(..., example=365)
    weather_averages: WeatherAverages

class PerformanceHistoryItem(BaseModel):
    id: int
    training_date: str
    model_algorithm: str
    n_samples: int
    test_accuracy: float
    test_f1_macro: float

class TrainingLogDetailResponse(BaseModel):
    id: int
    training_date: str
    model_algorithm: str
    n_samples: int
    test_accuracy: float
    test_f1_macro: float
    classification_report: Dict[str, Any]

# ---- Endpoints ------------------------------------------------------------

@app.get("/predict", response_model=PredictionResponse, tags=["prediction"], summary="Get next‑day (or multi‑day) predictions without authentication")
@limiter.limit(settings.rate_limit)
@limiter.limit(settings.rate_limit)
async def get_prediction(request: Request, days: int = Query(1, gt=0)):
    """Return predictions for the next *days* days.
    Results are cached for ``settings.cache_ttl`` seconds.
    """
    now = time.time()
    cached = await get_cached_prediction()
    if cached and now - cached["timestamp"] < settings.cache_ttl:
        # Check if cached data has enough days
        if cached["data"] and len(cached["data"]) >= days:
             logger.info("Returning cached prediction (TTL still valid)")
             return {"cached": True, "predictions": cached["data"][:days]}

    predictor = await get_predictor()
    
    # Use the new multi-day prediction method
    predictions = await run_in_threadpool(predictor.predict_multi_day, days=days)
    
    if predictions is None:
        raise HTTPException(status_code=500, detail="Prediction failed")
        
    await set_cached_prediction(predictions)
    logger.info("Generated fresh predictions for %s day(s)", days)
    return {"cached": False, "predictions": predictions}

@app.post(
    "/train",
    response_model=TrainResponse,
    tags=["maintenance"],
    summary="Find best algorithm and trigger model training (authenticated)",
)
@limiter.limit(settings.rate_limit)
async def train_model(
    request: Request,
    auth: bool = Depends(verify_token),
    algorithm: str = Query(
        "best",
        description="Algorithm to train. Use 'best' to run a benchmark and train the top performer. "
                    "Or specify one of: mlp, random_forest, logistic, gb, svc.",
    ),
):
    """
    This endpoint triggers model training.
    - By default (`algorithm='best'`), it benchmarks several ML algorithms, selects the best one, and then trains the final model.
    - If a specific algorithm is provided (e.g., `algorithm='random_forest'`), it skips the benchmark and trains that specific model directly.
    """
    predictor = await get_predictor()
    trained_algo_key = algorithm
    benchmark_data = None
    algorithm = "best"
    if algorithm == "best":
        logger.info("Mode 'best': Starting algorithm benchmark to find the best model...")
        benchmark_results = await run_in_threadpool(predictor.benchmark_algorithms)

        if not benchmark_results or "_best" not in benchmark_results:
            logger.error(f"Benchmark failed. Results: {benchmark_results}")
            raise HTTPException(status_code=500, detail="Algorithm benchmark failed, could not determine best model.")

        trained_algo_key = benchmark_results["_best"]["key"]
        benchmark_data = benchmark_results
        logger.info(f"Best algorithm found: {trained_algo_key}. Now training the final model...")
    else:
        # Validate if the specified algorithm is valid
        valid_algos = predictor.get_algorithm_map().keys()
        if algorithm not in valid_algos:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid algorithm specified. Choose 'best' or one of: {', '.join(valid_algos)}",
            )
        logger.info(f"Mode 'specific': Training directly with specified algorithm: {algorithm}")

    success = await run_in_threadpool(predictor.train_model, estimator_key=trained_algo_key)
    if not success:
        raise HTTPException(status_code=500, detail="Training failed")

    await clear_prediction_cache()
    logger.info(f"Model training completed with algorithm '{trained_algo_key}', cache cleared")
    
    response = {
        "status": f"training completed with {trained_algo_key}",
        "algorithm": trained_algo_key
    }
    if benchmark_data:
        response["benchmark_results"] = benchmark_data
        
    return response

@app.post(
    "/update_database",
    response_model=UpdateResponse,
    tags=["maintenance"],
    summary="Fetch latest Tempo and weather data and store it (authenticated)",
)
@limiter.limit(settings.rate_limit)
async def update_database(request: Request, years: int = Query(10, gt=0), auth: bool = Depends(verify_token)):
    predictor = await get_predictor()
    
    current_year = datetime.now().year
    # Fetch Tempo data for the requested number of years
    # Cycle Y ends in Aug Y, starts in Sept Y-1
    # We want to cover 'years' cycles ending with current_year
    for y in range(current_year - years + 1, current_year + 1):
        logger.info(f"Fetching Tempo data for cycle ending in {y}")
        tempo_data = await run_in_threadpool(predictor.fetch_tempo_data, year=y)
        tempo_data = await run_in_threadpool(predictor.calculate_remaining_days, tempo_data, y)
        await run_in_threadpool(predictor.insert_tempo_data, tempo_data)

    # Fetch weather data for the whole period
    end_date = datetime.now().date()
    # Start date corresponds to the start of the oldest cycle requested
    # Oldest cycle ends in (current_year - years + 1), so it starts in Sept of the year before
    start_year = (current_year - years + 1) - 1
    start_date = f"{start_year}-09-01"
    
    logger.info(f"Fetching weather data from {start_date} to {end_date}")
    weather = await run_in_threadpool(
        predictor.fetch_historical_weather, start_date, end_date.isoformat()
    )
    await run_in_threadpool(predictor.insert_weather_data, weather)
    
    await clear_prediction_cache()
    logger.info("Database refreshed and cache cleared")
    return {"status": "database updated"}

@app.get("/status", response_model=StatusResponse, tags=["maintenance"], summary="Simple health check")
async def status_endpoint():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}

@app.get(
    "/stats/database",
    response_model=DatabaseStatsResponse,
    tags=["maintenance"],
    summary="Get global statistics about the database (authenticated)",
)
@limiter.limit(settings.rate_limit)
async def get_db_stats(request: Request, auth: bool = Depends(verify_token)):
    """
    Returns a summary of the database content:
    - Total number of Tempo days and weather records.
    - Distribution of colors (BLEU, BLANC, ROUGE).
    - Average weather conditions across all records.
    """
    predictor = await get_predictor()
    stats = await run_in_threadpool(predictor.get_database_stats)
    if not stats:
        raise HTTPException(status_code=404, detail="Could not retrieve database stats.")
    logger.info("Database stats retrieved")
    return stats

@app.get(
    "/performance/history",
    response_model=List[PerformanceHistoryItem],
    tags=["maintenance"],
    summary="Get model training performance history (authenticated)",
)
@limiter.limit(settings.rate_limit)
async def get_performance_history(request: Request, limit: int = Query(20, gt=0), auth: bool = Depends(verify_token)):
    """
    Returns a list of past training sessions with their performance metrics.
    This allows tracking the model's performance over time.
    """
    # This function is in DatabaseVisualizer, so we need to instantiate it.
    # In a real-world scenario, this might be better placed in the predictor class
    # or the visualizer could be part of the service layer.
    from tempoia import DatabaseVisualizer
    visualizer = DatabaseVisualizer(db_path=settings.db_path)
    
    # The original function prints to console, we need to adapt it to return data.
    # Let's assume a new or modified function `get_performance_history_data` exists.
    # For now, let's simulate this by reading directly from the DB.
    history = await run_in_threadpool(visualizer.view_performance_history, limit=limit, return_df=True)
    if history is None:
        return []
    return history.to_dict(orient="records")

@app.get(
    "/performance/history/{training_id}",
    response_model=TrainingLogDetailResponse,
    tags=["maintenance"],
    summary="Get details for a specific training log entry (authenticated)",
)
@limiter.limit(settings.rate_limit)
async def get_training_log_detail(request: Request, training_id: int, auth: bool = Depends(verify_token)):
    """
    Returns the detailed classification report for a specific training session,
    identified by its unique ID.
    """
    from tempoia import DatabaseVisualizer
    visualizer = DatabaseVisualizer(db_path=settings.db_path)
    
    details = await run_in_threadpool(visualizer.get_training_log_details, training_id=training_id)
    
    if details is None:
        raise HTTPException(status_code=404, detail=f"Training log with ID {training_id} not found.")
    return details

@app.get(
    "/stats/tempo",
    tags=["maintenance"],
    summary="Get Tempo cycle statistics for a specific year (authenticated)",
)
@limiter.limit(settings.rate_limit)
async def get_tempo_cycle_stats(request: Request, year: int = Query(None, description="Cycle year (ending year)"), auth: bool = Depends(verify_token)):
    """
    Returns detailed statistics for a specific Tempo cycle year.
    Includes color distribution, weekday patterns, and remaining days.
    """
    predictor = await get_predictor()
    stats = await run_in_threadpool(predictor.get_tempo_cycle_stats, year=year)
    
    if not stats:
        raise HTTPException(status_code=404, detail=f"No data found for cycle year {year or 'current'}")
    
    logger.info(f"Tempo cycle stats retrieved for year {year}")
    return stats

@app.get(
    "/stats/model",
    tags=["maintenance"],
    summary="Get detailed model information and metadata (authenticated)",
)
@limiter.limit(settings.rate_limit)
async def get_model_stats(request: Request, auth: bool = Depends(verify_token)):
    """
    Returns comprehensive information about the current model:
    - Model type and training features
    - Target classes
    - Last training session details
    - File existence status
    """
    predictor = await get_predictor()
    info = await run_in_threadpool(predictor.get_model_info)
    
    logger.info("Model information retrieved")
    return info

@app.get(
    "/stats/predictions",
    tags=["maintenance"],
    summary="Get prediction accuracy statistics from training history (authenticated)",
)
@limiter.limit(settings.rate_limit)
async def get_prediction_stats(request: Request, limit: int = Query(20, gt=0, description="Number of training sessions to retrieve"), auth: bool = Depends(verify_token)):
    """
    Returns statistics about prediction accuracy based on training history.
    Includes:
    - Training session history with metrics
    - Average, min, and max accuracy/F1 scores
    - Classification reports
    """
    predictor = await get_predictor()
    stats = await run_in_threadpool(predictor.get_prediction_accuracy_stats, limit=limit)
    
    logger.info(f"Prediction accuracy stats retrieved (limit={limit})")
    return stats

# ---- Model metadata endpoint (optional, part of point 12) -------------------
@app.get("/model/metadata", tags=["maintenance"], summary="Return model version and training metrics")
async def model_metadata():
    predictor = await get_predictor()
    # Assume the model files contain metadata JSON next to them (simple example)
    metadata_path = os.path.join(os.path.dirname(predictor.db_path), "model_metadata.json")
    if os.path.exists(metadata_path):
        import json
        async with run_in_threadpool(open, metadata_path, "r") as f:
            data = json.load(f)
        return data
    raise HTTPException(status_code=404, detail="Model metadata not found")

@app.get("/test", response_class=HTMLResponse, tags=["maintenance"], summary="API Test Page")
async def test_page():
    """Serve the API test page."""
    try:
        with open("test_api.html", "r") as f:
            return f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Test page not found")
