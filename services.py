"""Service layer for TempoIA API.
Handles business logic, predictor singleton, and caching.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from tempoia import TempoWeatherPredictor
from settings import settings

# Global predictor instance
_predictor: Optional[TempoWeatherPredictor] = None

# In-memory cache
_prediction_cache: Dict[str, Any] = {"data": None, "timestamp": 0}

async def get_predictor() -> TempoWeatherPredictor:
    """Get or create the global TempoWeatherPredictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = TempoWeatherPredictor(db_path=settings.db_path)
        # Ensure database structure exists
        _predictor.create_database()
    return _predictor

async def get_cached_prediction() -> Optional[Dict[str, Any]]:
    """Retrieve predictions from cache if available."""
    return _prediction_cache

async def set_cached_prediction(predictions: List[Dict[str, Any]]) -> None:
    """Update the prediction cache with new data."""
    _prediction_cache["data"] = predictions
    _prediction_cache["timestamp"] = time.time()

async def clear_prediction_cache() -> None:
    """Invalidate the prediction cache."""
    _prediction_cache["data"] = None
    _prediction_cache["timestamp"] = 0
