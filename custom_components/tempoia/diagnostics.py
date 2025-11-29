"""Diagnostics support for TempoIA."""
from __future__ import annotations

from typing import Any

from homeassistant.components.diagnostics import async_redact_data
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_TOKEN
from homeassistant.core import HomeAssistant

from .const import DOMAIN, DATA_KEY_COORDINATOR

TO_REDACT = {CONF_TOKEN}


async def async_get_config_entry_diagnostics(
    hass: HomeAssistant, entry: ConfigEntry
) -> dict[str, Any]:
    """Return diagnostics for a config entry."""
    coordinator = hass.data[DOMAIN][entry.entry_id][DATA_KEY_COORDINATOR]

    diagnostics_data = {
        "entry": {
            "title": entry.title,
            "version": entry.version,
            "data": async_redact_data(entry.data, TO_REDACT),
            "options": async_redact_data(entry.options, TO_REDACT),
        },
        "coordinator": {
            "last_update_success": coordinator.last_update_success,
            "last_update_time": coordinator.last_update_success_time.isoformat()
            if coordinator.last_update_success_time
            else None,
            "update_interval": str(coordinator.update_interval),
        },
    }

    # Add coordinator data summary (without sensitive info)
    if coordinator.data:
        predictions = coordinator.data.get("predictions", [])
        diagnostics_data["coordinator"]["data_summary"] = {
            "cached": coordinator.data.get("cached", False),
            "predictions_count": len(predictions),
            "first_prediction_date": predictions[0].get("date") if predictions else None,
            "last_prediction_date": predictions[-1].get("date") if predictions else None,
        }

    return diagnostics_data
