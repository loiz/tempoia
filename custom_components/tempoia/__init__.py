"""TempoIA integration entry point for Home Assistant.

Provides:
- 14‑day forecast sensors (sensor platform)
- Admin services ``tempoia.train_model`` and ``tempoia.update_database``

All services are registered programmatically, so no ``services.yaml`` is required.
"""

from __future__ import annotations

import logging
from typing import Any

import aiohttp
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.const import CONF_HOST, CONF_TOKEN, Platform, CONF_SCAN_INTERVAL
from homeassistant.helpers import aiohttp_client
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator

from .const import DOMAIN, DEFAULT_SCAN_INTERVAL, DATA_KEY_COORDINATOR
from .sensor import TempoiaDataUpdateCoordinator

PLATFORMS: list[Platform] = [Platform.SENSOR, Platform.CALENDAR]

_LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Core setup / unload
# ---------------------------------------------------------------------------

async def async_setup(hass: HomeAssistant, config: dict) -> bool:
    """Legacy setup from configuration.yaml (no action needed)."""
    return True

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up TempoIA from a config entry.

    Stores the entry data, forwards to the sensor platform (creates the 14 sensors)
    and registers admin services for training and database updates.
    """
    _LOGGER.debug("Setting up TempoIA config entry %s", entry.entry_id)
    hass.data.setdefault(DOMAIN, {})

    # Create and refresh the coordinator
    scan_interval_min = entry.options.get(CONF_SCAN_INTERVAL, DEFAULT_SCAN_INTERVAL)
    coordinator = TempoiaDataUpdateCoordinator(
        hass,
        entry.data.get(CONF_HOST),
        entry.data.get(CONF_TOKEN),
        scan_interval_min,
    )
    await coordinator.async_config_entry_first_refresh()

    # Store the coordinator for the sensor platform to use
    hass.data[DOMAIN][entry.entry_id] = {
        DATA_KEY_COORDINATOR: coordinator,
    }

    # Forward to sensor platform
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    # Register services (no services.yaml needed)
    await _register_services(hass, entry)
    return True

async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry, remove services and clean up stored data."""
    _LOGGER.debug("Unloading TempoIA config entry %s", entry.entry_id)
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if unload_ok:
        # Remove registered services
        hass.services.async_remove(DOMAIN, "train_model")
        hass.services.async_remove(DOMAIN, "update_database")
        hass.data[DOMAIN].pop(entry.entry_id)
    return unload_ok

# ---------------------------------------------------------------------------
# Service registration
# ---------------------------------------------------------------------------

async def _register_services(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Register ``tempoia.train_model`` and ``tempoia.update_database`` services.

    The services call the corresponding TempoIA API endpoints using the host and
    token stored in the config entry.
    """
    host: str = entry.data.get(CONF_HOST, "")
    token: str | None = entry.data.get(CONF_TOKEN)

    async def _call_api(endpoint: str, payload: dict | None = None) -> Any:
        url = f"{host.rstrip('/')}{endpoint}"
        headers: dict[str, str] = {}
        if token:
            headers["X-API-Token"] = token
        session: aiohttp.ClientSession = aiohttp_client.async_get_clientsession(hass)
        async with session.post(url, json=payload or {}, headers=headers) as resp:
            if resp.status != 200:
                raise Exception(f"TempoIA service call failed ({resp.status})")
            return await resp.json()

    async def train_model_service(call):
        _LOGGER.info("Calling TempoIA /train via Home Assistant service")
        await _call_api("/train")
        _LOGGER.info("TempoIA model training triggered successfully")

    async def update_database_service(call):
        years = call.data.get("years", 10)
        _LOGGER.info("Calling TempoIA /update_database via Home Assistant service (years=%s)", years)
        await _call_api("/update_database", {"years": years})
        _LOGGER.info("TempoIA database update triggered successfully")

    hass.services.async_register(
        DOMAIN,
        "train_model",
        train_model_service
    )
    hass.services.async_register(
        DOMAIN, "update_database", update_database_service
    )

    async def refresh_forecast_service(call):
        """Service to manually refresh the forecast data."""
        _LOGGER.info("Manual forecast refresh triggered for TempoIA")
        coordinator: DataUpdateCoordinator = hass.data[DOMAIN][entry.entry_id][DATA_KEY_COORDINATOR]
        await coordinator.async_request_refresh()

    hass.services.async_register(
        DOMAIN, "refresh_forecast", refresh_forecast_service
    )
# End of file
