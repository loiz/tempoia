"""Sensor platform for TempoIA predictions."""

import logging
from datetime import date, timedelta

from aiohttp import ClientSession
from homeassistant.components.sensor import SensorEntity
from homeassistant.const import CONF_HOST, CONF_TOKEN
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import (
    CoordinatorEntity,
    DataUpdateCoordinator,
    UpdateFailed,
)

COLOR_TO_EMOJI = {
    "BLEU": "🔵",
    "BLANC": "⚪",
    "ROUGE": "🔴",
}

from .const import DOMAIN, DATA_KEY_COORDINATOR

_LOGGER = logging.getLogger(__name__)

async def async_setup_entry(hass: HomeAssistant, entry, async_add_entities: AddEntitiesCallback) -> bool:
    """Set up the TempoIA sensors from a config entry."""
    # Main coordinator for predictions
    coordinator = hass.data[DOMAIN][entry.entry_id][DATA_KEY_COORDINATOR]
    # Separate coordinator for tariffs
    tariff_coordinator = TariffDataUpdateCoordinator(hass, entry.data.get(CONF_HOST, ''), entry.data.get(CONF_TOKEN), entry.options.get('scan_interval', 10))
    # Create day prediction sensors
    day_sensors = [TempoiaPredictionSensor(coordinator, day_index=i) for i in range(14)]
    # Create tariff sensors using the tariff coordinator
    tariff_keys = ["bleuHC", "bleuHP", "blancHC", "blancHP", "rougeHC", "rougeHP"]
    tariff_sensors = [TariffSensor(tariff_coordinator, key) for key in tariff_keys]
    # Add all sensors
    async_add_entities(day_sensors + tariff_sensors, True)
    return True

class TempoiaDataUpdateCoordinator(DataUpdateCoordinator):
    """Class to fetch data from the TempoIA API."""

    def __init__(self, hass: HomeAssistant, host: str, token: str | None, scan_interval_min: int):
        super().__init__(
            hass,
            _LOGGER,
            name="TempoIA Prediction",
            update_interval=timedelta(minutes=scan_interval_min),
        )
        self.host = host.rstrip('/')
        self.token = token

    async def _async_update_data(self) -> dict:
        # Fetch prediction data
        predict_url = f"{self.host}/predict?days=14"
        headers: dict[str, str] = {}
        if self.token:
            headers["X-API-Token"] = self.token
        session: ClientSession = async_get_clientsession(self.hass)
        async with session.get(predict_url, headers=headers) as resp:
            if resp.status != 200:
                raise UpdateFailed(f"Error fetching prediction: {resp.status}")
            prediction_data = await resp.json()
            _LOGGER.debug("Received prediction data from API: %s", prediction_data)
        # Fetch tariff data
        tarifs_url = "https://www.api-couleur-tempo.fr/api/tarifs"
        async with session.get(tarifs_url, headers=headers) as resp:
            if resp.status != 200:
                raise UpdateFailed(f"Error fetching tarifs: {resp.status}")
            tarifs_data = await resp.json()
            _LOGGER.debug("Received tarifs data from API: %s", tarifs_data)
        # Combine both datasets
        combined = {
            "predictions": prediction_data.get("predictions", []),
            "tarifs": tarifs_data,
        }
        return combined

class TempoiaPredictionSensor(CoordinatorEntity, SensorEntity):
    """Sensor that shows the next‑day prediction from TempoIA."""

    _attr_icon = "mdi:weather-cloudy"

    def __init__(self, coordinator: TempoiaDataUpdateCoordinator, day_index: int):
        """Initialize the sensor."""
        super().__init__(coordinator)
        self.day_index = day_index
        self._attr_unique_id = f"tempoia_prediction_day_{day_index + 1}"
        self._attr_name = f"TempoIA Jour {day_index + 1}"
        # Initialize attributes
        self._attr_native_value = None
        self._attr_available = False
        self._attr_extra_state_attributes = {}

    @property
    def available(self) -> bool:
        """Return if entity is available."""
        return self._attr_available

    def _handle_coordinator_update(self) -> None:
        """Handle updated data from the coordinator."""
        _LOGGER.debug("Updating sensor for day %s", self.day_index)
        
        if not self.coordinator.data:
            _LOGGER.warning("No coordinator data available")
            self._attr_available = False
            self._attr_native_value = None
            self.async_write_ha_state()
            return

        predictions = self.coordinator.data.get("predictions", [])
        _LOGGER.debug("Found %s predictions", len(predictions))
        
        if self.day_index >= len(predictions):
            _LOGGER.warning("Day index %s out of range (max: %s)", self.day_index, len(predictions)-1)
            self._attr_available = False
            self._attr_native_value = None
            self.async_write_ha_state()
            return

        day_data = predictions[self.day_index]
        _LOGGER.debug("Day %s data: %s", self.day_index, day_data)
        
        # Construire le dictionnaire de probabilités à partir de la structure plate
        probabilities = {
            key: value for key, value in day_data.items() if key.upper() in ["BLEU", "BLANC", "ROUGE"]
        }
        _LOGGER.debug("Probabilities for day %s: %s", self.day_index, probabilities)

        if not probabilities or not isinstance(probabilities, dict):
            _LOGGER.warning("No valid probabilities for day %s", self.day_index)
            self._attr_available = False
            self._attr_native_value = None
            self.async_write_ha_state()
            return

        # Set the main state value (color with highest probability)
        if probabilities:
            maxvalue = max(probabilities.items(), key=lambda x: x[1])[0]
            self._attr_native_value = COLOR_TO_EMOJI.get(maxvalue, "❓")
            _LOGGER.debug("Set native value to: %s", self._attr_native_value)
        else:
            self._attr_native_value = None

        # Update attributes
        jour_semaine = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
        try:
            prediction_date = date.fromisoformat(day_data["date"])
            jour = jour_semaine[prediction_date.weekday()]
        except (KeyError, ValueError):
            _LOGGER.warning("Invalid date format for day %s: %s", self.day_index, day_data.get("date"))
            jour = "Inconnu"

        self._attr_extra_state_attributes = {
            "date": day_data.get("date"),
            "jour": jour,
            "proba_bleu": probabilities.get("BLEU"),
            "proba_blanc": probabilities.get("BLANC"),
            "proba_rouge": probabilities.get("ROUGE"),
            # Tariff information (same for all days)
            "tarif_bleu_hc": self.coordinator.data.get("tarifs", {}).get("bleuHC"),
            "tarif_bleu_hp": self.coordinator.data.get("tarifs", {}).get("bleuHP"),
            "tarif_blanc_hc": self.coordinator.data.get("tarifs", {}).get("blancHC"),
            "tarif_blanc_hp": self.coordinator.data.get("tarifs", {}).get("blancHP"),
            "tarif_rouge_hc": self.coordinator.data.get("tarifs", {}).get("rougeHC"),
            "tarif_rouge_hp": self.coordinator.data.get("tarifs", {}).get("rougeHP"),
            "tarif_date_debut": self.coordinator.data.get("tarifs", {}).get("dateDebut"),
        }

        self._attr_available = True
        _LOGGER.debug("Sensor update completed for day %s", self.day_index)
        self.async_write_ha_state()

    async def async_added_to_hass(self) -> None:
        """When entity is added to hass."""
        await super().async_added_to_hass()
        # Trigger initial update
        self._handle_coordinator_update()


class TariffSensor(CoordinatorEntity, SensorEntity):
    """Sensor that exposes a single tariff value from the API."""

    _attr_icon = "mdi:currency-eur"
    _attr_unit_of_measurement = "€/kWh"

    def __init__(self, coordinator: DataUpdateCoordinator, tariff_key: str):
        super().__init__(coordinator)
        self.tariff_key = tariff_key
        self._attr_unique_id = f"tempoia_tarif_{tariff_key.lower()}"
        self._attr_name = f"Tarif {tariff_key}"
        self._attr_native_value = None
        self._attr_available = False

    def _handle_coordinator_update(self) -> None:
        """Update the tariff value when coordinator data changes."""
        tariffs = self.coordinator.data if self.coordinator.data else {}
        value = tariffs.get(self.tariff_key)
        if value is None:
            self._attr_available = False
            self._attr_native_value = None
        else:
            self._attr_available = True
            self._attr_native_value = value
        self.async_write_ha_state()

class TariffDataUpdateCoordinator(DataUpdateCoordinator):
    """Coordinator for fetching tariff data only."""

    def __init__(self, hass: HomeAssistant, host: str, token: str | None, scan_interval_min: int):
        super().__init__(
            hass,
            _LOGGER,
            name="TempoIA Tariffs",
            update_interval=timedelta(minutes=scan_interval_min),
        )
        self.host = host.rstrip('/')
        self.token = token

    async def _async_update_data(self) -> dict:
        headers: dict[str, str] = {}
        if self.token:
            headers["X-API-Token"] = self.token
        session: ClientSession = async_get_clientsession(self.hass)
        tarifs_url = "https://www.api-couleur-tempo.fr/api/tarifs"
        async with session.get(tarifs_url, headers=headers) as resp:
            if resp.status != 200:
                raise UpdateFailed(f"Error fetching tarifs: {resp.status}")
            tarifs_data = await resp.json()
            _LOGGER.debug("Received tarifs data from API: %s", tarifs_data)
        return tarifs_data