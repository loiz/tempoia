"""Calendar platform for TempoIA predictions."""
from __future__ import annotations

from datetime import date, datetime, timedelta

from homeassistant.components.calendar import (
    CalendarEntity,
    CalendarEvent,
)
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import DOMAIN, DATA_KEY_COORDINATOR


async def async_setup_entry(
    hass: HomeAssistant, entry, async_add_entities: AddEntitiesCallback
) -> None:
    """Set up the TempoIA calendar from a config entry."""
    coordinator = hass.data[DOMAIN][entry.entry_id][DATA_KEY_COORDINATOR]
    async_add_entities([TempoiaCalendar(coordinator, entry.entry_id)])


class TempoiaCalendar(CoordinatorEntity, CalendarEntity):
    """The TempoIA calendar."""

    _attr_name = "TempoIA Forecast"

    def __init__(self, coordinator, entry_id: str):
        """Initialize the calendar."""
        super().__init__(coordinator)
        self._attr_unique_id = f"{entry_id}-forecast-calendar"
        self._event: CalendarEvent | None = None

    @property
    def event(self) -> CalendarEvent | None:
        """Return the next upcoming event."""
        # Pour ce calendrier, nous ne montrons que les événements dans la vue calendrier,
        # pas un "prochain événement".
        return None

    async def async_get_events(
        self, hass: HomeAssistant, start_date: datetime, end_date: datetime
    ) -> list[CalendarEvent]:
        """Return calendar events within a datetime range."""
        events: list[CalendarEvent] = []
        predictions = self.coordinator.data.get("predictions", [])

        for day_data in predictions:
            prediction_date = date.fromisoformat(day_data["date"])
            
            # Construire le dictionnaire de probabilités à partir de la structure plate
            probabilities = {
                key: value for key, value in day_data.items() if key.upper() in ["BLEU", "BLANC", "ROUGE"]
            }

            if probabilities:
                # La couleur est celle avec la plus haute probabilité
                summary = max(probabilities, key=probabilities.get)
                event = CalendarEvent(
                    summary=f"Jour {summary}",
                    start=prediction_date,
                    end=prediction_date + timedelta(days=1),
                )
                events.append(event)

        return events