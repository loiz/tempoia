"""Config flow for TempoIA integration."""

from __future__ import annotations

import logging
from typing import Any

import aiohttp
import voluptuous as vol

from homeassistant import config_entries
from homeassistant.const import CONF_HOST, CONF_TOKEN
from homeassistant.core import callback
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.data_entry_flow import FlowResult

from .const import DOMAIN, CONF_SCAN_INTERVAL, DEFAULT_SCAN_INTERVAL

_LOGGER = logging.getLogger(__name__)

DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_HOST): str,
        vol.Optional(CONF_TOKEN): str,
        vol.Optional(CONF_SCAN_INTERVAL, default=DEFAULT_SCAN_INTERVAL): vol.All(
            vol.Coerce(int), vol.Range(min=5, max=1440)
        ),
    }
)


class TempoiaConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for TempoIA."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the initial step."""
        errors = {}

        if user_input is not None:
            # Validate connection
            host = user_input[CONF_HOST].rstrip("/")
            token = user_input.get(CONF_TOKEN)

            try:
                session = async_get_clientsession(self.hass)
                headers = {}
                if token:
                    headers["X-API-Token"] = token

                async with session.get(f"{host}/status", headers=headers, timeout=10) as resp:
                    if resp.status == 401:
                        errors["base"] = "invalid_auth"
                    elif resp.status != 200:
                        errors["base"] = "cannot_connect"
                    else:
                        # Connection successful
                        await self.async_set_unique_id(host)
                        self._abort_if_unique_id_configured()

                        return self.async_create_entry(
                            title="TempoIA", data=user_input
                        )

            except aiohttp.ClientError:
                errors["base"] = "cannot_connect"
            except Exception:  # pylint: disable=broad-except
                _LOGGER.exception("Unexpected exception")
                errors["base"] = "unknown"

        return self.async_show_form(
            step_id="user",
            data_schema=DATA_SCHEMA,
            errors=errors,
        )

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> TempoiaOptionsFlow:
        """Get the options flow for this handler."""
        return TempoiaOptionsFlow(config_entry)


class TempoiaOptionsFlow(config_entries.OptionsFlow):
    """Handle options flow for TempoIA."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        """Initialize options flow."""
        self.config_entry = config_entry

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Manage the options."""
        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)

        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(
                {
                    vol.Optional(
                        CONF_SCAN_INTERVAL,
                        default=self.config_entry.options.get(
                            CONF_SCAN_INTERVAL,
                            self.config_entry.data.get(CONF_SCAN_INTERVAL, DEFAULT_SCAN_INTERVAL),
                        ),
                    ): vol.All(vol.Coerce(int), vol.Range(min=5, max=1440)),
                }
            ),
        )
