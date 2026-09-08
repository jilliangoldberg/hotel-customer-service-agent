"""Deterministic tools available to the language model."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, time
import hashlib
import hmac
import json
from pathlib import Path
import re
from typing import Any, Protocol
from zoneinfo import ZoneInfo


PACIFIC_TIME = ZoneInfo("America/Los_Angeles")
EARLY_RISERS_START = time(8, 0)
EARLY_RISERS_END = time(10, 0)
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


@dataclass(frozen=True)
class ToolSpec:
    """A model-facing schema paired with its server-side implementation."""

    definition: dict[str, Any]
    handler: Callable[..., dict[str, Any]]

    @property
    def name(self) -> str:
        return str(self.definition["name"])


class ToolHost(Protocol):
    """The small tool interface required by the agent and eval harness."""

    @property
    def definitions(self) -> list[dict[str, Any]]: ...

    def execute(self, name: str, arguments: str) -> dict[str, Any]: ...


def _reject(error: str, hint: str) -> dict[str, Any]:
    """Return a failed tool result the model can act on."""

    return {"ok": False, "error": error, "hint": hint}


class DataError(ValueError):
    """Raised when a static dataset is missing or malformed."""


class HotelTools:
    """Load hotel data and execute the agent's approved tools."""

    def __init__(
        self,
        data_dir: Path,
        promotion_secret: str,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._reservations = self._load_records(
            data_dir / "guest_reservations.json",
            {
                "Email",
                "ReservationNumber",
                "Status",
                "CheckInDate",
                "RoomsReserved",
            },
        )
        self._rooms = self._load_records(
            data_dir / "room_catalog.json",
            {
                "RoomName",
                "RoomTypeID",
                "AvailableRooms",
                "Description",
                "Tags",
            },
        )
        self._validate_rooms()
        self._validate_reservations()
        self._promotion_secret = promotion_secret.encode("utf-8")
        self._now = now or (lambda: datetime.now(PACIFIC_TIME))
        self._specs = self._build_specs()

    def _build_specs(self) -> tuple[ToolSpec, ...]:
        """Register every schema beside the handler that implements it."""

        return (
            ToolSpec(
                definition={
                    "type": "function",
                    "name": "lookup_reservation",
                    "description": (
                        "Look up one reservation after the customer has given both "
                        "their email and reservation number. Call only when both values "
                        "are present. Pass them as typed; spacing, capitalization, "
                        "a missing #, and extra symbols are normalized in Python. "
                        "Do not call with only one field. A miss does not say "
                        "which identifier was wrong."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "email": {
                                "type": "string",
                                "description": (
                                    "The customer's complete email address."
                                ),
                            },
                            "reservation_number": {
                                "type": "string",
                                "description": (
                                    "The complete reservation number, such as #H001."
                                ),
                            },
                        },
                        "required": ["email", "reservation_number"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
                handler=self.lookup_reservation,
            ),
            ToolSpec(
                definition={
                    "type": "function",
                    "name": "get_available_rooms",
                    "description": (
                        "Return every available catalog item. Call before "
                        "recommending rooms. Recommend only items and details "
                        "in this result. Do not invent rooms, prices, or "
                        "availability. The full available list is returned because "
                        "the sample catalog is small."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
                handler=self.get_available_rooms,
            ),
            ToolSpec(
                definition={
                    "type": "function",
                    "name": "check_early_risers_window",
                    "description": (
                        "Check whether the Early Risers Promotion is currently "
                        "open, without collecting an email. Call this first when "
                        "a customer explicitly requests a code. If the window is "
                        "closed, explain the hours and do not ask for an email. "
                        "If it is open, then ask for the email needed to create "
                        "the code. Do not call for a purely informational question."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
                handler=self.check_early_risers_window,
            ),
            ToolSpec(
                definition={
                    "type": "function",
                    "name": "create_early_risers_code",
                    "description": (
                        "Create today's Early Risers code for one email. Call only "
                        "after check_early_risers_window returned available=true, "
                        "and pass the window_token from that result. The customer "
                        "must have explicitly requested a code and given an email. "
                        "Do not call for an informational question. Python verifies "
                        "the token and window again before building the code."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "email": {
                                "type": "string",
                                "description": (
                                    "The customer's complete email address."
                                ),
                            },
                            "window_token": {
                                "type": "string",
                                "description": (
                                    "The opaque window_token returned by the "
                                    "successful window check."
                                ),
                            },
                        },
                        "required": ["email", "window_token"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
                handler=self.create_early_risers_code,
            ),
        )

    @property
    def specs(self) -> tuple[ToolSpec, ...]:
        """Return the registered schemas and handlers for inspection."""

        return self._specs

    @property
    def definitions(self) -> list[dict[str, Any]]:
        """Return the OpenAI schemas for all registered tools."""

        return [spec.definition for spec in self._specs]

    @staticmethod
    def _load_records(path: Path, required_fields: set[str]) -> list[dict[str, Any]]:
        """Load a JSON array and fail early when required fields are absent."""

        try:
            records = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise DataError(f"Missing dataset: {path.name}") from error
        except json.JSONDecodeError as error:
            raise DataError(f"Invalid JSON in {path.name}: {error.msg}") from error

        if not isinstance(records, list):
            raise DataError(f"{path.name} must contain a JSON array.")

        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise DataError(f"{path.name} record {index} must be an object.")
            missing = required_fields - record.keys()
            if missing:
                names = ", ".join(sorted(missing))
                raise DataError(f"{path.name} record {index} is missing: {names}.")

        return records

    @staticmethod
    def _has_text(value: Any) -> bool:
        return isinstance(value, str) and bool(value.strip())

    def _validate_reservations(self) -> None:
        """Validate reservation fields once, before customer requests arrive."""

        known_room_type_ids = {
            str(room["RoomTypeID"]).strip()
            for room in self._rooms
            if self._has_text(room.get("RoomTypeID"))
        }
        for index, reservation in enumerate(self._reservations):
            if self._normalize_email(reservation["Email"]) is None:
                raise DataError(
                    f"guest_reservations.json record {index} has an invalid Email."
                )
            if self._normalize_reservation_number(reservation["ReservationNumber"]) is None:
                raise DataError(
                    f"guest_reservations.json record {index} has an invalid ReservationNumber."
                )
            if not self._has_text(reservation["Status"]):
                raise DataError(
                    f"guest_reservations.json record {index} has an invalid Status."
                )
            check_in_date = reservation["CheckInDate"]
            if check_in_date is not None and not self._has_text(check_in_date):
                raise DataError(
                    f"guest_reservations.json record {index} has an invalid CheckInDate."
                )
            rooms = reservation["RoomsReserved"]
            if not isinstance(rooms, list) or not rooms or not all(
                self._has_text(room) for room in rooms
            ):
                raise DataError(
                    f"guest_reservations.json record {index} has invalid RoomsReserved."
                )
            unknown_room_type_ids = sorted(
                str(room).strip()
                for room in rooms
                if str(room).strip() not in known_room_type_ids
            )
            if unknown_room_type_ids:
                names = ", ".join(unknown_room_type_ids)
                raise DataError(
                    f"guest_reservations.json record {index} references unknown "
                    f"room RoomTypeID(s): {names}."
                )

    def _validate_rooms(self) -> None:
        """Validate catalog fields once, before recommendations are requested."""

        for index, room in enumerate(self._rooms):
            text_fields = ("RoomName", "RoomTypeID", "Description")
            if not all(self._has_text(room[field]) for field in text_fields):
                raise DataError(
                    f"room_catalog.json record {index} has an invalid text field."
                )
            availability = room["AvailableRooms"]
            if (
                not isinstance(availability, int)
                or isinstance(availability, bool)
                or availability < 0
            ):
                raise DataError(
                    f"room_catalog.json record {index} has an invalid AvailableRooms."
                )
            tags = room["Tags"]
            if not isinstance(tags, list) or not all(self._has_text(tag) for tag in tags):
                raise DataError(
                    f"room_catalog.json record {index} has invalid Tags."
                )

    @staticmethod
    def _normalize_email(email: Any) -> str | None:
        if not isinstance(email, str):
            return None
        normalized = email.strip().casefold()
        return normalized if EMAIL_PATTERN.fullmatch(normalized) else None

    @staticmethod
    def _normalize_reservation_number(reservation_number: Any) -> str | None:
        if not isinstance(reservation_number, str):
            return None
        compact = re.sub(r"\s+", "", reservation_number).upper()
        compact = re.sub(r"[^#A-Z0-9]", "", compact).lstrip("#")
        return f"#{compact}" if compact else None

    def _room_names_for_room_type_ids(self, room_type_ids: Any) -> list[str]:
        """Map ordered RoomTypeIDs to catalog names, keeping unknown RoomTypeIDs as-is."""

        if not isinstance(room_type_ids, list):
            return []
        names_by_room_type_id = {
            str(room["RoomTypeID"]): str(room["RoomName"])
            for room in self._rooms
        }
        names: list[str] = []
        for room_type_id in room_type_ids:
            key = str(room_type_id)
            names.append(names_by_room_type_id.get(key, key))
        return names

    def lookup_reservation(self, email: str, reservation_number: str) -> dict[str, Any]:
        """Find a reservation only when both customer identifiers match."""

        normalized_email = self._normalize_email(email)
        normalized_reservation = self._normalize_reservation_number(reservation_number)
        if normalized_email is None or normalized_reservation is None:
            return _reject(
                "invalid_reservation_details",
                "Ask for a complete email (name@domain.com) and the full reservation "
                "number. Do not guess which field was wrong.",
            )

        for reservation in self._reservations:
            saved_email = str(reservation["Email"]).strip().casefold()
            saved_reservation = self._normalize_reservation_number(str(reservation["ReservationNumber"]))
            if saved_email == normalized_email and saved_reservation == normalized_reservation:
                check_in_date = reservation.get("CheckInDate")
                guest_name = reservation.get("GuestName")
                return {
                    "ok": True,
                    "found": True,
                    "reservation_number": saved_reservation,
                    "guest_name": (
                        guest_name.strip()
                        if isinstance(guest_name, str) and guest_name.strip()
                        else None
                    ),
                    "status": str(reservation["Status"]),
                    "rooms": self._room_names_for_room_type_ids(
                        reservation.get("RoomsReserved")
                    ),
                    "check_in_date": check_in_date,
                }

        # A generic result avoids revealing which identifier exists.
        return {"ok": True, "found": False}

    def get_available_rooms(self) -> dict[str, Any]:
        """Return rooms with positive availability."""

        rooms: list[dict[str, Any]] = []
        for room in self._rooms:
            availability = room["AvailableRooms"]
            if not isinstance(availability, int):
                raise DataError("Room availability values must be integers.")
            if availability <= 0:
                continue
            rooms.append(
                {
                    "name": room["RoomName"],
                    "room_type_id": room["RoomTypeID"],
                    "availability": availability,
                    "description": room["Description"],
                    "tags": room["Tags"],
                }
            )

        return {"ok": True, "rooms": rooms}

    def _pacific_now(self) -> datetime:
        current = self._now()
        if current.tzinfo is None:
            raise ValueError("The clock must return a timezone-aware datetime.")
        return current.astimezone(PACIFIC_TIME)

    @staticmethod
    def _early_risers_is_open(pacific_now: datetime) -> bool:
        local_time = pacific_now.time().replace(tzinfo=None)
        return EARLY_RISERS_START <= local_time < EARLY_RISERS_END

    def check_early_risers_window(self) -> dict[str, Any]:
        """Check promotion timing without requesting customer information."""

        pacific_now = self._pacific_now()
        available = self._early_risers_is_open(pacific_now)
        result: dict[str, Any] = {
            "ok": True,
            "available": available,
            "hours": "8:00 AM to 10:00 AM",
            "timezone": "Pacific Time",
        }
        if available:
            result["window_token"] = self._promotion_window_token(pacific_now)
        else:
            result["reason"] = (
                "The Early Risers Promotion is available from 8:00 AM until "
                "10:00 AM Pacific Time."
            )
        return result

    def _promotion_window_token(self, pacific_now: datetime) -> str:
        """Return an opaque token proving an open-window check was observed."""

        message = f"early-risers-window:{pacific_now.date().isoformat()}".encode()
        return hmac.new(
            self._promotion_secret,
            message,
            hashlib.sha256,
        ).hexdigest()

    def create_early_risers_code(
        self,
        email: str,
        window_token: str,
    ) -> dict[str, Any]:
        """Return a stable daily code when the current Pacific time is eligible."""

        normalized_email = self._normalize_email(email)
        if normalized_email is None:
            return _reject(
                "invalid_email",
                "Ask for a complete email address like name@domain.com.",
            )

        pacific_now = self._pacific_now()

        if not self._early_risers_is_open(pacific_now):
            return {
                "ok": True,
                "eligible": False,
                "reason": (
                    "The Early Risers Promotion is available from 8:00 AM until "
                    "10:00 AM Pacific Time."
                ),
            }

        expected_token = self._promotion_window_token(pacific_now)
        if not isinstance(window_token, str) or not hmac.compare_digest(
            window_token,
            expected_token,
        ):
            return _reject(
                "promotion_window_not_checked",
                "Call check_early_risers_window first, then pass its window_token.",
            )

        message = f"{pacific_now.date().isoformat()}:{normalized_email}".encode()
        digest = hmac.new(
            self._promotion_secret,
            message,
            hashlib.sha256,
        ).hexdigest()
        return {
            "ok": True,
            "eligible": True,
            "code": f"EARLY-{digest[:12].upper()}",
        }

    def execute(self, name: str, arguments_json: str) -> dict[str, Any]:
        """Validate and dispatch one model-requested tool call."""

        try:
            arguments = json.loads(arguments_json)
        except json.JSONDecodeError:
            return _reject(
                "invalid_tool_arguments",
                "Call the tool again with the required fields from the schema.",
            )
        if not isinstance(arguments, dict):
            return _reject(
                "invalid_tool_arguments",
                "Call the tool again with the required fields from the schema.",
            )

        spec = next((item for item in self._specs if item.name == name), None)
        if spec is None:
            known = ", ".join(item.name for item in self._specs)
            return _reject(
                "unknown_tool",
                f"Use only one of the registered tools: {known}.",
            )

        try:
            return spec.handler(**arguments)
        except TypeError:
            return _reject(
                "invalid_tool_arguments",
                "Call the tool again with the required fields from the schema.",
            )
