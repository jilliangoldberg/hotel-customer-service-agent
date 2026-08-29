"""Deterministic tools available to the language model."""

from collections.abc import Callable
from datetime import datetime, time
import hashlib
import hmac
import json
from pathlib import Path
import re
from typing import Any
from zoneinfo import ZoneInfo


PACIFIC_TIME = ZoneInfo("America/Los_Angeles")
EARLY_RISERS_START = time(8, 0)
EARLY_RISERS_END = time(10, 0)
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "lookup_reservation",
        "description": (
            "Look up one reservation after the customer has supplied both their email "
            "address and reservation number. Spacing, capitalization, a missing #, and "
            "extra symbols are normalized automatically."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "email": {
                    "type": "string",
                    "description": "The customer's complete email address.",
                },
                "reservation_number": {
                    "type": "string",
                    "description": "The complete reservation number, such as #H001.",
                },
            },
            "required": ["email", "reservation_number"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "get_available_rooms",
        "description": (
            "Return the complete available Trailhead Hotel room catalog for "
            "grounded room recommendations."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "create_early_risers_code",
        "description": (
            "Check the current Pacific time and, when eligible, create today's "
            "Early Risers Promotion code for the supplied email."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "email": {
                    "type": "string",
                    "description": "The customer's complete email address.",
                }
            },
            "required": ["email"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


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
        self._validate_reservations()
        self._validate_rooms()
        self._promotion_secret = promotion_secret.encode("utf-8")
        self._now = now or (lambda: datetime.now(PACIFIC_TIME))

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
            return {"ok": False, "error": "invalid_reservation_details"}

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

    def create_early_risers_code(self, email: str) -> dict[str, Any]:
        """Return a stable daily code when the current Pacific time is eligible."""

        normalized_email = self._normalize_email(email)
        if normalized_email is None:
            return {"ok": False, "error": "invalid_email"}

        current = self._now()
        if current.tzinfo is None:
            raise ValueError("The clock must return a timezone-aware datetime.")
        pacific_now = current.astimezone(PACIFIC_TIME)
        local_time = pacific_now.time().replace(tzinfo=None)

        if not EARLY_RISERS_START <= local_time < EARLY_RISERS_END:
            return {
                "ok": True,
                "eligible": False,
                "reason": (
                    "The Early Risers Promotion is available from 8:00 AM until "
                    "10:00 AM Pacific Time."
                ),
            }

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
            return {"ok": False, "error": "invalid_tool_arguments"}
        if not isinstance(arguments, dict):
            return {"ok": False, "error": "invalid_tool_arguments"}

        handlers: dict[str, Callable[..., dict[str, Any]]] = {
            "lookup_reservation": self.lookup_reservation,
            "get_available_rooms": self.get_available_rooms,
            "create_early_risers_code": self.create_early_risers_code,
        }
        handler = handlers.get(name)
        if handler is None:
            return {"ok": False, "error": "unknown_tool"}

        try:
            return handler(**arguments)
        except TypeError:
            return {"ok": False, "error": "invalid_tool_arguments"}
