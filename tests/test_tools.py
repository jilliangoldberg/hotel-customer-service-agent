"""Unit tests for deterministic business rules."""

from datetime import datetime
import json
from pathlib import Path

import pytest

from support_agent.tools import PACIFIC_TIME, HotelTools


ORDERS = [
    {
        "GuestName": "Hotel Guest",
        "Email": "hiker@example.com",
        "ReservationNumber": "#H001",
        "RoomsReserved": ["ROOM1"],
        "Status": "confirmed",
        "CheckInDate": "2026-11-10",
    },
    {
        "GuestName": "Returning Guest",
        "Email": "camp@example.com",
        "ReservationNumber": "#H002",
        "RoomsReserved": ["ROOM1"],
        "Status": "checked-in",
        "CheckInDate": None,
    },
]

PRODUCTS = [
    {
        "RoomName": "Garden King Room",
        "RoomTypeID": "ROOM1",
        "AvailableRooms": 2,
        "Description": "A king room with a garden view.",
        "Tags": ["Travel"],
    },
    {
        "RoomName": "Unavailable Twin Room",
        "RoomTypeID": "ROOM0",
        "AvailableRooms": 0,
        "Description": "A twin room with a courtyard view.",
        "Tags": ["Travel"],
    },
]


def make_tools(
    tmp_path: Path,
    current_time: datetime | None = None,
) -> HotelTools:
    (tmp_path / "guest_reservations.json").write_text(
        json.dumps(ORDERS),
        encoding="utf-8",
    )
    (tmp_path / "room_catalog.json").write_text(
        json.dumps(PRODUCTS),
        encoding="utf-8",
    )
    now = (lambda: current_time) if current_time is not None else None
    return HotelTools(tmp_path, "a-test-secret-value", now=now)


def open_window_token(tools: HotelTools) -> str:
    result = tools.check_early_risers_window()
    assert result["available"] is True
    return str(result["window_token"])


def test_reservation_lookup_normalizes_both_identifiers(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)

    result = tools.lookup_reservation(" HIKER@example.com ", " h-001. ")

    assert result == {
        "ok": True,
        "found": True,
        "reservation_number": "#H001",
        "guest_name": "Hotel Guest",
        "status": "confirmed",
        "rooms": ["Garden King Room"],
        "check_in_date": (
            "2026-11-10"
        ),
    }


def test_reservation_lookup_confirms_details_without_tracking(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)

    result = tools.lookup_reservation("camp@example.com", "#H002")

    assert result == {
        "ok": True,
        "found": True,
        "reservation_number": "#H002",
        "guest_name": "Returning Guest",
        "status": "checked-in",
        "rooms": ["Garden King Room"],
        "check_in_date": None,
    }


def test_reservation_lookup_does_not_reveal_which_identifier_failed(
    tmp_path: Path,
) -> None:
    tools = make_tools(tmp_path)

    wrong_email = tools.lookup_reservation("other@example.com", "#H001")
    wrong_reservation = tools.lookup_reservation("hiker@example.com", "#H999")

    assert wrong_email == {"ok": True, "found": False}
    assert wrong_reservation == {"ok": True, "found": False}


def test_available_rooms_exclude_zero_availability(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)

    result = tools.get_available_rooms()

    assert result["ok"] is True
    assert [room["room_type_id"] for room in result["rooms"]] == ["ROOM1"]


@pytest.mark.parametrize(
    ("hour", "minute", "available"),
    [
        (7, 30, False),
        (9, 0, True),
    ],
)
def test_promotion_window_can_be_checked_without_email(
    tmp_path: Path,
    hour: int,
    minute: int,
    available: bool,
) -> None:
    current_time = datetime(2026, 8, 27, hour, minute, tzinfo=PACIFIC_TIME)
    tools = make_tools(tmp_path, current_time)

    result = tools.check_early_risers_window()

    assert result["ok"] is True
    assert result["available"] is available
    assert result["timezone"] == "Pacific Time"


def test_tool_registry_has_unique_matching_schemas_and_handlers(
    tmp_path: Path,
) -> None:
    tools = make_tools(tmp_path)
    names = [spec.name for spec in tools.specs]

    assert len(names) == len(set(names))
    assert [definition["name"] for definition in tools.definitions] == names
    assert all(callable(spec.handler) for spec in tools.specs)
    assert all(spec.definition["strict"] is True for spec in tools.specs)


def test_rejects_malformed_catalog_at_startup(tmp_path: Path) -> None:
    malformed_rooms = [{**PRODUCTS[0], "AvailableRooms": "2"}]
    (tmp_path / "guest_reservations.json").write_text(
        json.dumps(ORDERS),
        encoding="utf-8",
    )
    (tmp_path / "room_catalog.json").write_text(
        json.dumps(malformed_rooms),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid AvailableRooms"):
        HotelTools(tmp_path, "a-test-secret-value")


def test_rejects_reservation_with_unknown_catalog_room_type_id(tmp_path: Path) -> None:
    reservations = [{**ORDERS[0], "RoomsReserved": ["MISSING"]}]
    (tmp_path / "guest_reservations.json").write_text(
        json.dumps(reservations),
        encoding="utf-8",
    )
    (tmp_path / "room_catalog.json").write_text(
        json.dumps(PRODUCTS),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown room RoomTypeID"):
        HotelTools(tmp_path, "a-test-secret-value")


@pytest.mark.parametrize(
    ("hour", "minute", "eligible"),
    [
        (7, 59, False),
        (8, 0, True),
        (9, 59, True),
        (10, 0, False),
    ],
)
def test_promotion_uses_exact_pacific_time_boundaries(
    tmp_path: Path,
    hour: int,
    minute: int,
    eligible: bool,
) -> None:
    current_time = datetime(2026, 8, 27, hour, minute, tzinfo=PACIFIC_TIME)
    tools = make_tools(tmp_path, current_time)

    window = tools.check_early_risers_window()
    token = str(window.get("window_token", "closed"))
    result = tools.create_early_risers_code("hiker@example.com", token)

    assert result["eligible"] is eligible
    assert ("code" in result) is eligible


def test_promotion_code_is_stable_for_normalized_email_and_date(
    tmp_path: Path,
) -> None:
    current_time = datetime(2026, 8, 27, 9, 0, tzinfo=PACIFIC_TIME)
    tools = make_tools(tmp_path, current_time)
    token = open_window_token(tools)

    first = tools.create_early_risers_code("HIKER@example.com", token)
    repeated = tools.create_early_risers_code(" hiker@example.com ", token)
    other_user = tools.create_early_risers_code("climber@example.com", token)

    assert first["code"] == repeated["code"]
    assert first["code"] != other_user["code"]
    assert first["code"].startswith("EARLY-")


def test_invalid_email_never_creates_a_code(tmp_path: Path) -> None:
    current_time = datetime(2026, 8, 27, 9, 0, tzinfo=PACIFIC_TIME)
    tools = make_tools(tmp_path, current_time)
    token = open_window_token(tools)

    result = tools.create_early_risers_code("not-an-email", token)

    assert result["ok"] is False
    assert result["error"] == "invalid_email"
    assert "email" in result["hint"].lower()
    assert "code" not in result


def test_promotion_code_requires_token_from_prior_window_check(
    tmp_path: Path,
) -> None:
    current_time = datetime(2026, 8, 27, 9, 0, tzinfo=PACIFIC_TIME)
    tools = make_tools(tmp_path, current_time)

    result = tools.create_early_risers_code(
        "hiker@example.com",
        "not-a-window-token",
    )

    assert result["ok"] is False
    assert result["error"] == "promotion_window_not_checked"
    assert "code" not in result


def test_dispatch_rejects_unknown_or_malformed_calls(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)

    unknown = tools.execute("missing_tool", {})
    assert unknown["error"] == "unknown_tool"
    assert "lookup_reservation" in unknown["hint"]

    malformed = tools.execute("lookup_reservation", "not-a-dict")
    assert malformed["error"] == "invalid_tool_arguments"
    assert "required fields" in malformed["hint"]

    invalid_types = tools.execute(
        "lookup_reservation",
        {"email": ["not", "a", "string"], "reservation_number": 1},
    )
    assert invalid_types["error"] == "invalid_reservation_details"
    assert "hint" in invalid_types
    assert "found" not in invalid_types
