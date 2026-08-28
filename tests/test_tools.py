"""Unit tests for deterministic business rules."""

from datetime import datetime
import json
from pathlib import Path

import pytest

from sierra_agent.tools import PACIFIC_TIME, HotelTools


ORDERS = [
    {
        "Email": "hiker@example.com",
        "ReservationNumber": "#H001",
        "Status": "confirmed",
        "CheckInDate": "2026-11-10",
    }
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


def test_reservation_lookup_normalizes_both_identifiers(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)

    result = tools.lookup_reservation(" HIKER@example.com ", "h001")

    assert result == {
        "ok": True,
        "found": True,
        "status": "confirmed",
        "check_in_date": (
            "2026-11-10"
        ),
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

    result = tools.create_early_risers_code("hiker@example.com")

    assert result["eligible"] is eligible
    assert ("code" in result) is eligible


def test_promotion_code_is_stable_for_normalized_email_and_date(
    tmp_path: Path,
) -> None:
    current_time = datetime(2026, 8, 27, 9, 0, tzinfo=PACIFIC_TIME)
    tools = make_tools(tmp_path, current_time)

    first = tools.create_early_risers_code("HIKER@example.com")
    repeated = tools.create_early_risers_code(" hiker@example.com ")
    other_user = tools.create_early_risers_code("climber@example.com")

    assert first["code"] == repeated["code"]
    assert first["code"] != other_user["code"]
    assert first["code"].startswith("EARLY-")


def test_invalid_email_never_creates_a_code(tmp_path: Path) -> None:
    current_time = datetime(2026, 8, 27, 9, 0, tzinfo=PACIFIC_TIME)
    tools = make_tools(tmp_path, current_time)

    result = tools.create_early_risers_code("not-an-email")

    assert result == {"ok": False, "error": "invalid_email"}


def test_dispatch_rejects_unknown_or_malformed_calls(tmp_path: Path) -> None:
    tools = make_tools(tmp_path)

    assert tools.execute("missing_tool", "{}")["error"] == "unknown_tool"
    assert tools.execute("lookup_reservation", "not-json")["error"] == (
        "invalid_tool_arguments"
    )
    invalid_types = tools.execute(
        "lookup_reservation",
        '{"email":["not","a","string"],"reservation_number":1}',
    )
    assert invalid_types["error"] == "invalid_reservation_details"
