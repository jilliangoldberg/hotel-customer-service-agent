"""Hotel-specific dataset validation."""
import json
import pytest
from support_agent.config import PROJECT_ROOT
from support_agent.tools import DataError, HotelTools


def test_rejects_invalid_check_in_date(tmp_path):
    data = json.loads((PROJECT_ROOT / 'data/guest_reservations.json').read_text())
    data[0]['CheckInDate'] = '2026-02-30'
    (tmp_path / 'guest_reservations.json').write_text(json.dumps(data))
    (tmp_path / 'room_catalog.json').write_bytes((PROJECT_ROOT / 'data/room_catalog.json').read_bytes())
    with pytest.raises(DataError, match='invalid CheckInDate'):
        HotelTools(tmp_path, 'a-valid-promotion-secret')
