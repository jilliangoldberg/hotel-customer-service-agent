"""Focused tests for shared capability policy."""

from sierra_agent.policy import (
    DEFAULT_CAPABILITIES,
    DEFAULT_POLICY,
    Capability,
    CapabilityPolicy,
    PolicyContext,
)


def test_context_records_only_trusted_catalog_results() -> None:
    context = PolicyContext()

    context.update(
        "get_available_rooms",
        {"ok": True, "rooms": [{"name": "Garden King Room"}, {"room_type_id": "NO-NAME"}]},
    )
    context.update(
        "lookup_reservation",
        {"ok": True, "rooms": [{"name": "Reservation Item"}]},
    )

    assert context.room_names == {"Garden King Room"}


def test_context_records_reservation_promo_and_room_evidence() -> None:
    context = PolicyContext()
    context.update(
        "lookup_reservation",
        {
            "ok": True,
            "found": True,
            "reservation_number": "#H001",
            "status": "confirmed",
            "check_in_date": "2026-11-10",
        },
    )
    context.update(
        "create_early_risers_code",
        {"ok": True, "eligible": True, "code": "EARLY-ABCDEF123456"},
    )
    context.update(
        "get_available_rooms",
        {
            "ok": True,
            "rooms": [
                {
                    "name": "Guide's Garden King Room",
                    "description": "A weatherproof pack.",
                    "tags": ["Travel"],
                }
            ],
        },
    )

    assert context.reservation_statuses == {"#h001": "confirmed"}
    assert "2026-11-10" in (
        context.check_in_dates
    )
    assert context.promo_codes == {"early-abcdef123456"}
    assert context.room_evidence["garden king room"] == (
        "guide's garden king room a weatherproof pack. travel"
    )


def test_policy_blocks_known_unsupported_commitments() -> None:
    context = PolicyContext()

    assert DEFAULT_POLICY.find_violations(
        "I'll escalate this to a manager.",
        context,
    ) == ("unsupported_staff_action",)
    assert DEFAULT_POLICY.find_violations(
        "I've processed your refund.",
        context,
    ) == ("unsupported_reservation_change",)
    assert DEFAULT_POLICY.find_violations(
        "I'll process your cancellation.",
        context,
    ) == ("unsupported_reservation_change",)
    assert DEFAULT_POLICY.find_violations(
        "I will keep you updated.",
        context,
    ) == ("unsupported_outbound_message",)
    assert DEFAULT_POLICY.find_violations(
        "I can send you an email.",
        context,
    ) == ("unsupported_outbound_message",)
    assert DEFAULT_POLICY.find_violations(
        "I'll keep you posted when new rooms arrive.",
        context,
    ) == ("unsupported_outbound_message",)
    assert DEFAULT_POLICY.find_violations(
        "Would you like me to keep an eye out for another room?",
        context,
    ) == ("unsupported_outbound_message",)
    assert DEFAULT_POLICY.find_violations(
        "I can keep an eye out for rooms with that feature.",
        context,
    ) == ("unsupported_outbound_message",)


def test_policy_allows_refusals_and_neutral_discussion() -> None:
    context = PolicyContext()

    assert DEFAULT_POLICY.find_violations(
        "I can't process refunds or contact a manager here.",
        context,
    ) == ()
    assert DEFAULT_POLICY.find_violations(
        "Refund requests are not available in this chat.",
        context,
    ) == ()
    assert DEFAULT_POLICY.find_violations(
        "I can process your reservation-status request.",
        context,
    ) == ()
    assert DEFAULT_POLICY.find_violations(
        "I can send you recommendations here.",
        context,
    ) == ()


def test_enabling_a_capability_disables_its_unsupported_action_rule() -> None:
    policy = CapabilityPolicy(
        capabilities=(
            *DEFAULT_CAPABILITIES,
            Capability("reservation_changes", "manage eligible reservation changes"),
        )
    )

    assert policy.find_violations(
        "I can cancel your reservation.",
        PolicyContext(),
    ) == ()


def test_mixed_external_recommendation_is_rejected() -> None:
    context = PolicyContext(room_names={"Garden King Room"})

    assert DEFAULT_POLICY.find_violations(
        "I recommend Garden King Room, and I also recommend Marriott.",
        context,
    ) == ("off_catalog_recommendation",)
    assert DEFAULT_POLICY.find_violations(
        "I recommend Garden King Room for this trip.",
        context,
    ) == ()


def test_observed_external_brand_phrasing_is_rejected() -> None:
    context = PolicyContext(room_names={"Garden King Room"})

    assert DEFAULT_POLICY.find_violations(
        "Some widely recommended brands are Hilton, Hyatt, and Sheraton.",
        context,
    ) == ("off_catalog_recommendation",)
    assert DEFAULT_POLICY.find_violations(
        "The Hilton Atmos AG 50 is a solid pick for this trip.",
        context,
    ) == ("off_catalog_recommendation",)
    assert DEFAULT_POLICY.find_violations(
        "I can't recommend Hilton because it is not in our catalog.",
        context,
    ) == ()


def test_hedged_room_spec_inferences_are_rejected() -> None:
    context = PolicyContext(room_names={"Garden King Room"})

    assert DEFAULT_POLICY.find_violations(
        "Garden King Room likely fits your four-guest capacity needs.",
        context,
    ) == ("unsupported_room_inference",)
    assert DEFAULT_POLICY.find_violations(
        "The catalog says Garden King Room has ample storage.",
        context,
    ) == ()


def test_unstated_room_details_are_rejected_against_tool_evidence() -> None:
    context = PolicyContext()
    context.update(
        "get_available_rooms",
        {
            "ok": True,
            "rooms": [
                {
                    "name": "Garden King Room",
                    "description": "Ample storage and weatherproof materials.",
                    "tags": ["Room", "Travel"],
                }
            ],
        },
    )

    assert DEFAULT_POLICY.find_violations(
        "The Garden King Room matches your comfort preference.",
        context,
    ) == ("unsupported_room_detail:comfort",)
    assert DEFAULT_POLICY.find_violations(
        "I recommend Garden King Room. "
        "It is comfortable for your trip.",
        context,
    ) == ("unsupported_room_detail:comfort",)
    assert DEFAULT_POLICY.find_violations(
        "Use any padded hip belt it may have.",
        context,
    ) == ("unsupported_room_inference",)
    assert DEFAULT_POLICY.find_violations(
        "The Garden King Room has ample storage and weatherproof materials.",
        context,
    ) == ()
    assert DEFAULT_POLICY.find_violations(
        "The Garden King Room's capacity and weight are not listed, "
        "so I can't confirm those details.",
        context,
    ) == ()


def test_reservation_tracking_and_promo_claims_require_matching_tool_evidence() -> None:
    empty = PolicyContext()

    assert DEFAULT_POLICY.find_violations(
        "Your reservation #H999 has shipped.",
        empty,
    ) == ("unsupported_reservation_status",)
    assert DEFAULT_POLICY.find_violations(
        "Your code is EARLY-ABCDEF123456.",
        empty,
    ) == ("unsupported_promo_code",)
    assert DEFAULT_POLICY.find_violations(
        "Track it at 2026-11-10.",
        empty,
    ) == ("unsupported_check_in_date",)

    grounded = PolicyContext()
    grounded.update(
        "lookup_reservation",
        {
            "ok": True,
            "found": True,
            "reservation_number": "#H001",
            "status": "checked-out",
            "check_in_date": "2026-11-10",
        },
    )
    grounded.update(
        "create_early_risers_code",
        {"ok": True, "eligible": True, "code": "EARLY-ABCDEF123456"},
    )
    assert DEFAULT_POLICY.find_violations(
        "Your reservation #H001 is checked-out. "
        "Track it at 2026-11-10. "
        "Your code is EARLY-ABCDEF123456.",
        grounded,
    ) == ()
