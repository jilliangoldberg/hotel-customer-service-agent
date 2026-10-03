"""Focused tests for shared capability policy."""

from support_agent.policy import (
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