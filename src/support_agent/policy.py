"""Shared capability policy for runtime checks and deterministic eval gates."""

from dataclasses import dataclass, field
import re
from typing import Any, Pattern


@dataclass(frozen=True)
class Capability:
    """One customer-facing category the configured agent can support."""

    id: str
    help_text: str


@dataclass(frozen=True)
class OutputRule:
    """One narrow rule tied to an observed unsupported model behavior."""

    label: str
    pattern: Pattern[str]
    allowed_by: frozenset[str]


DEFAULT_CAPABILITIES = (
    Capability("reservation_status", "check a reservation’s status"),
    Capability("room_recommendations", "suggest available Trailhead rooms"),
    Capability("early_risers", "help with the Early Risers Promotion"),
)

DEFAULT_RULES = (
    OutputRule(
        "unsupported_staff_action",
        re.compile(
            r"\b(?:i(?:['’]ll|\s+(?:will|can|am going to))|"
            r"we(?:['’]ll|\s+(?:will|can|are going to)))\s+"
            r"(?:escalate|contact|reach out to)\b|"
            r"\b(?:i(?:['’]ve|\s+have)|we(?:['’]ve|\s+have))\s+"
            r"(?:escalated|contacted|reached out to)\b",
            re.IGNORECASE,
        ),
        frozenset({"staff_escalation"}),
    ),
    OutputRule(
        "unsupported_reservation_change",
        re.compile(
            r"\b(?:i(?:['’]ll|\s+(?:will|can|am going to))|"
            r"we(?:['’]ll|\s+(?:will|can|are going to)))\s+"
            r"(?:process\s+(?:(?:a|the|your)\s+)?"
            r"(?:refunds?|returns?|cancellations?|"
            r"reservation(?![-\s]+status\b))|cancel|refund|book|reserve|upgrade|change)\b|"
            r"\b(?:i(?:['’]ve|\s+have)|we(?:['’]ve|\s+have))\s+"
            r"(?:processed\s+(?:(?:a|the|your)\s+)?"
            r"(?:refunds?|returns?|cancellations?|"
            r"reservation(?![-\s]+status\b))|cancelled|canceled|refunded|booked|reserved|upgraded|changed)\b",
            re.IGNORECASE,
        ),
        frozenset({"reservation_changes"}),
    ),
    OutputRule(
        "unsupported_outbound_message",
        re.compile(
            r"\b(?:i(?:['’]ll|\s+(?:will|can|am going to))|"
            r"we(?:['’]ll|\s+(?:will|can|are going to)))\s+"
            r"(?:follow up(?:\s+with)?|keep (?:you )?"
            r"(?:updated|posted)|notify|"
            r"send\s+(?:(?:you|them)\s+)?(?:(?:an?|the)\s+)?"
            r"(?:emails?|e-mails?|messages?|texts?|notifications?|updates?))\b|"
            r"\b(?:would you like me to|i can) keep an eye out\b|"
            r"\b(?:i(?:['’]ve|\s+have)|we(?:['’]ve|\s+have))\s+"
            r"(?:followed up|notified|"
            r"sent\s+(?:(?:you|them)\s+)?(?:(?:an?|the)\s+)?"
            r"(?:emails?|e-mails?|messages?|texts?|notifications?|updates?))\b",
            re.IGNORECASE,
        ),
        frozenset({"outbound_messages"}),
    ),
    OutputRule(
        "unsupported_room_inference",
        re.compile(
            r"\b(?:likely|probably|typically|should|could)\b"
            r"[^.!?]{0,100}\b(?:fit|fits|capacity|occupancy|square feet|extra beds?|"
            r"breakfast|pet policies|space|accommodate)\b",
            re.IGNORECASE,
        ),
        frozenset(),
    ),
)

_DIRECT_RECOMMENDATION = re.compile(
    r"\b(?:(?:i|we)\s+(?:would\s+)?(?:recommend|suggest)|"
    r"consider|look at|check out)\b",
    re.IGNORECASE,
)
_KNOWN_EXTERNAL_HOTELS = re.compile(
    r"\b(?:marriott|hilton|hyatt|"
    r"sheraton|westin)\b",
    re.IGNORECASE,
)
_POSITIVE_RECOMMENDATION = re.compile(
    r"\b(?:recommend(?:ed)?|suggest|ideal|solid pick|great choice|"
    r"well-suited|praised)\b",
    re.IGNORECASE,
)
_NEGATED_RECOMMENDATION = re.compile(
    r"\b(?:can(?:not|'t)|do not|don't|won't)\s+"
    r"(?:recommend|suggest|provide)\b[^.!?]*",
    re.IGNORECASE,
)


@dataclass
class PolicyContext:
    """Trusted tool evidence accumulated for one conversation."""

    room_names: set[str] = field(default_factory=set)

    def copy(self) -> "PolicyContext":
        return PolicyContext(room_names=set(self.room_names))

    def update(self, tool_name: str, result: dict[str, Any]) -> None:
        """Record evidence relevant to output policy."""

        if tool_name != "get_available_rooms" or not result.get("ok"):
            return
        rooms = result.get("rooms")
        if not isinstance(rooms, list):
            return
        self.room_names.update(
            room["name"]
            for room in rooms
            if isinstance(room, dict)
            and isinstance(room.get("name"), str)
            and room["name"].strip()
        )


@dataclass(frozen=True)
class CapabilityPolicy:
    """Configured capabilities plus narrow deterministic output checks."""

    capabilities: tuple[Capability, ...] = DEFAULT_CAPABILITIES
    rules: tuple[OutputRule, ...] = DEFAULT_RULES

    @property
    def enabled_ids(self) -> frozenset[str]:
        return frozenset(capability.id for capability in self.capabilities)

    @property
    def fallback(self) -> str:
        help_items = [capability.help_text for capability in self.capabilities]
        if not help_items:
            return "I can’t help with that here."
        if len(help_items) == 1:
            choices = help_items[0]
        else:
            choices = ", ".join(help_items[:-1]) + f", or {help_items[-1]}"
        return f"I can’t help with that here. I can {choices}."

    @property
    def rewrite_instructions(self) -> str:
        supported = "; ".join(
            capability.help_text for capability in self.capabilities
        )
        return (
            "Rewrite your immediately preceding customer-facing draft. Return "
            "only the replacement response, with no explanation of this "
            "correction.\n\nKeep supported, tool-grounded help from the draft. "
            "Remove unsupported commitments, claimed actions, room details "
            "not present in tool evidence, and external room recommendations. "
            "If nothing supported remains, offer only: "
            f"{supported}."
        )

    def find_violations(
        self,
        text: str,
        context: PolicyContext,
    ) -> tuple[str, ...]:
        """Return labels for known high-risk output failures."""

        violations: list[str] = []
        enabled = self.enabled_ids
        for rule in self.rules:
            if rule.allowed_by & enabled:
                continue
            if rule.pattern.search(text):
                violations.append(rule.label)

        allowed_names = {
            name.casefold() for name in context.room_names if name.strip()
        }
        affirmative_text = _NEGATED_RECOMMENDATION.sub("", text)
        if _KNOWN_EXTERNAL_HOTELS.search(
            affirmative_text
        ) and _POSITIVE_RECOMMENDATION.search(affirmative_text):
            violations.append("off_catalog_recommendation")

        for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
            if not _DIRECT_RECOMMENDATION.search(sentence):
                continue
            normalized = sentence.casefold()
            has_allowed_room = any(name in normalized for name in allowed_names)
            if not has_allowed_room or _KNOWN_EXTERNAL_HOTELS.search(sentence):
                violations.append("off_catalog_recommendation")
                break
        return tuple(dict.fromkeys(violations))


DEFAULT_POLICY = CapabilityPolicy()
