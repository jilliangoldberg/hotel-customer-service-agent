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
            r"reservation(?![-\s]+status\b))|cancel|refund)\b|"
            r"\b(?:i(?:['’]ve|\s+have)|we(?:['’]ve|\s+have))\s+"
            r"(?:processed\s+(?:(?:a|the|your)\s+)?"
            r"(?:refunds?|returns?|cancellations?|"
            r"reservation(?![-\s]+status\b))|cancelled|canceled|refunded)\b",
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
            r"(?:\b(?:likely|probably|typically|should|could|may|might)\b"
            r"[^.!?]{0,100}\b(?:fit|fits|capacity|liters?|balconies?|"
            r"hydration|comfort|space|accommodate|hip\s+belt|weight|gps)\b|"
            r"\b(?:fit|fits|capacity|liters?|balconies?|hydration|comfort|"
            r"space|accommodate|hip\s+belt|weight|gps)\b[^.!?]{0,100}"
            r"\b(?:likely|probably|typically|should|could|may|might)\b)",
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
    r"\b(?:marriott|hilton(?:\s+atmos)?|hyatt(?:\s+aircontact)?|"
    r"sheraton(?:\s+baltoro)?|westin)\b",
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
_RESERVATION_STATUS_CLAIM = re.compile(
    r"\b(?:your|the)\s+reservation(?:\s+(#[a-z0-9-]+))?\s+"
    r"(?:is|was|has(?:\s+been)?)\s+(?:currently\s+)?"
    r"(checked-out|checked-in|shipped|in[-\s]+transit|cancelled|canceled|refunded)\b",
    re.IGNORECASE,
)
_CHECK_IN_DATE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}\b",
    re.IGNORECASE,
)
_PROMO_CODE = re.compile(r"\bEARLY-[A-F0-9]{12}\b", re.IGNORECASE)
_ROOM_DETAIL_CHECKS = (
    (
        "comfort",
        re.compile(r"\bcomfort(?:able)?\b", re.IGNORECASE),
        re.compile(r"\bcomfort(?:able)?\b", re.IGNORECASE),
    ),
    (
        "balconies",
        re.compile(r"\bbalconies?\b", re.IGNORECASE),
        re.compile(r"\bbalconies?\b", re.IGNORECASE),
    ),
    (
        "hip_belt",
        re.compile(r"\b(?:padded\s+)?hip\s+belts?\b", re.IGNORECASE),
        re.compile(r"\bhip\s+belts?\b", re.IGNORECASE),
    ),
    (
        "capacity",
        re.compile(r"\b(?:capacity|liters?)\b", re.IGNORECASE),
        re.compile(r"\b(?:capacity|liters?)\b", re.IGNORECASE),
    ),
    (
        "weight",
        re.compile(r"\b(?:lightweight|lighter|weight)\b", re.IGNORECASE),
        re.compile(r"\b(?:lightweight|lighter|weight)\b", re.IGNORECASE),
    ),
    (
        "gps",
        re.compile(r"\bgps\b", re.IGNORECASE),
        re.compile(r"\bgps\b", re.IGNORECASE),
    ),
)
_UNKNOWN_ROOM_DETAIL = re.compile(
    r"\b(?:not|isn['’]t|aren['’]t)\s+"
    r"(?:listed|stated|provided|specified|known|available)\b|"
    r"\b(?:unknown|unavailable)\b|"
    r"\b(?:can(?:not|['’]t)|do not|don['’]t)\s+"
    r"(?:confirm|verify|tell|have)\b",
    re.IGNORECASE,
)


def _normalized_status(value: str) -> str:
    return re.sub(r"[-\s]+", " ", value.strip().casefold())


def _room_aliases(name: str) -> set[str]:
    normalized = name.strip().casefold()
    aliases = {normalized}
    possessive = re.split(r"['’]s\s+", normalized, maxsplit=1)
    if len(possessive) == 2 and possessive[1]:
        aliases.add(possessive[1])
    return aliases


@dataclass
class PolicyContext:
    """Trusted tool evidence accumulated for one conversation."""

    room_names: set[str] = field(default_factory=set)
    room_evidence: dict[str, str] = field(default_factory=dict)
    reservation_statuses: dict[str, str] = field(default_factory=dict)
    check_in_dates: set[str] = field(default_factory=set)
    promo_codes: set[str] = field(default_factory=set)

    def copy(self) -> "PolicyContext":
        return PolicyContext(
            room_names=set(self.room_names),
            room_evidence=dict(self.room_evidence),
            reservation_statuses=dict(self.reservation_statuses),
            check_in_dates=set(self.check_in_dates),
            promo_codes=set(self.promo_codes),
        )

    def update(self, tool_name: str, result: dict[str, Any]) -> None:
        """Record evidence relevant to output policy."""

        if not result.get("ok"):
            return

        if tool_name == "lookup_reservation" and result.get("found") is True:
            reservation_number = result.get("reservation_number")
            status = result.get("status")
            if isinstance(reservation_number, str) and isinstance(status, str):
                self.reservation_statuses[reservation_number.strip().casefold()] = (
                    _normalized_status(status)
                )
            check_in_date = result.get("check_in_date")
            if isinstance(check_in_date, str) and check_in_date.strip():
                self.check_in_dates.add(check_in_date.strip())
            return

        if tool_name == "create_early_risers_code" and result.get("eligible") is True:
            code = result.get("code")
            if isinstance(code, str) and code.strip():
                self.promo_codes.add(code.strip().casefold())
            return

        if tool_name == "get_available_rooms":
            rooms = result.get("rooms")
            if not isinstance(rooms, list):
                return
            for room in rooms:
                if not isinstance(room, dict):
                    continue
                name = room.get("name")
                if not isinstance(name, str) or not name.strip():
                    continue
                self.room_names.add(name)
                details = [
                    name,
                    room.get("description"),
                    *(room.get("tags") or []),
                ]
                evidence = " ".join(
                    str(detail)
                    for detail in details
                    if isinstance(detail, str) and detail.strip()
                ).casefold()
                for alias in _room_aliases(name):
                    self.room_evidence[alias] = evidence


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

        for match in _RESERVATION_STATUS_CLAIM.finditer(text):
            claimed_number = match.group(1)
            claimed_status = _normalized_status(match.group(2))
            if claimed_number:
                trusted_status = context.reservation_statuses.get(
                    claimed_number.casefold()
                )
                if trusted_status != claimed_status:
                    violations.append("unsupported_reservation_status")
            elif claimed_status not in context.reservation_statuses.values():
                violations.append("unsupported_reservation_status")

        for link in _CHECK_IN_DATE.findall(text):
            if link.rstrip(".,!?") not in context.check_in_dates:
                violations.append("unsupported_check_in_date")

        for code in _PROMO_CODE.findall(text):
            if code.casefold() not in context.promo_codes:
                violations.append("unsupported_promo_code")

        normalized_text = text.casefold()
        for alias, evidence in context.room_evidence.items():
            if alias not in normalized_text:
                continue
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
                if _UNKNOWN_ROOM_DETAIL.search(sentence):
                    continue
                for label, claim_pattern, evidence_pattern in _ROOM_DETAIL_CHECKS:
                    if claim_pattern.search(sentence) and not evidence_pattern.search(
                        evidence
                    ):
                        violations.append(f"unsupported_room_detail:{label}")
        return tuple(dict.fromkeys(violations))


DEFAULT_POLICY = CapabilityPolicy()
