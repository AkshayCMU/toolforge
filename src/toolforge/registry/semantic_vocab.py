"""Seed semantic type vocabulary for F1.5.

Two-tier taxonomy:
  CHAIN_ONLY     — values that must come from a prior tool output.
  USER_PROVIDED  — values the user can supply directly from intent or knowledge.
"""

from __future__ import annotations

CHAIN_ONLY_VOCAB: frozenset[str] = frozenset(
    {
        "hotel_id",
        "booking_id",
        "user_id",
        "order_id",
        "product_id",
        "account_id",
        "customer_id",
        "client_id",
        "venue_id",
        "operation_id",
        "market_id",
        "conversation_id",
        "channel_id",
        "device_id",
        "project_id",
        "tenant_id",
        "flight_id",
        "property_id",
        "media_id",
        "checkout_id",
        "buy_id",
        "deposit_id",
        "dest_id",
        "access_token",
    }
)

USER_PROVIDED_VOCAB: frozenset[str] = frozenset(
    {
        "search_query",
        "city_name",
        "country_code",
        "country_name",   # users say "United States" directly; not an opaque ID
        "company_name",   # users type company names directly from intent
        "city_code",      # analogous to country_code; user-supplied location shorthand
        "date",
        "datetime",
        "currency_code",
        "page_number",
        "limit",
        "email",
        "phone_number",
        "url",
        "price",
        "latitude",
        "longitude",
        "league_id",
        "season",
        "team_name",
        "language_code",
        "postal_code",
    }
)

# Types that the post-processor forces to null regardless of what the LLM proposed.
# Used when a type is too generic, arrives out-of-band, or is static config rather
# than a chained value.
NULL_OVERRIDE_TYPES: frozenset[str] = frozenset(
    {
        "public_key",   # static cryptographic config, not produced by a prior tool call
        "private_key",  # same — and exposing in graph edges is a security antipattern
        "otp",          # one-time passwords arrive out-of-band (SMS/email), not via tool chain
        "ticket",       # too generic: conflates event tickets, support tickets, booking refs
    }
)

ALL_VOCAB: frozenset[str] = CHAIN_ONLY_VOCAB | USER_PROVIDED_VOCAB
