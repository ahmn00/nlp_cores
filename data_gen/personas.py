import json
import re
from dataclasses import dataclass, field

from openai import OpenAI

from .config import CONFIG
from .events import EVENT_TYPES


ARCHETYPE_SEEDS = [
    {
        "archetype": "senior_tax_partner",
        "org_type": "large law firm",
        "description": "Senior Tax Partner at an Am Law 100 firm, 20+ years experience, leads complex cross-border tax matters and M&A transactions",
    },
    {
        "archetype": "junior_tax_associate",
        "org_type": "large law firm",
        "description": "Junior Tax Associate (2-4 years PQE) at a large law firm, supports partners on research and document drafting",
    },
    {
        "archetype": "inhouse_tax_counsel",
        "org_type": "Fortune 500 corporation",
        "description": "In-house Tax Counsel at a Fortune 500 company, manages global tax compliance, transfer pricing, and tax planning",
    },
    {
        "archetype": "tax_research_analyst",
        "org_type": "financial research firm",
        "description": "Tax Research Analyst at a Bloomberg subscriber, tracks legislative changes and writes research reports on tax policy",
    },
    {
        "archetype": "ma_tax_specialist",
        "org_type": "investment bank",
        "description": "M&A Tax Specialist at a bulge-bracket investment bank, structures deals to optimize tax outcomes in transactions",
    },
    {
        "archetype": "transfer_pricing_specialist",
        "org_type": "Big 4 accounting firm",
        "description": "Transfer Pricing Specialist at a Big 4 firm, advises multinationals on intercompany pricing and OECD BEPS compliance",
    },
    {
        "archetype": "tax_technology_manager",
        "org_type": "consulting firm",
        "description": "Tax Technology Manager at a consulting firm, implements tax software and automation solutions for corporate clients",
    },
    {
        "archetype": "compliance_officer",
        "org_type": "financial institution",
        "description": "Tax Compliance Officer at a major bank, oversees regulatory filings, monitors tax law changes, and manages audit risk",
    },
    {
        "archetype": "tax_litigator",
        "org_type": "boutique litigation firm",
        "description": "Tax Litigator at a boutique firm specializing in IRS disputes, Tax Court cases, and appellate tax litigation",
    },
    {
        "archetype": "estate_planning_attorney",
        "org_type": "private wealth firm",
        "description": "Estate Planning Attorney at a private wealth management firm, advises ultra-high-net-worth clients on trusts, estates, and gift tax",
    },
]


ACTIVITY_DEFAULTS = CONFIG.activity_defaults
ARCHETYPE_ACTIVITY_HINTS = CONFIG.archetype_activity_hints


@dataclass
class Persona:
    archetype: str
    org_type: str
    name: str
    firm: str
    city: str
    years_experience: int
    primary_tools: list[str]
    behavior_traits: dict
    search_topics: list[str]
    event_weights: dict[str, float]
    activity_level: str = "regular"
    days_active_per_week: int = 4
    avg_sessions_per_active_day: int = 2


def generate_personas(
    client: OpenAI,
    num_personas: int = 10,
    model: str = "gpt-4o-mini",
    distinct_only: bool = False,
) -> list[Persona]:
    if distinct_only and num_personas > len(ARCHETYPE_SEEDS):
        print(f"  [distinct_personas_only=True] capping {num_personas} -> {len(ARCHETYPE_SEEDS)}")
        num_personas = len(ARCHETYPE_SEEDS)

    personas = []
    for i in range(num_personas):
        seed = ARCHETYPE_SEEDS[i % len(ARCHETYPE_SEEDS)]
        instance = i // len(ARCHETYPE_SEEDS)
        print(f"  Generating persona {i+1}/{num_personas}: {seed['archetype']} (instance {instance + 1})...")
        persona = _generate_single_persona(client, seed, model, instance=instance)
        personas.append(persona)
    return personas


def _generate_single_persona(
    client: OpenAI,
    seed: dict,
    model: str,
    instance: int = 0,
) -> Persona:
    event_types_str = ", ".join(EVENT_TYPES)

    activity_hint = ARCHETYPE_ACTIVITY_HINTS.get(seed["archetype"], "regular")
    variation_note = (
        f"\nThis is instance #{instance + 1} of this archetype — generate a DIFFERENT person "
        f"(different name, firm, city, years of experience) than other instances of the same archetype."
        if instance > 0 else ""
    )

    prompt = f"""You are generating a realistic synthetic user profile for a Bloomberg Terminal user in the tax and legal domain.

Archetype: {seed['archetype']}
Organization type: {seed['org_type']}
Description: {seed['description']}
Suggested activity level for this archetype: {activity_hint}{variation_note}

Return a JSON object with exactly these fields:
{{
  "name": "Full Name (realistic US professional name)",
  "firm": "Realistic firm/company name appropriate for the org type",
  "city": "US city where this professional likely works",
  "years_experience": <integer 1-35>,
  "primary_tools": ["list", "of", "Bloomberg", "function", "codes", "they", "use", "most"],
  "behavior_traits": {{
    "usage_pattern": "morning_heavy|evening_heavy|spread|burst",
    "session_length": "short|medium|long",
    "frequency": "daily|weekly|intensive"
  }},
  "search_topics": [
    "list of 12 realistic search queries this person would type into Bloomberg",
    "queries should reflect their specialty and day-to-day work",
    "mix of specific (case names, IRS ruling numbers) and broad topics"
  ],
  "event_weights": {{
    "session_start": <float>, "session_end": <float>,
    "screen_view": <float>, "search_query": <float>,
    "document_view": <float>, "document_download": <float>,
    "news_view": <float>, "alert_create": <float>,
    "alert_delete": <float>, "data_export": <float>,
    "function_run": <float>, "annotation_add": <float>,
    "contact_lookup": <float>
  }},
  "activity_level": "power_user|regular|occasional|inactive"
}}

For event_weights: values are relative frequencies (they will be normalized). Higher values mean this person does that action more. session_start and session_end should always be 1.0. Tailor other weights to the archetype's role.

For activity_level: pick the level that best matches this archetype's real-world Bloomberg usage. Most users should be "regular"; only the most intense data-driven roles (senior partners, M&A specialists, research analysts) are "power_user"; admin/support roles are "occasional"; rarely-active users are "inactive".

Return only valid JSON, no markdown fences, no explanation."""

    response = client.chat.completions.create(
        model=model,
        max_tokens=CONFIG.max_tokens_per_persona,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    data = json.loads(raw)

    weights = data.get("event_weights", {})
    total = sum(weights.values()) or 1.0
    normalized_weights = {k: v / total for k, v in weights.items()}

    for et in EVENT_TYPES:
        normalized_weights.setdefault(et, 0.01)

    activity_level = data.get("activity_level", activity_hint)
    if activity_level not in ACTIVITY_DEFAULTS:
        activity_level = activity_hint
    activity_cfg = ACTIVITY_DEFAULTS[activity_level]

    return Persona(
        archetype=seed["archetype"],
        org_type=seed["org_type"],
        name=data["name"],
        firm=data["firm"],
        city=data["city"],
        years_experience=int(data["years_experience"]),
        primary_tools=data.get("primary_tools", []),
        behavior_traits=data.get("behavior_traits", {}),
        search_topics=data.get("search_topics", []),
        event_weights=normalized_weights,
        activity_level=activity_level,
        days_active_per_week=activity_cfg["days_active_per_week"],
        avg_sessions_per_active_day=activity_cfg["avg_sessions_per_active_day"],
    )
