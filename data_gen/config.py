"""
Central configuration for data generation.

Edit the values below, then run:
    python -m data_gen.generate

CLI flags (--num-personas, --days, --output, --seed, --model) override
the matching fields here when provided.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GenerationConfig:
    # =============================================
    # Volume
    # =============================================
    num_personas: int = 10
    num_days: int = 7

    # If True, num_personas is capped at the number of unique archetypes (10).
    # If False, archetypes cycle (e.g. num_personas=50 -> 5 of each archetype).
    distinct_personas_only: bool = False

    # =============================================
    # Output
    # =============================================
    output_path: str = "data_gen/output/amplitude_events.csv"

    # =============================================
    # LLM
    # =============================================
    model: str = "gpt-4o-mini"
    max_tokens_per_persona: int = 1024

    # =============================================
    # Reproducibility
    # =============================================
    seed: int = 42

    # =============================================
    # Date window
    # =============================================
    # ISO date YYYY-MM-DD (last simulated day, inclusive).
    # None -> defaults to 2026-05-16.
    end_date: Optional[str] = None

    # =============================================
    # Activity-level templates
    # =============================================
    # Each activity level maps to a behavior profile.
    # days_active_per_week: how many days/week this user logs in
    # avg_sessions_per_active_day: mean number of sessions on an active day
    activity_defaults: dict = field(default_factory=lambda: {
        "power_user": {"days_active_per_week": 6, "avg_sessions_per_active_day": 4},
        "regular":    {"days_active_per_week": 4, "avg_sessions_per_active_day": 2},
        "occasional": {"days_active_per_week": 2, "avg_sessions_per_active_day": 1},
        "inactive":   {"days_active_per_week": 1, "avg_sessions_per_active_day": 1},
    })

    # =============================================
    # Default activity per archetype
    # =============================================
    # Used as a hint in the LLM prompt and as a fallback if the LLM
    # returns an invalid activity_level value.
    archetype_activity_hints: dict = field(default_factory=lambda: {
        "senior_tax_partner": "power_user",
        "junior_tax_associate": "regular",
        "inhouse_tax_counsel": "regular",
        "tax_research_analyst": "power_user",
        "ma_tax_specialist": "power_user",
        "transfer_pricing_specialist": "regular",
        "tax_technology_manager": "occasional",
        "compliance_officer": "regular",
        "tax_litigator": "regular",
        "estate_planning_attorney": "occasional",
    })

    # =============================================
    # Weekday / weekend pattern
    # =============================================
    # Multiplier on the daily active-probability for weekdays vs weekends.
    # Bloomberg is a work tool, so Mon-Fri >> Sat-Sun by default.
    weekday_active_multiplier: float = 1.5
    weekend_active_multiplier: float = 0.2

    # =============================================
    # Sessions per active day
    # =============================================
    # Std dev around persona.avg_sessions_per_active_day (gaussian draw).
    sessions_per_day_stddev: float = 1.0

    # =============================================
    # Inter-event delays (seconds, log-normal distribution)
    # =============================================
    # delay = clip(lognormal(mu, sigma), min_s, max_s)
    # Defaults: median ~33s, p95 ~175s — typical "think time" between clicks.
    inter_event_delay_mu: float = 3.5
    inter_event_delay_sigma: float = 1.0
    inter_event_delay_min_s: float = 5.0
    inter_event_delay_max_s: float = 600.0

    # =============================================
    # Session duration (used in session_end property)
    # =============================================
    session_duration_mu: float = 6.5
    session_duration_sigma: float = 1.0


CONFIG = GenerationConfig()
