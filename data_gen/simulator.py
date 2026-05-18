"""
Day-based, workflow-driven Bloomberg interaction simulator.

Each session executes one named workflow from `workflows.py`. Within a session,
events share a context dict (doc_id, doc_type, doc_title, query, function_code)
so the sequence is semantically coherent (view -> annotate -> download all
reference the same document). The `workflow_type` is written as a top-level
column in the CSV to serve as a ground-truth label for LSTM/transformer training.
"""
import hashlib
import json
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from .config import CONFIG
from .personas import Persona
from .workflows import (
    WORKFLOWS,
    WorkflowDefinition,
    WorkflowStep,
    get_workflows_for_persona,
)


@dataclass
class EventRow:
    user_id: str
    amplitude_id: int
    event_type: str
    event_time: str
    session_id: int
    workflow_type: str
    device_type: str
    os_name: str
    platform: str
    country: str
    city: str
    event_properties: str
    user_properties: str
    sequence_number: int


def simulate_events(
    personas: list[Persona],
    num_days: int = 7,
    seed: int = 42,
    end_date: datetime | None = None,
) -> list[EventRow]:
    rng = random.Random(seed)
    if end_date is None:
        if CONFIG.end_date:
            end_date = datetime.fromisoformat(CONFIG.end_date).replace(tzinfo=timezone.utc)
        else:
            end_date = datetime(2026, 5, 16, tzinfo=timezone.utc)
    start_date = end_date - timedelta(days=num_days - 1)

    all_rows: list[EventRow] = []
    for idx, persona in enumerate(personas):
        workflows = get_workflows_for_persona(persona.archetype)
        if not workflows:
            workflows = WORKFLOWS
        rows = _simulate_persona(persona, idx, workflows, start_date, num_days, rng)
        print(f"  {persona.name:<28} ({persona.activity_level:<11}): {len(rows):>5} events across {_count_sessions(rows)} sessions")
        all_rows.extend(rows)

    all_rows.sort(key=lambda r: r.event_time)
    return all_rows


def _count_sessions(rows: list[EventRow]) -> int:
    return len({r.session_id for r in rows})


def _simulate_persona(
    persona: Persona,
    persona_idx: int,
    workflows: list[WorkflowDefinition],
    start_date: datetime,
    num_days: int,
    rng: random.Random,
) -> list[EventRow]:
    user_id = _make_user_id(persona_idx)
    amplitude_id = _make_amplitude_id(persona_idx)
    hour_weights = _get_hour_weights(persona.behavior_traits)

    base_daily_prob = persona.days_active_per_week / 7.0
    rows: list[EventRow] = []
    session_idx = 0

    for day_offset in range(num_days):
        day = start_date + timedelta(days=day_offset)
        is_weekday = day.weekday() < 5
        multiplier = CONFIG.weekday_active_multiplier if is_weekday else CONFIG.weekend_active_multiplier
        active_prob = min(1.0, base_daily_prob * multiplier)
        if rng.random() >= active_prob:
            continue

        num_sessions = max(1, int(rng.gauss(persona.avg_sessions_per_active_day, CONFIG.sessions_per_day_stddev)))
        for _ in range(num_sessions):
            workflow = _pick_workflow(workflows, persona.activity_level, rng)
            session_id = _make_session_id(user_id, session_idx)
            session_idx += 1
            session_rows = _simulate_session(
                persona=persona,
                user_id=user_id,
                amplitude_id=amplitude_id,
                workflow=workflow,
                session_id=session_id,
                day=day,
                hour_weights=hour_weights,
                rng=rng,
            )
            rows.extend(session_rows)

    return rows


def _pick_workflow(
    workflows: list[WorkflowDefinition],
    activity_level: str,
    rng: random.Random,
) -> WorkflowDefinition:
    if activity_level == "inactive":
        candidates = [w for w in workflows if w.length_class == "short"] or workflows
    elif activity_level == "occasional":
        candidates = [w for w in workflows if w.length_class in ("short", "medium")] or workflows
    else:
        candidates = workflows
    return rng.choice(candidates)


def _simulate_session(
    persona: Persona,
    user_id: str,
    amplitude_id: int,
    workflow: WorkflowDefinition,
    session_id: int,
    day: datetime,
    hour_weights: list[float],
    rng: random.Random,
) -> list[EventRow]:
    ctx = _make_session_context(persona, workflow, rng)
    start_dt = _pick_time_within_day(day, hour_weights, rng)
    current_dt = start_dt

    user_props = {
        "persona": persona.archetype,
        "org_type": persona.org_type,
        "firm_size": _infer_firm_size(persona.org_type),
        "years_experience": persona.years_experience,
        "activity_level": persona.activity_level,
        "city": persona.city,
    }
    user_props_json = json.dumps(user_props)

    expanded = _expand_workflow_steps(workflow, rng)
    full_sequence = ["session_start"] + expanded + ["session_end"]

    rows: list[EventRow] = []
    for seq_pos, event_type in enumerate(full_sequence):
        props = _generate_event_properties(event_type, persona, ctx, rng)
        row = EventRow(
            user_id=user_id,
            amplitude_id=amplitude_id,
            event_type=event_type,
            event_time=current_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            session_id=session_id,
            workflow_type=workflow.name,
            device_type="Desktop",
            os_name=rng.choice(["Windows", "Windows", "Windows", "macOS"]),
            platform="Bloomberg Terminal",
            country="United States",
            city=persona.city,
            event_properties=json.dumps(props),
            user_properties=user_props_json,
            sequence_number=seq_pos,
        )
        rows.append(row)
        current_dt = current_dt + timedelta(seconds=_inter_event_delay(rng))

    return rows


def _expand_workflow_steps(
    workflow: WorkflowDefinition, rng: random.Random
) -> list[str]:
    out: list[str] = []
    for step in workflow.steps:
        reps = rng.randint(step.repeat_min, step.repeat_max)
        out.extend([step.event_type] * reps)
    return out


def _make_session_context(
    persona: Persona, workflow: WorkflowDefinition, rng: random.Random
) -> dict:
    doc_type = workflow.primary_doc_type or rng.choice(
        ["IRS_ruling", "tax_regulation", "case_law", "guidance", "commentary"]
    )
    if doc_type == "case_law":
        doc_title = rng.choice(_CASE_LAW)
    elif doc_type == "IRS_ruling":
        doc_title = rng.choice(_IRS_RULINGS)
    elif doc_type == "treaty":
        doc_title = rng.choice(_TREATIES)
    elif doc_type == "news_article":
        doc_title = rng.choice(persona.search_topics[:8]) if persona.search_topics else "Tax policy update"
    else:
        doc_title = rng.choice(_TAX_REGS)

    function_code = workflow.primary_function
    if function_code is None:
        function_code = rng.choice(persona.primary_tools) if persona.primary_tools else rng.choice(_BLOOMBERG_FUNCTIONS)

    query = rng.choice(persona.search_topics) if persona.search_topics else doc_title

    return {
        "doc_id": f"DOC-{rng.randint(10000, 99999)}",
        "doc_type": doc_type,
        "doc_title": doc_title,
        "query": query,
        "function_code": function_code,
        "jurisdiction": rng.choice(_JURISDICTIONS),
    }


def _generate_event_properties(
    event_type: str, persona: Persona, ctx: dict, rng: random.Random
) -> dict:
    if event_type == "session_start":
        return {"login_method": rng.choice(["password", "sso", "biometric"])}

    if event_type == "session_end":
        return {
            "duration_seconds": int(rng.lognormvariate(CONFIG.session_duration_mu, CONFIG.session_duration_sigma)),
            "logout_type": rng.choice(["manual", "timeout", "idle"]),
        }

    if event_type == "screen_view":
        return {
            "function_code": ctx["function_code"],
            "previous_screen": rng.choice(_BLOOMBERG_FUNCTIONS),
        }

    if event_type == "search_query":
        return {
            "query_text": ctx["query"],
            "search_scope": _scope_for_doc_type(ctx["doc_type"], rng),
            "result_count": rng.randint(3, 150),
        }

    if event_type == "document_view":
        return {
            "doc_id": ctx["doc_id"],
            "doc_type": ctx["doc_type"],
            "doc_title": ctx["doc_title"],
            "jurisdiction": ctx["jurisdiction"],
        }

    if event_type == "document_download":
        return {
            "doc_id": ctx["doc_id"],
            "doc_type": ctx["doc_type"],
            "export_format": rng.choice(["PDF", "PDF", "PDF", "DOCX"]),
        }

    if event_type == "news_view":
        headline = ctx["doc_title"] if ctx["doc_type"] == "news_article" else rng.choice(persona.search_topics[:6]) if persona.search_topics else "Tax reform update"
        return {
            "story_id": f"NEWS-{rng.randint(100000, 999999)}",
            "headline": headline,
            "topic_tags": rng.sample(
                ["tax", "IRS", "OECD", "transfer_pricing", "M&A", "compliance", "regulation"],
                k=rng.randint(1, 3),
            ),
            "source": rng.choice(["Bloomberg Tax", "Bloomberg Law", "Tax Notes", "BNA", "Reuters"]),
        }

    if event_type == "alert_create":
        alert_query = ctx["query"] if rng.random() < 0.6 else rng.choice(_ALERT_QUERIES)
        return {
            "alert_type": rng.choice(["keyword", "company", "regulation", "case"]),
            "alert_query": alert_query,
            "frequency": rng.choice(["real_time", "daily", "weekly"]),
        }

    if event_type == "alert_delete":
        return {
            "alert_id": f"ALT-{rng.randint(1000, 9999)}",
            "alert_type": rng.choice(["keyword", "regulation"]),
        }

    if event_type == "data_export":
        return {
            "dataset_type": rng.choice(["tax_rates", "transfer_pricing", "treaty_table", "case_list", "regulation_index"]),
            "row_count": rng.randint(50, 5000),
            "export_format": rng.choice(["CSV", "Excel", "CSV"]),
            "date_range": "2023-01-01/2024-12-31",
        }

    if event_type == "function_run":
        return {
            "function_code": ctx["function_code"],
            "input_params": {
                "jurisdiction": rng.choice(["US", "EU", "UK"]),
                "year": rng.choice([2023, 2024, 2025]),
            },
        }

    if event_type == "annotation_add":
        return {
            "doc_id": ctx["doc_id"],
            "annotation_type": rng.choice(["highlight", "note", "bookmark", "tag"]),
            "note_text": rng.choice([
                f"See also {rng.choice(_IRS_RULINGS)}",
                "Check with partner",
                "Cite in brief",
                "Confirm with client",
                f"Compare to {rng.choice(_CASE_LAW)}",
            ]),
        }

    if event_type == "contact_lookup":
        return {
            "lookup_type": rng.choice(["attorney", "regulator", "tax_authority", "expert_witness"]),
            "query": rng.choice(["IRS Appeals Office", "DOJ Tax Division", "Treasury OTP", "OECD Secretariat"]),
        }

    return {}


def _scope_for_doc_type(doc_type: str, rng: random.Random) -> str:
    if doc_type == "case_law":
        return "case_law"
    if doc_type in ("tax_regulation", "treaty", "guidance"):
        return "regulation"
    if doc_type == "news_article":
        return "news"
    return rng.choice(["full_text", "document"])


def _make_user_id(idx: int) -> str:
    return f"usr_{idx:04d}"


def _make_amplitude_id(idx: int) -> int:
    return 4200000 + idx


def _make_session_id(user_id: str, session_idx: int) -> int:
    raw = f"{user_id}:{session_idx}"
    return int(hashlib.md5(raw.encode()).hexdigest()[:8], 16)


def _inter_event_delay(rng: random.Random) -> float:
    delay = rng.lognormvariate(mu=CONFIG.inter_event_delay_mu, sigma=CONFIG.inter_event_delay_sigma)
    return max(CONFIG.inter_event_delay_min_s, min(delay, CONFIG.inter_event_delay_max_s))


def _pick_time_within_day(
    day: datetime, hour_weights: list[float], rng: random.Random
) -> datetime:
    hours = list(range(24))
    hour = rng.choices(hours, weights=hour_weights, k=1)[0]
    minute = rng.randint(0, 59)
    second = rng.randint(0, 59)
    return day.replace(hour=hour, minute=minute, second=second)


def _get_hour_weights(behavior_traits: dict) -> list[float]:
    pattern = behavior_traits.get("usage_pattern", "spread")
    weights = [0.05] * 24
    if pattern == "morning_heavy":
        for h in range(7, 13):
            weights[h] = 1.0
        for h in range(13, 18):
            weights[h] = 0.4
    elif pattern == "evening_heavy":
        for h in range(17, 22):
            weights[h] = 1.0
        for h in range(9, 17):
            weights[h] = 0.5
    elif pattern == "burst":
        for h in (8, 9, 14, 15):
            weights[h] = 1.0
        for h in (10, 11, 12, 13, 16, 17):
            weights[h] = 0.3
    else:
        for h in range(8, 19):
            weights[h] = 0.8
    return weights


def _infer_firm_size(org_type: str) -> str:
    large = ["large law firm", "Fortune 500", "investment bank", "Big 4"]
    if any(l in org_type for l in large):
        return "large"
    if "boutique" in org_type or "private wealth" in org_type:
        return "small"
    return "medium"


_IRS_RULINGS = [
    "Rev. Rul. 2023-14", "Rev. Proc. 2024-01", "PLR 202412001", "CCA 202415010",
    "Notice 2024-55", "T.D. 9999", "Rev. Rul. 2022-08", "PLR 202308003",
    "Rev. Rul. 2024-22", "Notice 2023-63",
]
_CASE_LAW = [
    "Coca-Cola Co. v. Commissioner", "3M Co. v. Commissioner",
    "Medtronic Inc. v. Commissioner", "Altera Corp. v. Commissioner",
    "Whirlpool Corp. v. Commissioner", "Amazon.com Inc. v. Commissioner",
    "Mayo Foundation v. United States", "Chevron U.S.A. v. NRDC",
]
_TAX_REGS = [
    "Treas. Reg. § 1.482-1", "Treas. Reg. § 1.368-1", "Treas. Reg. § 1.351-1",
    "Treas. Reg. § 1.1001-1", "IRC § 382 regulations", "GILTI regulations § 951A",
    "BEAT regulations § 59A", "FDII regulations § 250", "Section 163(j) regs",
]
_TREATIES = [
    "US-UK Income Tax Treaty", "US-Germany Tax Treaty Protocol",
    "OECD Model Tax Convention", "US-Japan Tax Treaty",
    "US-Netherlands Tax Treaty", "OECD MLI Article 7",
]
_BLOOMBERG_FUNCTIONS = ["TAX", "BTAX", "NI", "LAW", "BLAW", "CRED", "CF", "MA", "BRIEF", "COURT", "REGN", "TRA", "TRAX"]
_JURISDICTIONS = ["US_federal", "US_federal", "US_federal", "US_state", "EU", "UK", "OECD", "international"]
_ALERT_QUERIES = [
    "GILTI regulations", "transfer pricing penalty", "IRS audit",
    "Section 382 ownership change", "BEPS Pillar Two", "digital services tax",
    "qualified opportunity zone", "R&D tax credit", "net operating loss",
]
