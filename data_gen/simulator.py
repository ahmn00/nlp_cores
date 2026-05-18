import hashlib
import json
import random
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from openai import OpenAI

from .events import EVENT_MAP, EVENT_TYPES
from .personas import Persona


@dataclass
class EventRow:
    user_id: str
    amplitude_id: int
    event_type: str
    event_time: str
    session_id: int
    device_type: str
    os_name: str
    platform: str
    country: str
    city: str
    event_properties: str
    user_properties: str
    sequence_number: int


def simulate_events(
    client: OpenAI,
    personas: list[Persona],
    events_per_persona: int = 500,
    seed: int = 42,
    model: str = "gpt-4o-mini",
) -> list[EventRow]:
    rng = random.Random(seed)
    all_rows: list[EventRow] = []

    for idx, persona in enumerate(personas):
        print(f"  Simulating events for: {persona.name} ({persona.archetype})...")
        sequences = _generate_event_sequences(client, persona, model)
        rows = _simulate_persona_events(persona, idx, sequences, events_per_persona, rng)
        all_rows.extend(rows)

    all_rows.sort(key=lambda r: r.event_time)
    return all_rows


def _generate_event_sequences(
    client: OpenAI,
    persona: Persona,
    model: str,
) -> list[list[str]]:
    sample_topics = persona.search_topics[:5]

    prompt = f"""You are generating realistic event sequence templates for a Bloomberg Terminal user.

User archetype: {persona.archetype}
Role: {persona.org_type}
Primary Bloomberg tools: {', '.join(persona.primary_tools)}
Sample search topics they care about: {', '.join(sample_topics)}

Generate 30 realistic event sequence templates showing what this user does during a Bloomberg session.
Each template is an ordered list of event_type strings from this vocabulary:
session_start, session_end, screen_view, search_query, document_view, document_download,
news_view, alert_create, alert_delete, data_export, function_run, annotation_add, contact_lookup

Rules:
- Every sequence must start with session_start and end with session_end
- Sequences should be 4-12 events long
- Make sequences realistic for this user's role and tasks
- Vary the patterns (research sessions, quick lookups, deep analysis sessions, news monitoring)

Return a JSON array of arrays (no markdown fences), e.g.:
[["session_start","screen_view","search_query","document_view","session_end"], ...]

Return only valid JSON."""

    response = client.chat.completions.create(
        model=model,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    sequences = json.loads(raw)

    valid = []
    for seq in sequences:
        if (
            isinstance(seq, list)
            and len(seq) >= 2
            and all(e in EVENT_TYPES for e in seq)
        ):
            valid.append(seq)

    if not valid:
        valid = [
            ["session_start", "screen_view", "search_query", "document_view", "session_end"],
            ["session_start", "screen_view", "news_view", "session_end"],
            ["session_start", "function_run", "data_export", "session_end"],
        ]

    return valid


def _simulate_persona_events(
    persona: Persona,
    persona_idx: int,
    sequences: list[list[str]],
    events_per_persona: int,
    rng: random.Random,
) -> list[EventRow]:
    user_id = _make_user_id(persona_idx)
    amplitude_id = _make_amplitude_id(persona_idx)
    hour_weights = _get_hour_weights(persona.behavior_traits)

    window_start = datetime(2026, 2, 16, tzinfo=timezone.utc)
    window_end = datetime(2026, 5, 16, tzinfo=timezone.utc)
    window_seconds = int((window_end - window_start).total_seconds())

    rows: list[EventRow] = []
    session_idx = 0
    generated = 0

    while generated < events_per_persona:
        template = rng.choice(sequences)
        session_id = _make_session_id(user_id, session_idx)
        session_idx += 1

        anchor_offset = rng.randint(0, window_seconds)
        anchor_dt = window_start + timedelta(seconds=anchor_offset)
        anchor_dt = _snap_to_hour_weights(anchor_dt, hour_weights, rng)

        current_dt = anchor_dt
        for seq_pos, event_type in enumerate(template):
            if generated >= events_per_persona:
                break

            props = _generate_event_properties(event_type, persona, rng)
            user_props = {
                "persona": persona.archetype,
                "org_type": persona.org_type,
                "firm_size": _infer_firm_size(persona.org_type),
                "years_experience": persona.years_experience,
                "city": persona.city,
            }

            row = EventRow(
                user_id=user_id,
                amplitude_id=amplitude_id,
                event_type=event_type,
                event_time=current_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                session_id=session_id,
                device_type="Desktop",
                os_name=rng.choice(["Windows", "Windows", "macOS"]),
                platform="Bloomberg Terminal",
                country="United States",
                city=persona.city,
                event_properties=json.dumps(props),
                user_properties=json.dumps(user_props),
                sequence_number=seq_pos,
            )
            rows.append(row)
            generated += 1

            delay = _lognormal_delay(rng)
            current_dt = current_dt + timedelta(seconds=delay)

    return rows


def _make_user_id(idx: int) -> str:
    return f"usr_{idx:04d}"


def _make_amplitude_id(idx: int) -> int:
    return 4200000 + idx


def _make_session_id(user_id: str, session_idx: int) -> int:
    raw = f"{user_id}:{session_idx}"
    return int(hashlib.md5(raw.encode()).hexdigest()[:8], 16)


def _lognormal_delay(rng: random.Random) -> float:
    delay = rng.lognormvariate(mu=3.5, sigma=1.2)
    return max(5.0, min(delay, 600.0))


def _get_hour_weights(behavior_traits: dict) -> list[float]:
    pattern = behavior_traits.get("usage_pattern", "spread")
    weights = [0.1] * 24
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
        for h in range(8, 10):
            weights[h] = 1.0
        for h in range(14, 16):
            weights[h] = 1.0
    else:
        for h in range(8, 19):
            weights[h] = 0.8
    return weights


def _snap_to_hour_weights(
    dt: datetime, hour_weights: list[float], rng: random.Random
) -> datetime:
    hours = list(range(24))
    chosen_hour = rng.choices(hours, weights=hour_weights, k=1)[0]
    minute = rng.randint(0, 59)
    second = rng.randint(0, 59)
    return dt.replace(hour=chosen_hour, minute=minute, second=second)


_IRS_RULINGS = [
    "Rev. Rul. 2023-14", "Rev. Proc. 2024-01", "PLR 202412001", "CCA 202415010",
    "Notice 2024-55", "T.D. 9999", "Rev. Rul. 2022-08", "PLR 202308003",
]
_CASE_LAW = [
    "Coca-Cola Co. v. Commissioner", "3M Co. v. Commissioner",
    "Medtronic Inc. v. Commissioner", "Altera Corp. v. Commissioner",
    "Whirlpool Corp. v. Commissioner", "Amazon.com Inc. v. Commissioner",
]
_TAX_REGS = [
    "Treas. Reg. § 1.482-1", "Treas. Reg. § 1.368-1", "Treas. Reg. § 1.351-1",
    "Treas. Reg. § 1.1001-1", "IRC § 382 regulations", "GILTI regulations § 951A",
    "BEAT regulations § 59A", "FDII regulations § 250",
]
_BLOOMBERG_FUNCTIONS = ["TAX", "BTAX", "NI", "LAW", "BLAW", "CRED", "CF", "MA", "BRIEF", "COURT", "REGN", "TRA", "TRAX"]
_JURISDICTIONS = ["US_federal", "US_federal", "US_state", "EU", "UK", "OECD", "international"]
_ALERT_QUERIES = [
    "GILTI regulations", "transfer pricing penalty", "IRS audit",
    "Section 382 ownership change", "BEPS Pillar Two", "digital services tax",
    "qualified opportunity zone", "R&D tax credit", "net operating loss",
]


def _generate_event_properties(
    event_type: str, persona: Persona, rng: random.Random
) -> dict:
    if event_type == "session_start":
        return {"login_method": rng.choice(["password", "sso", "biometric"])}

    if event_type == "session_end":
        return {
            "duration_seconds": int(rng.lognormvariate(6.5, 1.0)),
            "logout_type": rng.choice(["manual", "timeout", "idle"]),
        }

    if event_type == "screen_view":
        func = rng.choice(persona.primary_tools) if persona.primary_tools else rng.choice(_BLOOMBERG_FUNCTIONS)
        return {"function_code": func, "previous_screen": rng.choice(_BLOOMBERG_FUNCTIONS)}

    if event_type == "search_query":
        query = rng.choice(persona.search_topics) if persona.search_topics else "tax reform 2024"
        return {
            "query_text": query,
            "search_scope": rng.choice(["full_text", "document", "case_law", "regulation", "news"]),
            "result_count": rng.randint(3, 150),
        }

    if event_type == "document_view":
        doc_type = rng.choice(["IRS_ruling", "tax_regulation", "case_law", "treaty", "guidance", "commentary"])
        if doc_type == "IRS_ruling":
            title = rng.choice(_IRS_RULINGS)
        elif doc_type == "case_law":
            title = rng.choice(_CASE_LAW)
        else:
            title = rng.choice(_TAX_REGS)
        return {
            "doc_id": f"DOC-{rng.randint(10000, 99999)}",
            "doc_type": doc_type,
            "doc_title": title,
            "jurisdiction": rng.choice(_JURISDICTIONS),
        }

    if event_type == "document_download":
        return {
            "doc_id": f"DOC-{rng.randint(10000, 99999)}",
            "doc_type": rng.choice(["IRS_ruling", "tax_regulation", "case_law", "guidance"]),
            "export_format": rng.choice(["PDF", "DOCX", "PDF", "PDF"]),
        }

    if event_type == "news_view":
        return {
            "story_id": f"NEWS-{rng.randint(100000, 999999)}",
            "headline": rng.choice(persona.search_topics[:6]) if persona.search_topics else "Tax reform update",
            "topic_tags": rng.sample(["tax", "IRS", "OECD", "transfer_pricing", "M&A", "compliance"], k=rng.randint(1, 3)),
            "source": rng.choice(["Bloomberg Tax", "Bloomberg Law", "Tax Notes", "BNA", "Reuters"]),
        }

    if event_type == "alert_create":
        return {
            "alert_type": rng.choice(["keyword", "company", "regulation", "case"]),
            "alert_query": rng.choice(_ALERT_QUERIES),
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
        func = rng.choice(persona.primary_tools) if persona.primary_tools else rng.choice(_BLOOMBERG_FUNCTIONS)
        return {
            "function_code": func,
            "input_params": {"jurisdiction": rng.choice(["US", "EU", "UK"]), "year": rng.choice([2023, 2024])},
        }

    if event_type == "annotation_add":
        return {
            "doc_id": f"DOC-{rng.randint(10000, 99999)}",
            "annotation_type": rng.choice(["highlight", "note", "bookmark", "tag"]),
            "note_text": rng.choice(["See also Rev. Rul. 2023-14", "Check with partner", "Cite in brief", "Confirm with client"]),
        }

    if event_type == "contact_lookup":
        return {
            "lookup_type": rng.choice(["attorney", "regulator", "tax_authority", "expert_witness"]),
            "query": rng.choice(["IRS Appeals Office", "DOJ Tax Division", "Treasury OTP", "OECD Secretariat"]),
        }

    return {}


def _infer_firm_size(org_type: str) -> str:
    large = ["large law firm", "Fortune 500", "investment bank", "Big 4"]
    if any(l in org_type for l in large):
        return "large"
    if "boutique" in org_type or "private wealth" in org_type:
        return "small"
    return "medium"
