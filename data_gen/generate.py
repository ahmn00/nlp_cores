"""
Synthetic Bloomberg terminal interaction data generator.

Usage (from nlp_cores/ directory):
    python -m data_gen.generate
    python -m data_gen.generate --events-per-persona 50 --output /tmp/test.csv
"""
import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

from .personas import generate_personas
from .simulator import simulate_events


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic Bloomberg terminal interaction data in Amplitude CSV format"
    )
    parser.add_argument(
        "--output",
        default="data_gen/output/amplitude_events.csv",
        help="Output CSV path (default: data_gen/output/amplitude_events.csv)",
    )
    parser.add_argument(
        "--num-personas",
        type=int,
        default=10,
        help="Number of personas to generate (default: 10)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of days to simulate (default: 7)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini)",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    print(f"\n=== Bloomberg NLP Tax Data Generator ===")
    print(f"Model: {args.model}")
    print(f"Personas: {args.num_personas}")
    print(f"Days to simulate: {args.days}")
    print(f"Random seed: {args.seed}")
    print(f"Output: {args.output}\n")

    print("Phase 1/3: Generating personas via OpenAI API...")
    personas = generate_personas(client, num_personas=args.num_personas, model=args.model)
    print(f"  -> {len(personas)} personas generated\n")

    print("Phase 2/3: Simulating events (workflow-driven, day-based)...")
    rows = simulate_events(
        personas,
        num_days=args.days,
        seed=args.seed,
    )
    print(f"  -> {len(rows)} total events generated\n")

    print("Phase 3/3: Writing CSV...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = [
        {
            "user_id": r.user_id,
            "amplitude_id": r.amplitude_id,
            "event_type": r.event_type,
            "event_time": r.event_time,
            "session_id": r.session_id,
            "workflow_type": r.workflow_type,
            "device_type": r.device_type,
            "os_name": r.os_name,
            "platform": r.platform,
            "country": r.country,
            "city": r.city,
            "event_properties": r.event_properties,
            "user_properties": r.user_properties,
            "sequence_number": r.sequence_number,
        }
        for r in rows
    ]

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)

    print(f"\n=== Summary ===")
    print(f"Personas generated : {len(personas)}")
    for p in personas:
        print(f"  {p.archetype:<30} {p.activity_level:<11}  {p.name}  ({p.firm}, {p.city})")

    print(f"\nTotal events       : {len(rows)}")
    print(f"Unique sessions    : {len({r.session_id for r in rows})}")
    if rows:
        times = sorted(r.event_time for r in rows)
        print(f"Date range         : {times[0]} -> {times[-1]}")

    event_counts = Counter(r.event_type for r in rows)
    print("\nEvent type distribution:")
    for event_type, count in sorted(event_counts.items(), key=lambda x: -x[1]):
        print(f"  {event_type:<25} {count:>6}")

    workflow_counts = Counter(r.workflow_type for r in rows)
    print("\nWorkflow distribution (sessions):")
    workflow_sessions: dict[str, set] = {}
    for r in rows:
        workflow_sessions.setdefault(r.workflow_type, set()).add(r.session_id)
    for wf, sess_set in sorted(workflow_sessions.items(), key=lambda x: -len(x[1])):
        print(f"  {wf:<30} {len(sess_set):>5} sessions")

    print(f"\nOutput written to  : {output_path.resolve()}")


if __name__ == "__main__":
    main()
