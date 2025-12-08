import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

from .config import DATA_RAW


@dataclass
class DialogueTurn:
    sequence_number: int
    actor: str
    text: str


@dataclass
class Simulation:
    sim_id: str
    company: str
    json_path: Path
    dialogue: List[DialogueTurn] = field(default_factory=list)
    merged_text: str = ""
    customer_text: str = ""
    agent_text: str = ""


def find_json_files(root: Path = DATA_RAW) -> List[Path]:
    return [p for p in root.rglob("*.json")]


def load_simulation_from_json(json_path: Path) -> Simulation:
    # company = folder name two levels up, adjust as needed
    parts = json_path.parts
    try:
        company = parts[-2]
    except IndexError:
        company = "unknown_company"

    sim_id = json_path.stem

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    audio_items = data.get("audioContentItems", [])
    turns: List[DialogueTurn] = []

    for item in audio_items:
        seq = item.get("sequenceNumber", 0)
        actor = item.get("actor", "Unknown")
        text = item.get("fileTranscript", "").strip()
        if text:
            turns.append(DialogueTurn(seq, actor, text))

    # Sort by sequence number just in case
    turns.sort(key=lambda t: t.sequence_number)

    # Merge dialogue text and separate by roles
    merged_lines = []
    customer_lines = []
    agent_lines = []

    for t in turns:
        line = f"{t.actor}: {t.text}"
        merged_lines.append(line)

        # Adjust actor labels once you inspect the actual JSON
        actor_lower = t.actor.lower()
        if "customer" in actor_lower or "sym" in actor_lower:
            customer_lines.append(t.text)
        else:
            agent_lines.append(t.text)

    merged_text = " ".join(merged_lines)
    customer_text = " ".join(customer_lines)
    agent_text = " ".join(agent_lines)

    return Simulation(
        sim_id=sim_id,
        company=company,
        json_path=json_path,
        dialogue=turns,
        merged_text=merged_text,
        customer_text=customer_text,
        agent_text=agent_text,
    )


def load_all_simulations(root: Path = DATA_RAW) -> List[Simulation]:
    json_files = find_json_files(root)
    sims = [load_simulation_from_json(p) for p in json_files]
    return sims


def simulations_to_dataframe(sims: List[Simulation]) -> pd.DataFrame:
    rows = []
    for s in sims:
        rows.append(
            {
                "sim_id": s.sim_id,
                "company": s.company,
                "json_path": str(s.json_path),
                "merged_text": s.merged_text,
                "customer_text": s.customer_text,
                "agent_text": s.agent_text,
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    sims = load_all_simulations()
    df = simulations_to_dataframe(sims)
    print(df.head())
    DATA_RAW.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(DATA_RAW.parent / "simulations_raw.csv", index=False)