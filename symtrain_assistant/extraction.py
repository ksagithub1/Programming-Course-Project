# symtrain_assistant/extraction.py
from typing import Dict
import pandas as pd


def transformer_reason_steps(row: pd.Series) -> Dict[str, any]:
    """
    Very lightweight heuristic extraction with NO ML models.
    - reason: first sentence of customer_text or merged_text
    - steps: split agent_text into short sentences
    """
    customer_text = (row.get("customer_text", "") or "").strip()
    agent_text = (row.get("agent_text", "") or "").strip()
    merged_text = (row.get("merged_text", "") or "").strip()

    src = customer_text if customer_text else merged_text
    reason = src.split(".")[0].strip()
    if not reason:
        reason = "Customer requested assistance."

    src_steps = agent_text if agent_text else merged_text
    raw_steps = [
        s.strip()
        for s in src_steps.replace("\n", " ").split(".")
        if s.strip()
    ]

    steps = raw_steps[:6] if raw_steps else ["Assist the customer following standard procedure."]

    return {"reason_txf": reason, "steps_txf": steps}


def run_extraction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply ONLY heuristic extraction to all rows,
    and copy those values into the *_gpt columns so
    downstream code (categorization, few-shot, app) still works.
    """
    txf_reasons = []
    txf_steps = []
    gpt_reasons = []
    gpt_steps = []

    for _, row in df.iterrows():
        res = transformer_reason_steps(row)
        reason = res["reason_txf"]
        steps = res["steps_txf"]

        txf_reasons.append(reason)
        txf_steps.append(steps)

        gpt_reasons.append(reason)
        gpt_steps.append(steps)

    df = df.copy()
    df["reason_txf"] = txf_reasons
    df["steps_txf"] = txf_steps
    df["reason_gpt"] = gpt_reasons
    df["steps_gpt"] = gpt_steps
    return df
