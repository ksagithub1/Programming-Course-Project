from typing import List
import json
import numpy as np

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from symtrain_assistant.models import call_gpt_json, embed_texts
from symtrain_assistant.categorization import predict_category_knn



def retrieve_fewshot_examples(
    df: pd.DataFrame,
    category: str,
    n_examples: int = 3,
) -> List[dict]:
    subset = df[df["category_gpt"] == category]
    if subset.empty:
        subset = df
    subset = subset.sample(min(n_examples, len(subset)), random_state=42)

    examples = []
    for _, row in subset.iterrows():
        raw_steps = row.get("steps_gpt", [])

        if isinstance(raw_steps, np.ndarray):
            steps_list = [str(s) for s in raw_steps.tolist()]
        elif isinstance(raw_steps, (list, tuple)):
            steps_list = [str(s) for s in raw_steps]
        elif isinstance(raw_steps, str):
            steps_list = [raw_steps]
        elif raw_steps is None:
            steps_list = []
        else:
            steps_list = [str(raw_steps)]

        examples.append(
            {
                "user": (row.get("customer_text", "") or "")[:400],
                "reason": row.get("reason_gpt", "") or "",
                "steps": steps_list,
                "category": row.get("category_gpt", "") or "",
            }
        )
    return examples




def build_fewshot_prompt(user_input: str, examples: List[dict]) -> str:
    example_strs = []
    for ex in examples:
        ex_json = json.dumps(
            {
                "category": ex["category"],
                "reason": ex["reason"],
                "steps": ex["steps"],
            },
            indent=2,
        )
        example_strs.append(
            f"User: {ex['user']}\nAssistant:\n{ex_json}\n"
        )

    examples_block = "\n\n".join(example_strs)

    prompt = f"""
You are a customer service assistant that outputs structured JSON.

Here are some examples from past conversations and their structured outputs:

{examples_block}

Now here is a NEW user request:

User: {user_input}

First, decide which category it belongs to (you can reuse any category from the examples or create a new concise one).
Then, infer:
- category
- reason (1 sentence)
- steps (ordered list of short imperative steps).

Return JSON ONLY in this format:

{{
  "category": "string",
  "reason": "string",
  "steps": ["step 1", "step 2", "step 3"]
}}
"""
    return prompt


def run_fewshot_pipeline(
    user_input: str,
    df_labeled: pd.DataFrame,
    knn: KNeighborsClassifier,
) -> dict:
    pred_cat = predict_category_knn(user_input, knn)

    examples = retrieve_fewshot_examples(df_labeled, pred_cat, n_examples=3)

    prompt = build_fewshot_prompt(user_input, examples)
    result = call_gpt_json(prompt)

    if "category" not in result or not result["category"]:
        result["category"] = pred_cat

    return result
