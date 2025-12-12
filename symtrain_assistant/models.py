from typing import List, Dict, Any
import os

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from .config import EMBEDDING_MODEL_NAME, SUMMARIZATION_MODEL_NAME, OPENAI_MODEL_NAME

try:
    from openai import OpenAI
    _openai_client = OpenAI()
except Exception:
    _openai_client = None


_embedding_model = None
_summarizer_tokenizer = None
_summarizer_model = None


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def get_summarizer():
    global _summarizer_tokenizer, _summarizer_model
    if _summarizer_model is None:
        _summarizer_tokenizer = AutoTokenizer.from_pretrained(SUMMARIZATION_MODEL_NAME)
        _summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(
            SUMMARIZATION_MODEL_NAME
        )
    return _summarizer_tokenizer, _summarizer_model


def embed_texts(texts: List[str]):
    model = get_embedding_model()
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)


def summarize_text(text: str, max_length: int = 80) -> str:
    tokenizer, model = get_summarizer()
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    with torch.no_grad():
        summary_ids = model.generate(
            **inputs,
            max_length=max_length,
            min_length=10,
            num_beams=4,
        )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def call_gpt_json(prompt: str, system: str = "You are a helpful assistant.") -> Dict[str, Any]:
    """
    Simple wrapper around OpenAI responses that expects a JSON object back.
    Make sure you set OPENAI_API_KEY in your environment.
    """
    if _openai_client is None:
        raise RuntimeError("OpenAI client not available. Install openai and set API key.")

    response = _openai_client.chat.completions.create(
        model=OPENAI_MODEL_NAME,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    import json

    return json.loads(content)
