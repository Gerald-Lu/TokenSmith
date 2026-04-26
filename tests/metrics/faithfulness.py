from __future__ import annotations
import os
import warnings
from typing import Any, Dict, List, Optional, Sequence
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tests.metrics.base import MetricBase

def _flatten_context(chunks_info):
    parts = []
    for c in chunks_info or []:
        content = c.get("content")
        if isinstance(content, list):
            content = " ".join(str(x) for x in content)
        if content:
            parts.append(str(content))
    return "\n\n".join(parts)

class FaithfulnessMetric(MetricBase):
    def __init__(self):
        self._tokenizer = None
        self._model = None
        self._available = self._initialize()

    @property
    def name(self) -> str:
        return "faithfulness"

    @property
    def weight(self) -> float:
        return 0.0

    def _initialize(self) -> bool:
        try:
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
            warnings.filterwarnings("ignore", message=".*CUDA capability.*")
            model_name = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
            return True
        except Exception as e:
            print(f"Faithfulness metric initialization failed: {e}")
            return False

    def is_available(self) -> bool:
        return self._available
    def calculate(
        self,
        answer: str,
        chunks_info: Optional[Sequence[Dict[str, Any]]] = None,
        keywords: Optional[List[str]] = None,
    ) -> float:
        if not self.is_available() or not answer or not answer.strip():
            return 0.0
        context = _flatten_context(chunks_info or [])
        if not context.strip():
            return 0.0
        try:
            inputs = self._tokenizer(
                context, answer,
                truncation="only_first", max_length=512, return_tensors="pt",
            )
            with torch.no_grad():
                logits = self._model(inputs["input_ids"].to("cpu"))["logits"][0]
            probs = torch.softmax(logits, -1).tolist()
            p = dict(zip(["entailment", "neutral", "contradiction"], probs))
            return float(min(max(p["entailment"] - p["contradiction"], 0.0), 1.0))
        except Exception as e:
            print(f"Faithfulness calculation failed: {e}")
            return 0.0
