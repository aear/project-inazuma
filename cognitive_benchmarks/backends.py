from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Sequence


class HuggingFaceCausalScorer:
    """Length-normalized continuation likelihood for causal language models."""

    def __init__(self, model: str, *, device: str = "auto") -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "The Hugging Face backend needs the optional 'torch' and "
                "'transformers' packages."
            ) from exc

        self.name = model
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model)
        selected = None if device == "auto" else device
        self._model = AutoModelForCausalLM.from_pretrained(model)
        if selected:
            self._model.to(selected)
        self._model.eval()

    def score_choices(self, prompt: str, choices: Sequence[str]) -> Sequence[float]:
        torch = self._torch
        device = next(self._model.parameters()).device
        scores = []
        for choice in choices:
            prefix_ids = self._tokenizer(prompt, return_tensors="pt", add_special_tokens=False)[
                "input_ids"
            ]
            full_ids = self._tokenizer(
                prompt + choice, return_tensors="pt", add_special_tokens=False
            )["input_ids"]
            prefix_length = prefix_ids.shape[1]
            if full_ids.shape[1] <= prefix_length:
                scores.append(float("-inf"))
                continue
            full_ids = full_ids.to(device)
            with torch.inference_mode():
                logits = self._model(full_ids).logits[:, :-1, :]
            targets = full_ids[:, 1:]
            token_log_probs = torch.log_softmax(logits, dim=-1).gather(
                2, targets.unsqueeze(-1)
            ).squeeze(-1)
            start = max(0, prefix_length - 1)
            scores.append(float(token_log_probs[:, start:].mean().item()))
        return scores


class CommandScorer:
    """Adapter for an Ina or other local scorer using one JSON request per case."""

    def __init__(self, name: str, command: Sequence[str], *, timeout: float = 120.0) -> None:
        if not command:
            raise ValueError("command backend requires a command")
        self.name = name
        self.command = tuple(command)
        self.timeout = timeout

    def score_choices(self, prompt: str, choices: Sequence[str]) -> Sequence[float]:
        request = json.dumps({"prompt": prompt, "choices": list(choices)}) + "\n"
        completed = subprocess.run(
            self.command,
            input=request,
            text=True,
            capture_output=True,
            timeout=self.timeout,
            check=False,
        )
        if completed.returncode:
            raise RuntimeError(
                f"scorer command exited {completed.returncode}: {completed.stderr.strip()}"
            )
        try:
            response = json.loads(completed.stdout)
            return [float(value) for value in response["scores"]]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError("scorer command must return JSON: {\"scores\": [...]}") from exc
