"""Thin vLLM wrapper for batched chat generation.

Kept deliberately small so the reranking logic stays testable without a GPU.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence


class VLLMRanker:
    """Batched greedy chat generation with vLLM."""

    def __init__(
        self,
        model_path: str,
        max_model_len: int = 8192,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        dtype: str = "bfloat16",
        seed: int = 42,
        trust_remote_code: bool = True,
    ):
        # Imported lazily so that unit tests and prompt inspection do not need vLLM.
        from transformers import AutoTokenizer
        from vllm import LLM

        self.model_path = model_path
        self.max_model_len = max_model_len

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )
        self.llm = LLM(
            model=model_path,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
            seed=seed,
            trust_remote_code=trust_remote_code,
        )

    def render(self, messages: List[Dict[str, str]]) -> str:
        """Apply the model's chat template, returning a plain prompt string."""
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def count_tokens(self, messages: List[Dict[str, str]]) -> int:
        return len(self.tokenizer(self.render(messages))["input_ids"])

    def generate(
        self,
        batch_messages: Sequence[List[Dict[str, str]]],
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 256,
        seed: Optional[int] = None,
    ) -> List[str]:
        """Generate one completion per message list. Order is preserved."""
        from vllm import SamplingParams

        prompts = [self.render(m) for m in batch_messages]
        params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed,
        )
        outputs = self.llm.generate(prompts, params)
        return [o.outputs[0].text.strip() for o in outputs]


class EchoRanker:
    """Offline stub used by tests and for prompt inspection on the login node.

    Returns a correctly formatted identity permutation, so the full pipeline
    can be exercised end to end without a GPU.
    """

    def __init__(self, num_candidates_hint: int = 20):
        self.num_candidates_hint = num_candidates_hint

    def render(self, messages):
        return "\n\n".join(f"<{m['role']}>\n{m['content']}" for m in messages)

    def count_tokens(self, messages) -> int:
        return len(self.render(messages).split())

    def generate(self, batch_messages, **kwargs) -> List[str]:
        out = []
        for messages in batch_messages:
            text = messages[-1]["content"]
            # crude guess at the window size from the prompt text
            import re

            nums = [int(n) for n in re.findall(r"\[(\d+)\]", "\n".join(m["content"] for m in messages))]
            n = max(nums) if nums else self.num_candidates_hint
            out.append(" > ".join(f"[{i}]" for i in range(1, n + 1)))
        return out
