"""vLLM wrapper for batched permutation generation.

We use vLLM's offline ``LLM`` API rather than ``vllm serve``: every
prompt for a run is known before the job starts, so handing the whole
batch to the engine at once lets its continuous batching schedule them,
with no HTTP server, no port, and no readiness polling in the Slurm
script.

Prompts are rendered to strings with the tokenizer's chat template
before generation, instead of going through ``llm.chat()``. That costs
one extra line and buys three things: the exact prompt string is
loggable, the token count is known before generation, and the code does
not depend on the ``chat()`` signature, which has moved between vLLM
releases.

``vllm`` and ``torch`` are imported lazily inside :meth:`load`, so this
module can be imported on a login node to render prompts and count
tokens without a GPU.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

Message = Dict[str, str]


@dataclass
class LLMConfig:
    """Everything that affects generation, kept together so it can be logged.

    Serialising this into the run's output file is what makes a result
    reproducible: model, precision, decoding parameters and seed are all
    recorded alongside the rankings.
    """

    model_path: str
    model_label: str = ""

    # Engine
    dtype: str = "bfloat16"
    max_model_len: int = 16384
    gpu_memory_utilization: float = 0.90
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    trust_remote_code: bool = False
    seed: int = 0

    # Decoding. Greedy by default: a ranking task has one intended answer,
    # and sampling would add variance we would then have to average away.
    temperature: float = 0.0
    top_p: float = 1.0
    max_new_tokens: int = 256

    extra_engine_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model_label:
            self.model_label = os.path.basename(self.model_path.rstrip("/"))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def budget_for_tokens(num_passages: int) -> int:
    """Generation budget large enough for a permutation of ``num_passages``.

    A single item costs about 8 tokens as "[13] > ", so this scales with
    the window and keeps a fixed margin for a preamble the model may emit
    despite being told not to.
    """
    return num_passages * 8 + 64


class PermutationGenerator:
    """Renders ranking prompts and generates permutations in batches."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._llm: Any = None
        self._tokenizer: Any = None

    # -- setup -------------------------------------------------------------

    def load_tokenizer(self) -> Any:
        """Load just the tokenizer. Works on a login node, no GPU needed."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path,
                trust_remote_code=self.config.trust_remote_code,
            )
        return self._tokenizer

    def load(self) -> None:
        """Instantiate the vLLM engine. Requires a GPU; takes a minute or two."""
        if self._llm is not None:
            return

        from vllm import LLM

        self.load_tokenizer()

        self._llm = LLM(
            model=self.config.model_path,
            dtype=self.config.dtype,
            max_model_len=self.config.max_model_len,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            tensor_parallel_size=self.config.tensor_parallel_size,
            enforce_eager=self.config.enforce_eager,
            trust_remote_code=self.config.trust_remote_code,
            seed=self.config.seed,
            **self.config.extra_engine_kwargs,
        )

    @property
    def is_loaded(self) -> bool:
        return self._llm is not None

    # -- prompt rendering --------------------------------------------------

    def render(self, messages: Sequence[Message]) -> str:
        """Apply the model's chat template to a message list.

        ``add_generation_prompt=True`` appends the assistant header, which
        is what makes the model answer instead of continuing the user turn.
        """
        tokenizer = self.load_tokenizer()
        return tokenizer.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=True,
        )

    def count_tokens(self, prompt: str) -> int:
        """Exact prompt length in tokens, for the context-overflow check."""
        tokenizer = self.load_tokenizer()
        return len(tokenizer(prompt, add_special_tokens=False)["input_ids"])

    def fits_in_context(self, prompt: str, max_new_tokens: Optional[int] = None) -> bool:
        """Whether prompt plus generation budget stays inside max_model_len."""
        budget = self.config.max_new_tokens if max_new_tokens is None else max_new_tokens
        return self.count_tokens(prompt) + budget <= self.config.max_model_len

    # -- generation --------------------------------------------------------

    def generate(
        self,
        prompts: Sequence[str],
        max_new_tokens: Optional[int] = None,
    ) -> List[str]:
        """Generate one completion per prompt, preserving input order.

        vLLM returns results in the order the prompts were submitted, so
        ``result[i]`` corresponds to ``prompts[i]``. The whole list is
        passed in one call so the engine can batch it.
        """
        if not self.is_loaded:
            raise RuntimeError("call load() before generate()")

        from vllm import SamplingParams

        sampling = SamplingParams(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=(
                self.config.max_new_tokens if max_new_tokens is None else max_new_tokens
            ),
            seed=self.config.seed,
        )

        outputs = self._llm.generate(list(prompts), sampling)
        return [output.outputs[0].text for output in outputs]

    def generate_one(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        """Single-prompt convenience wrapper, used by the smoke test."""
        return self.generate([prompt], max_new_tokens=max_new_tokens)[0]