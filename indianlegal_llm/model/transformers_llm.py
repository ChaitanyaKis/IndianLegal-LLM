"""Real LLM backend: a Hugging Face Transformers causal LM, served 4-bit.

Default base model is ``microsoft/phi-4`` (MIT, CLAUDE.md §2), loaded in 4-bit via
bitsandbytes with ``device_map="auto"``. All heavy dependencies (torch,
transformers, bitsandbytes) are imported lazily so the package still imports with
the standard library alone; the real backend requires ``pip install -e .[model]``.

CLAUDE.md §5 (cost/bandwidth): weights download once IN THE CLOUD. Loading is hard-
gated on a CUDA GPU being present — on a CPU/laptop ``ensure_loaded`` raises BEFORE
any weights are fetched, so the multi-GB base model is never pulled to a laptop.
`build_pipeline()` then falls back to the StubLLM for local/offline dev.
"""

from __future__ import annotations

import sys

from .base import BaseLLM

# Models confirmed license-clean for serving (kept in sync with config §2).
_LICENSE_CLEAN = {"microsoft/phi-4"}


class TransformersLLM(BaseLLM):
    """A 4-bit, low-temperature causal LM for citation-grounded legal answering."""

    def __init__(
        self,
        model_id: str = "microsoft/phi-4",
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.0,  # greedy: maximal determinism for legal output
        top_p: float = 0.9,
    ) -> None:
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self._model = None
        self._tokenizer = None

    # -- loading ----------------------------------------------------------- #
    def ensure_loaded(self) -> None:
        """Load tokenizer + 4-bit model once. Raises (caught by the pipeline and
        downgraded to the stub) when torch/transformers are missing or no GPU is
        present — WITHOUT downloading any weights on a CPU/laptop (CLAUDE.md §5)."""
        if self._model is not None:
            return
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - exercised without the extra
            raise ImportError(
                "torch is required to serve a real model. "
                "Install the model extra: pip install -e .[model]"
            ) from exc

        # Hard GPU gate BEFORE any download: never pull multi-GB weights to a laptop.
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Real-model serving requires a CUDA GPU; refusing to download "
                "weights on a CPU/laptop (CLAUDE.md §5). Use LLM=stub for local dev, "
                "or run in a cloud GPU environment."
            )

        if self.model_id not in _LICENSE_CLEAN:  # CLAUDE.md §2 advisory
            print(
                f"[indianlegal_llm] WARNING: base model '{self.model_id}' is not on "
                "the MIT/Apache-2.0 allowlist; verify its license (CLAUDE.md §2).",
                file=sys.stderr,
            )

        try:
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
            )
        except ImportError as exc:  # pragma: no cover - exercised without the extra
            raise ImportError(
                "transformers + accelerate + bitsandbytes are required. "
                "Install the model extra: pip install -e .[model]"
            ) from exc

        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            quantization_config=quantization,
            torch_dtype=torch.bfloat16,
        )
        self._model.eval()

    # -- generation -------------------------------------------------------- #
    def generate(self, system: str, user: str) -> str:
        self.ensure_loaded()
        import torch

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        inputs = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self._model.device)

        gen_kwargs: dict = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0.0,
            "pad_token_id": self._tokenizer.eos_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,  # stop at end-of-turn
        }
        if self.temperature > 0.0:
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["top_p"] = self.top_p

        with torch.no_grad():
            output = self._model.generate(inputs, **gen_kwargs)

        # Decode only the newly generated continuation, not the prompt echo.
        generated = output[0][inputs.shape[-1] :]
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()
