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

# Models confirmed license-clean for serving (kept in sync with config.py §2).
_LICENSE_CLEAN = {
    "microsoft/phi-4",  # MIT — default zero-shot serving base
    "microsoft/Phi-3.5-mini-instruct",  # MIT — small (3.8B) default for the GPU eval
    "Qwen/Qwen3-4B-Instruct-2507",  # Apache-2.0 — fine-tune base
}

# 4-bit (QLoRA-style nf4) quantization knobs. Kept as plain data so the field set
# can be asserted offline WITHOUT importing transformers/torch (CLAUDE.md §6).
# ``bnb_4bit_compute_dtype`` is a torch dtype NAME here; it is resolved to the real
# ``torch`` dtype at load time. This is the config that makes a ~14B model occupy
# ~9 GB instead of ~16+ GB (the T4 OOM we are fixing).
QUANT_4BIT = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_compute_dtype": "bfloat16",
}


class TransformersLLM(BaseLLM):
    """A 4-bit, low-temperature causal LM for citation-grounded legal answering."""

    def __init__(
        self,
        model_id: str = "microsoft/phi-4",
        *,
        adapter: str = "",
        max_new_tokens: int = 512,
        temperature: float = 0.0,  # greedy: maximal determinism for legal output
        top_p: float = 0.9,
    ) -> None:
        self.model_id = model_id
        #: Optional LoRA/QLoRA adapter (local path or HF id) layered on the base.
        self.adapter = adapter
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

        # Build the 4-bit config from the offline-testable QUANT_4BIT data,
        # resolving the compute-dtype name to a real torch dtype.
        quant_kwargs = dict(QUANT_4BIT)
        quant_kwargs["bnb_4bit_compute_dtype"] = getattr(
            torch, quant_kwargs["bnb_4bit_compute_dtype"]
        )
        quantization = BitsAndBytesConfig(**quant_kwargs)

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",  # accelerate places/shards across all visible GPUs
            quantization_config=quantization,
            # Do NOT pass torch_dtype/dtype here: with a 4-bit quantization_config the
            # compute dtype comes from bnb_4bit_compute_dtype above. A top-level dtype
            # both triggers the "torch_dtype is deprecated" warning AND can load the
            # weights at full precision, defeating 4-bit (the T4 OOM we are fixing).
        )

        # Confirm 4-bit actually engaged and report the real footprint, so a silent
        # full-precision load (the bug this fixes) is impossible to miss in the logs.
        is_4bit = bool(getattr(model, "is_loaded_in_4bit", False))
        try:
            footprint_gb = model.get_memory_footprint() / 1e9
            footprint = f"~{footprint_gb:.1f} GB"
        except Exception:  # pragma: no cover - footprint is best-effort logging only
            footprint = "unknown size"
        print(
            f"[indianlegal_llm] loaded '{self.model_id}': "
            f"4-bit={is_4bit} ({footprint})",
            file=sys.stderr,
        )
        if not is_4bit:
            print(
                "[indianlegal_llm] WARNING: 4-bit quantization did NOT engage — the "
                "model is loaded at higher precision and may OOM on a 16 GB GPU. "
                "Check that bitsandbytes is installed (pip install -e .[model]).",
                file=sys.stderr,
            )

        # Layer the fine-tuned LoRA/QLoRA adapter on the 4-bit base, if configured.
        # The adapter is small (50-200 MB) and ships MIT (CLAUDE.md §2). Only
        # reached after the GPU gate, so nothing is downloaded on a laptop.
        if self.adapter:
            try:
                from peft import PeftModel
            except ImportError as exc:  # pragma: no cover - exercised without the extra
                raise ImportError(
                    "peft is required to load a LoRA adapter. "
                    "Install the model extra: pip install -e .[model]"
                ) from exc
            model = PeftModel.from_pretrained(model, self.adapter)

        model.eval()
        self._model = model

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
