"""Fine-tuning support: build an instruction dataset from the processed corpus.

The QLoRA training itself runs in the cloud (see notebooks/finetune_qlora.ipynb);
this package holds the reusable, *testable* dataset builder so the training data
provably matches the Hour-4 citation-guard contract.
"""

from .dataset import (
    InstructionExample,
    build_instruction_examples,
    examples_to_chat_records,
    write_jsonl,
)

__all__ = [
    "InstructionExample",
    "build_instruction_examples",
    "examples_to_chat_records",
    "write_jsonl",
]
