#!/usr/bin/env python3
"""Test that visual tokenizer creates correct ACT masks."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agimaze_predict.data.prepared import PreparedMapActionsToPosDataset
from tokenizer_visual_with_acts import collate_visual_examples_with_actions

# Load one example
dataset_path = Path(__file__).parent.parent / "datasets/seq/3x3-keys-rational-train1-1step.jsonl"
dataset = PreparedMapActionsToPosDataset(dataset_path)
examples = list(dataset)[:2]

print(f"Loaded {len(examples)} examples")

# Collate
batch = collate_visual_examples_with_actions(
    examples,
    context_length=768,
    canvas_height=9,
    canvas_width=17,
)

print(f"\nBatch keys: {list(batch.keys())}")
print(f"input_ids shape: {len(batch['input_ids'])}x{len(batch['input_ids'][0])}")
print(f"act_mask shape: {len(batch['act_mask'])}x{len(batch['act_mask'][0])}")

# Check masks
for i in range(len(examples)):
    act_mask = batch['act_mask'][i]
    pos_mask = batch['pos_mask'][i]
    labels = batch['labels'][i]
    
    act_bytes = sum(act_mask)
    pos_bytes = sum(pos_mask)
    active_bytes = sum(1 for l in labels if l != -100)
    
    print(f"\nExample {i}:")
    print(f"  ACT bytes: {act_bytes}")
    print(f"  POS bytes: {pos_bytes}")
    print(f"  Active bytes: {active_bytes}")
    print(f"  Total: {act_bytes + pos_bytes}")
    
    if act_bytes == 0:
        print(f"  ⚠️ WARNING: ACT mask is empty!")
        # Show first few tokens
        input_ids = batch['input_ids'][i]
        print(f"  First 20 input tokens: {input_ids[:20]}")
        print(f"  First 20 labels: {labels[:20]}")
        print(f"  First 20 act_mask: {act_mask[:20]}")

print("\nDone.")
