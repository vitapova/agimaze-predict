#!/usr/bin/env python3
"""Minimal test of visual transformer training loop."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from agimaze_predict.data.prepared import PreparedMapActionsToPosDataset
from agimaze_predict.baselines.visual_transformer.model import VisualTransformer, VisualTransformerConfig, target_cross_entropy
from tokenizer_visual_with_acts import collate_visual_examples_with_actions

# Load 2 examples
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

# Create model
config = VisualTransformerConfig(
    context_length=768,
    d_model=128,
    n_heads=4,
    n_layers=2,
    canvas_height=9,
    canvas_width=17,
    visual_d_model=128,
)

model = VisualTransformer(config)
print(f"\nModel created: {sum(p.numel() for p in model.parameters()):,} params")

# Convert to tensors
input_ids = torch.tensor(batch["input_ids"], dtype=torch.long)
labels = torch.tensor(batch["labels"], dtype=torch.long)
visual_maps = torch.tensor(batch["visual_maps"], dtype=torch.long)
event_positions = torch.tensor(batch["event_positions"], dtype=torch.long)
event_counts = torch.tensor(batch["event_counts"], dtype=torch.long)
act_mask = torch.tensor(batch["act_mask"], dtype=torch.bool)
pos_mask = torch.tensor(batch["pos_mask"], dtype=torch.bool)

print(f"\nShapes:")
print(f"  input_ids: {input_ids.shape}")
print(f"  visual_maps: {visual_maps.shape}")
print(f"  labels: {labels.shape}")

# Forward pass
print(f"\nRunning forward pass...")
logits = model(input_ids, visual_maps=visual_maps, event_positions=event_positions, event_counts=event_counts)
print(f"  logits shape: {logits.shape}")

# Compute loss
loss = target_cross_entropy(logits, labels)
print(f"  total loss: {loss.item():.4f}")

# Compute ACT/POS breakdown
act_labels = labels.clone()
act_labels[~act_mask] = -100
act_loss = target_cross_entropy(logits, act_labels)
act_bytes = act_mask.sum().item()

pos_labels = labels.clone()
pos_labels[~pos_mask] = -100
pos_loss = target_cross_entropy(logits, pos_labels)
pos_bytes = pos_mask.sum().item()

print(f"\nBreakdown:")
print(f"  ACT: {act_loss.item():.4f} ({act_bytes} bytes)")
print(f"  POS: {pos_loss.item():.4f} ({pos_bytes} bytes)")

if act_bytes == 0:
    print(f"\n⚠️ WARNING: act_bytes = 0!")
else:
    print(f"\n✅ SUCCESS: ACT loss is working!")
