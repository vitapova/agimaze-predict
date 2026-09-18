# Visual Transformer Training with Action Loss

Identical to byte-transformer training but uses visual-memory model.

## Usage

```bash
python3 scripts/train_visual_transformer_with_actions.py \
  --config experiments/rational-agent/visual/3x3-keys-mixed.toml
```

## Differences from Byte-Transformer

### Architecture

**Byte-Transformer:**
- Input: Literal byte sequence including `<MAP>` 
- Model: Vanilla decoder-only transformer

**Visual-Transformer:**
- Input: Separate 2D canvas (map) + text sequence (actions)
- Model: Hybrid visual-memory transformer
  - Visual backbone: 2D CNN + temporal updates
  - Text backbone: Decoder-only transformer
  - Event-triggered updates on `</ACT>`

### Model Parameters

**Additional (vs byte-transformer):**

| Parameter | Default | Description |
|-----------|---------|-------------|
| canvas_height | 9 | Map grid height |
| canvas_width | 17 | Map grid width |
| visual_d_model | 128 | Visual backbone dimension |
| visual_spatial_layers | 2 | Spatial convolution layers |
| visual_temporal_layers | 2 | Temporal update layers |
| temporal_history | 8 | Number of stored visual states |
| pos_readout | "full_text" | Position readout mode |

### Loss Computation

**Identical to byte-transformer:**
- ✅ Loss on `<ACT>content</ACT>` (action bytes + closing tag)
- ✅ Loss on `<POS>(...)</POS>` (position target)
- ✅ Separate metrics for ACT and POS

### Training Output

```
Epoch   5/150 | Train: 1.8234 / 1.9123 / 1.7456 | Val: 1.7890 / 1.8567 / 1.7234 | 45.3s
                       ↑       ↑       ↑              ↑       ↑       ↑
                    Total    ACT     POS          Total    ACT     POS
```

## Configuration

### Example TOML

```toml
[model]
# Text backbone (same as byte-transformer)
context_length = 768
d_model = 256
n_heads = 4
n_layers = 5

# Visual backbone (additional)
canvas_height = 9
canvas_width = 17
visual_d_model = 128
visual_spatial_layers = 2
visual_temporal_layers = 2
temporal_history = 8
pos_readout = "full_text"
```

### pos_readout Modes

- **`"full_text"`** (default): Position prediction uses both text and visual state
- **`"visual_only"`**: Position prediction uses only visual state (faster)

## Expected Performance

Similar to byte-transformer but potentially better due to spatial structure:

| Metric | Byte-Transformer | Visual-Transformer |
|--------|-----------------|-------------------|
| Success rate | 85-95% | 88-97% |
| ACT loss | 0.3-0.5 | 0.2-0.4 |
| POS loss | 0.1-0.3 | 0.1-0.2 |
| Training time | Baseline | +20-30% slower |

## Files

```
scripts/
├── train_visual_transformer_with_actions.py  ← Training script
└── tokenizer_visual_with_acts.py            ← Modified tokenizer

experiments/rational-agent/visual/
└── 3x3-keys-mixed.toml                      ← Example config

runs/
└── visual-rational-3x3-keys-mixed.pt        ← Checkpoint
```

## Comparison with Byte-Transformer

```bash
# Train both
python3 scripts/train_byte_transformer_with_actions.py \
  --config experiments/rational-agent/seq/3x3-keys-mixed.toml

python3 scripts/train_visual_transformer_with_actions.py \
  --config experiments/rational-agent/visual/3x3-keys-mixed.toml

# Compare results
# (Both should have similar loss breakdown and performance)
```

## Implementation Notes

1. **Tokenizer:** `tokenizer_visual_with_acts.py` handles visual serialization + ACT loss
2. **Model:** Uses existing `VisualTransformer` from baselines
3. **Loss:** Identical `compute_losses_by_type()` as byte-transformer
4. **Logging:** Same format and metrics

**No changes to dataset format** — same `.jsonl` files work for both!
