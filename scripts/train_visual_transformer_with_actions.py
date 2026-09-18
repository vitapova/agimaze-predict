#!/usr/bin/env python3
"""Train visual-transformer with loss on BOTH <POS> target AND <ACT> content.

This is identical to train_byte_transformer_with_actions.py but uses
the visual-memory model instead of vanilla byte-transformer.

Usage with TOML config:
    python scripts/train_visual_transformer_with_actions.py \
      --config experiments/rational-agent/visual/3x3-keys-mixed.toml

Usage with CLI args:
    python scripts/train_visual_transformer_with_actions.py \
      --train-dataset datasets/seq/3x3-keys-rational-train.jsonl \
      --validation-dataset datasets/seq/3x3-keys-rational-valid.jsonl \
      --output runs/visual-rational.pt \
      --context-length 896
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import random
import json
import time
from dataclasses import asdict
from datetime import datetime

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from typing import Sequence, Optional

from agimaze_predict.data.prepared import PreparedExample, PreparedMapActionsToPosDataset
from agimaze_predict.baselines.visual_transformer.model import (
    VisualTransformer,
    VisualTransformerConfig,
    target_cross_entropy,
)
from agimaze_predict.baselines.visual_transformer.tokenizer import VOCAB_SIZE

# Import modified tokenizer with ACT loss
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tokenizer_visual_with_acts import collate_visual_examples_with_actions


class Logger:
    """Dual logger that writes to both console and file."""
    
    def __init__(self, log_path: Optional[Path] = None):
        self.log_path = log_path
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, 'w') as f:
                f.write("")
    
    def log(self, message: str, console: bool = True):
        """Write message to log file and optionally to console."""
        if console:
            print(message)
        
        if self.log_path:
            with open(self.log_path, 'a') as f:
                f.write(message + "\n")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _collator_with_actions(
    context_length: int,
    canvas_height: int,
    canvas_width: int,
):
    """Collator that includes <ACT> content in loss."""
    def collate(examples: Sequence[PreparedExample]) -> dict[str, torch.Tensor]:
        batch = collate_visual_examples_with_actions(
            examples,
            context_length=context_length,
            canvas_height=canvas_height,
            canvas_width=canvas_width,
        )
        return {
            "input_ids": torch.tensor(batch["input_ids"], dtype=torch.long),
            "labels": torch.tensor(batch["labels"], dtype=torch.long),
            "visual_maps": torch.tensor(batch["visual_maps"], dtype=torch.long),
            "event_positions": torch.tensor(batch["event_positions"], dtype=torch.long),
            "event_counts": torch.tensor(batch["event_counts"], dtype=torch.long),
            "act_mask": torch.tensor(batch["act_mask"], dtype=torch.bool),
            "pos_mask": torch.tensor(batch["pos_mask"], dtype=torch.bool),
        }
    return collate


def make_dataloader(
    examples: Sequence[PreparedExample],
    *,
    batch_size: int,
    context_length: int,
    canvas_height: int,
    canvas_width: int,
    shuffle: bool = False,
) -> DataLoader:
    return DataLoader(
        examples,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=_collator_with_actions(context_length, canvas_height, canvas_width),
        drop_last=False,
    )


def compute_losses_by_type(logits, labels, act_mask, pos_mask):
    """Compute separate losses for ACT and POS tokens."""
    # Total loss
    total_loss = target_cross_entropy(logits, labels)
    
    # ACT loss
    act_labels = labels.clone()
    act_labels[~act_mask] = -100
    act_loss = target_cross_entropy(logits, act_labels)
    act_bytes = act_mask.sum().item()
    
    # POS loss
    pos_labels = labels.clone()
    pos_labels[~pos_mask] = -100
    pos_loss = target_cross_entropy(logits, pos_labels)
    pos_bytes = pos_mask.sum().item()
    
    return {
        "total_loss": total_loss,
        "act_loss": act_loss,
        "pos_loss": pos_loss,
        "act_bytes": act_bytes,
        "pos_bytes": pos_bytes,
        "total_bytes": act_bytes + pos_bytes,
    }


def train_epoch(model, dataloader, optimizer, device, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    act_loss = 0.0
    pos_loss = 0.0
    total_bytes = 0
    act_bytes = 0
    pos_bytes = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        visual_maps = batch["visual_maps"].to(device)
        event_positions = batch["event_positions"].to(device)
        event_counts = batch["event_counts"].to(device)
        act_mask = batch["act_mask"].to(device)
        pos_mask = batch["pos_mask"].to(device)

        logits = model(input_ids, visual_maps=visual_maps, event_positions=event_positions, event_counts=event_counts)
        
        # Compute losses with breakdown
        losses = compute_losses_by_type(logits, labels, act_mask, pos_mask)
        loss = losses["total_loss"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        # Accumulate weighted losses
        total_loss += float(loss.item()) * losses["total_bytes"]
        act_loss += float(losses["act_loss"].item()) * losses["act_bytes"]
        pos_loss += float(losses["pos_loss"].item()) * losses["pos_bytes"]
        total_bytes += losses["total_bytes"]
        act_bytes += losses["act_bytes"]
        pos_bytes += losses["pos_bytes"]

    return {
        "total": total_loss / total_bytes if total_bytes > 0 else 0.0,
        "act": act_loss / act_bytes if act_bytes > 0 else 0.0,
        "pos": pos_loss / pos_bytes if pos_bytes > 0 else 0.0,
    }


def evaluate_model(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    act_loss = 0.0
    pos_loss = 0.0
    total_bytes = 0
    act_bytes = 0
    pos_bytes = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            visual_maps = batch["visual_maps"].to(device)
            event_positions = batch["event_positions"].to(device)
            event_counts = batch["event_counts"].to(device)
            act_mask = batch["act_mask"].to(device)
            pos_mask = batch["pos_mask"].to(device)

            logits = model(input_ids, visual_maps=visual_maps, event_positions=event_positions, event_counts=event_counts)
            
            # Compute losses with breakdown
            losses = compute_losses_by_type(logits, labels, act_mask, pos_mask)

            # Accumulate weighted losses
            total_loss += float(losses["total_loss"].item()) * losses["total_bytes"]
            act_loss += float(losses["act_loss"].item()) * losses["act_bytes"]
            pos_loss += float(losses["pos_loss"].item()) * losses["pos_bytes"]
            total_bytes += losses["total_bytes"]
            act_bytes += losses["act_bytes"]
            pos_bytes += losses["pos_bytes"]

    return {
        "total": total_loss / total_bytes if total_bytes > 0 else 0.0,
        "act": act_loss / act_bytes if act_bytes > 0 else 0.0,
        "pos": pos_loss / pos_bytes if pos_bytes > 0 else 0.0,
    }


def train(args: argparse.Namespace, logger: Logger) -> dict[str, object]:
    """Main training function."""
    
    start_time = time.time()
    
    seed_everything(args.seed)
    
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load datasets
    logger.log("=" * 80)
    logger.log("LOADING DATASETS")
    logger.log("=" * 80)
    
    train_datasets = [PreparedMapActionsToPosDataset(path) for path in args.train_datasets]
    val_datasets = [PreparedMapActionsToPosDataset(path) for path in args.validation_datasets]
    
    # Flatten examples
    train_examples = [ex for ds in train_datasets for ex in ds]
    val_examples = [ex for ds in val_datasets for ex in ds]
    
    logger.log(f"Train: {len(train_examples)} examples from {len(train_datasets)} file(s)")
    for path in args.train_datasets:
        logger.log(f"  - {path}")
    
    logger.log(f"Valid: {len(val_examples)} examples from {len(val_datasets)} file(s)")
    for path in args.validation_datasets:
        logger.log(f"  - {path}")

    # Create dataloaders
    train_loader = make_dataloader(
        train_examples,
        batch_size=args.batch_size,
        context_length=args.context_length,
        canvas_height=args.canvas_height,
        canvas_width=args.canvas_width,
        shuffle=True,
    )

    val_loader = make_dataloader(
        val_examples,
        batch_size=args.batch_size,
        context_length=args.context_length,
        canvas_height=args.canvas_height,
        canvas_width=args.canvas_width,
        shuffle=False,
    )

    # Create model
    config = VisualTransformerConfig(
        vocab_size=VOCAB_SIZE,
        context_length=args.context_length,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        mlp_multiplier=args.mlp_multiplier,
        dropout=args.dropout,
        canvas_height=args.canvas_height,
        canvas_width=args.canvas_width,
        visual_d_model=args.visual_d_model,
        visual_spatial_layers=args.visual_spatial_layers,
        visual_temporal_layers=args.visual_temporal_layers,
        temporal_history=args.temporal_history,
        pos_readout=args.pos_readout,
    )

    model = VisualTransformer(config).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    logger.log("")
    logger.log("=" * 80)
    logger.log("MODEL ARCHITECTURE")
    logger.log("=" * 80)
    logger.log(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.log(f"  d_model: {config.d_model}")
    logger.log(f"  n_heads: {config.n_heads}")
    logger.log(f"  n_layers: {config.n_layers}")
    logger.log(f"  mlp_multiplier: {config.mlp_multiplier}")
    logger.log(f"  context_length: {config.context_length}")
    logger.log(f"  dropout: {config.dropout}")
    logger.log(f"  canvas: {config.canvas_height}x{config.canvas_width}")
    logger.log(f"  visual_d_model: {config.visual_d_model}")
    logger.log(f"  visual_spatial_layers: {config.visual_spatial_layers}")
    logger.log(f"  visual_temporal_layers: {config.visual_temporal_layers}")
    logger.log(f"  temporal_history: {config.temporal_history}")
    logger.log(f"  pos_readout: {config.pos_readout}")
    logger.log(f"  vocab_size: {config.vocab_size}")
    
    logger.log("")
    logger.log("=" * 80)
    logger.log("TRAINING SETUP")
    logger.log("=" * 80)
    logger.log(f"Device: {device}")
    logger.log(f"Seed: {args.seed}")
    logger.log(f"Epochs: {args.epochs}")
    logger.log(f"Batch size: {args.batch_size}")
    logger.log(f"Learning rate: {args.learning_rate}")
    logger.log(f"Weight decay: {args.weight_decay}")
    logger.log(f"Gradient clip: {args.grad_clip}")
    logger.log(f"Evaluate every: {args.evaluate_every} epochs")
    logger.log(f"Mode: Rational-agent (loss on <ACT> content + <POS> target)")
    logger.log("")

    best_val_loss = float("inf")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"checkpoint already exists: {output_path}\n"
            "Use --overwrite or choose a different --output path"
        )

    logger.log("=" * 80)
    logger.log("TRAINING")
    logger.log("=" * 80)
    logger.log("Format: Epoch | Total / ACT / POS | Time")
    logger.log("")
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_losses = train_epoch(model, train_loader, optimizer, device, grad_clip=args.grad_clip)
        epoch_time = time.time() - epoch_start

        # Evaluate every N epochs
        if epoch % args.evaluate_every == 0 or epoch == args.epochs:
            val_losses = evaluate_model(model, val_loader, device)
            
            msg = (f"Epoch {epoch:3d}/{args.epochs} | "
                   f"Train: {train_losses['total']:.4f} / {train_losses['act']:.4f} / {train_losses['pos']:.4f} | "
                   f"Val: {val_losses['total']:.4f} / {val_losses['act']:.4f} / {val_losses['pos']:.4f} | "
                   f"{epoch_time:.1f}s")
            logger.log(msg)

            if val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "model_config": config.to_dict(),
                    "epoch": epoch,
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "train_datasets": [str(p) for p in args.train_datasets],
                    "validation_datasets": [str(p) for p in args.validation_datasets],
                    "training_args": vars(args),
                }
                torch.save(checkpoint, output_path)
                logger.log(f"  → Checkpoint saved (val={val_losses['total']:.4f})")
        else:
            msg = (f"Epoch {epoch:3d}/{args.epochs} | "
                   f"Train: {train_losses['total']:.4f} / {train_losses['act']:.4f} / {train_losses['pos']:.4f} | "
                   f"{epoch_time:.1f}s")
            logger.log(msg)

    total_time = time.time() - start_time
    
    logger.log("")
    logger.log("=" * 80)
    logger.log("TRAINING COMPLETE")
    logger.log("=" * 80)
    logger.log(f"Best validation loss: {best_val_loss:.4f}")
    logger.log(f"Total training time: {total_time / 60:.1f} minutes")
    logger.log(f"Checkpoint saved: {output_path}")
    
    return {
        "best_val_loss": best_val_loss,
        "final_epoch": args.epochs,
        "output": str(output_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        argument_default=argparse.SUPPRESS,
    )
    
    parser.add_argument("--config", type=Path, help="TOML training configuration file")
    
    # Data
    parser.add_argument("--train-dataset", dest="train_datasets", action="append", type=Path)
    parser.add_argument("--validation-dataset", dest="validation_datasets", action="append", type=Path)
    
    # Model - text backbone
    parser.add_argument("--context-length", type=int)
    parser.add_argument("--d-model", type=int)
    parser.add_argument("--n-heads", type=int)
    parser.add_argument("--n-layers", type=int)
    parser.add_argument("--mlp-multiplier", type=int)
    parser.add_argument("--dropout", type=float)
    
    # Model - visual backbone
    parser.add_argument("--canvas-height", type=int)
    parser.add_argument("--canvas-width", type=int)
    parser.add_argument("--visual-d-model", type=int)
    parser.add_argument("--visual-spatial-layers", type=int)
    parser.add_argument("--visual-temporal-layers", type=int)
    parser.add_argument("--temporal-history", type=int)
    parser.add_argument("--pos-readout", type=str, choices=["full_text", "visual_only"])
    
    # Training
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", dest="epochs", type=int)
    parser.add_argument("--evaluate-every", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--grad-clip", type=float)
    
    # Run
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-overwrite", action="store_false", dest="overwrite")
    
    return parser


def load_visual_config(config_path: Path) -> dict:
    """Load TOML config for visual transformer (handle visual-specific keys)."""
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib
    
    with open(config_path, 'rb') as f:
        raw = tomllib.load(f)
    
    config = {}
    
    # Data section
    if 'data' in raw:
        data = raw['data']
        if 'train_files' in data:
            config['train_datasets'] = [config_path.parent / p for p in data['train_files']]
        if 'validation_files' in data:
            config['validation_datasets'] = [config_path.parent / p for p in data['validation_files']]
    
    # Model section
    if 'model' in raw:
        config.update(raw['model'])
    
    # Training section
    if 'training' in raw:
        config.update(raw['training'])
    
    # Run section
    if 'run' in raw:
        run = raw['run']
        if 'output' in run:
            config['output'] = config_path.parent / run['output']
        if 'overwrite' in run:
            config['overwrite'] = run['overwrite']
        if 'device' in run:
            config['device'] = run['device']
    
    return config


def main():
    parser = build_parser()
    
    # Parse args
    args = parser.parse_args()
    
    # Load TOML if provided
    if hasattr(args, 'config') and args.config:
        config_path = Path(args.config)
        config_values = load_visual_config(config_path)
        
        # Merge: CLI overrides TOML
        for key, value in config_values.items():
            if not hasattr(args, key):
                setattr(args, key, value)
    
    # Set defaults for missing values
    defaults = {
        'device': None,
        'seed': 42,
        'epochs': 50,
        'evaluate_every': 5,
        'batch_size': 32,
        'learning_rate': 3e-4,
        'weight_decay': 0.01,
        'grad_clip': 1.0,
        'context_length': 512,
        'd_model': 128,
        'n_heads': 4,
        'n_layers': 4,
        'mlp_multiplier': 4,
        'dropout': 0.0,
        'canvas_height': 9,
        'canvas_width': 17,
        'visual_d_model': 128,
        'visual_spatial_layers': 2,
        'visual_temporal_layers': 2,
        'temporal_history': 8,
        'pos_readout': 'full_text',
        'overwrite': False,
    }
    
    for key, default_value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, default_value)
    
    # Validate required
    if not hasattr(args, 'train_datasets'):
        parser.error("--train-dataset or --config required")
    if not hasattr(args, 'validation_datasets'):
        parser.error("--validation-dataset or --config required")
    if not hasattr(args, 'output'):
        parser.error("--output or --config required")
    
    config_path = args.config if hasattr(args, 'config') else None
    
    # Setup logging with timestamp
    output_stem = Path(args.output).stem
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_path = Path(args.output).parent / f"{output_stem}_{timestamp}.log"
    logger = Logger(log_path)
    
    # Log header
    logger.log("=" * 80)
    logger.log("RATIONAL AGENT TRAINING - VISUAL TRANSFORMER WITH ACTION LOSS")
    logger.log("=" * 80)
    logger.log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log(f"Log file: {log_path}")
    logger.log("")
    
    # Log TOML config if used
    if config_path:
        logger.log("=" * 80)
        logger.log(f"CONFIGURATION: {config_path}")
        logger.log("=" * 80)
        try:
            with open(config_path, 'r') as f:
                logger.log(f.read())
        except Exception as e:
            logger.log(f"Could not read config file: {e}")
        logger.log("")
    
    # Log all arguments
    logger.log("=" * 80)
    logger.log("ARGUMENTS")
    logger.log("=" * 80)
    for key, value in sorted(vars(args).items()):
        if not key.startswith('_'):
            logger.log(f"{key}: {value}")
    logger.log("")
    
    # Run training
    try:
        result = train(args, logger)
        logger.log("")
        logger.log("=" * 80)
        logger.log(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.log("=" * 80)
        return result
    except Exception as e:
        logger.log("")
        logger.log("=" * 80)
        logger.log("ERROR")
        logger.log("=" * 80)
        logger.log(f"{type(e).__name__}: {e}")
        import traceback
        logger.log(traceback.format_exc())
        raise


if __name__ == "__main__":
    raise SystemExit(main())
