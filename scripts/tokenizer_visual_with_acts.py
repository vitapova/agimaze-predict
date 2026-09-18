"""Modified visual tokenizer that includes <ACT>...</ACT> content in loss.

This is for training on rational-agent datasets where actions are predictable.
Identical to tokenizer_with_acts.py but uses visual-memory serialization.
"""

from __future__ import annotations

from typing import Sequence
from agimaze_predict.data.prepared import PreparedExample
from agimaze_predict.baselines.visual_transformer.tokenizer import (
    serialize_visual_example,
    SerializedVisualExample,
    IGNORE_INDEX,
    PAD_TOKEN_ID,
    _canvas,
    BYTE_VOCAB_SIZE,
)


def find_act_content_ranges(token_ids: list[int]) -> list[tuple[int, int]]:
    """Find all <ACT>...</ACT> content byte ranges (not including tags).
    
    Returns list of (start, end) where start is first content byte after <ACT>
    and end is first byte of </ACT>.
    """
    
    #<ACT> = [60, 65, 67, 84, 62]
    # </ACT> = [60, 47, 65, 67, 84, 62]
    act_open = [60, 65, 67, 84, 62]  # b'<ACT>'
    act_close = [60, 47, 65, 67, 84, 62]  # b'</ACT>'
    
    ranges = []
    i = 0
    
    while i < len(token_ids):
        # Look for <ACT>
        if i + len(act_open) <= len(token_ids):
            if token_ids[i:i+len(act_open)] == act_open:
                content_start = i + len(act_open)
                
                # Find matching </ACT>
                j = content_start
                while j + len(act_close) <= len(token_ids):
                    if token_ids[j:j+len(act_close)] == act_close:
                        content_end = j
                        ranges.append((content_start, content_end))
                        i = j + len(act_close)
                        break
                    j += 1
                else:
                    # No closing tag found, skip
                    i += 1
            else:
                i += 1
        else:
            i += 1
    
    return ranges


def collate_visual_examples_with_actions(
    examples: Sequence[PreparedExample],
    *,
    context_length: int,
    canvas_height: int,
    canvas_width: int,
) -> dict[str, list]:
    """Create padded arrays with loss on BOTH target AND <ACT> content.

    Returns dict with:
        - input_ids: text token sequences
        - labels: target labels (combined ACT + POS)
        - map_canvas: 2D character grids
        - event_positions: trigger positions for visual updates
        - act_mask: binary mask (1 = ACT token, 0 = other)
        - pos_mask: binary mask (1 = POS token, 0 = other)
    """

    if not examples:
        raise ValueError("cannot collate an empty batch")
    if context_length < 2:
        raise ValueError("context_length must be at least 2")

    serialized = [serialize_visual_example(example) for example in examples]
    lengths = [len(item.token_ids) for item in serialized]
    longest = max(lengths)
    if longest > context_length:
        raise ValueError(
            f"serialized example length {longest} exceeds context_length {context_length}; "
            "increase --context-length"
        )

    # Pad token sequences
    width = longest - 1
    input_ids: list[list[int]] = []
    labels: list[list[int]] = []
    act_masks: list[list[int]] = []
    pos_masks: list[list[int]] = []

    for item in serialized:
        ids = item.token_ids
        row_input = ids[:-1] + [PAD_TOKEN_ID] * (width - (len(ids) - 1))
        row_labels = [IGNORE_INDEX] * width
        row_act_mask = [0] * width
        row_pos_mask = [0] * width
        
        # POS target (original behavior)
        for index in range(item.target_start - 1, len(ids) - 1):
            row_labels[index] = ids[index + 1]
            row_pos_mask[index] = 1
        
        # ACT content (NEW)
        # Include closing </ACT> tag in loss so model learns to stop properly
        act_ranges = find_act_content_ranges(ids)
        for content_start, content_end in act_ranges:
            # content_end points to first byte of </ACT>
            # We want to include the entire </ACT> tag (6 bytes)
            act_close_len = 6  # len("</ACT>")
            full_end = min(content_end + act_close_len, len(ids))
            
            # Predict content bytes + closing tag (causal: predict token i+1 from i)
            for index in range(content_start - 1, min(full_end - 1, len(ids) - 1)):
                row_labels[index] = ids[index + 1]
                row_act_mask[index] = 1
                # Remove from POS mask if accidentally overlapping
                row_pos_mask[index] = 0
        
        input_ids.append(row_input)
        labels.append(row_labels)
        act_masks.append(row_act_mask)
        pos_masks.append(row_pos_mask)

    # Build visual canvases using original _canvas function
    visual_maps: list[list[list[int]]] = []
    for item in serialized:
        canvas = _canvas(item.map_rows, height=canvas_height, width=canvas_width, blank_char=ord(" "))
        visual_maps.append(canvas)

    # Pad event positions and compute event counts
    max_events = max(len(item.event_positions) for item in serialized)
    event_positions: list[list[int]] = []
    event_counts: list[list[int]] = []
    
    for item in serialized:
        padded_positions = list(item.event_positions) + [-1] * (max_events - len(item.event_positions))
        event_positions.append(padded_positions)
        
        # Compute cumulative event count at each text position
        counts = []
        for position in range(width):
            counts.append(sum(event <= position for event in item.event_positions))
        event_counts.append(counts)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "visual_maps": visual_maps,
        "event_positions": event_positions,
        "event_counts": event_counts,
        "act_mask": act_masks,
        "pos_mask": pos_masks,
    }
