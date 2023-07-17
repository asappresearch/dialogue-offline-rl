from typing import Any, Dict, List, Set

import torch
from transformers.data.data_collator import (
    DataCollatorWithPadding,
    InputDataClass,
    torch_default_data_collator,
)

def mask_context_torch_data_collator(
    features: List[InputDataClass], 
    rep_start_tokenid: int, 
    additional_mask_tokens: Set[int] = set(),
) -> Dict[str, Any]:
    pad_value = -100
    batch = torch_default_data_collator(features)
    for i in range(len(batch["labels"])):
        orig_labels = batch["labels"][i]
        pad_mask = torch.zeros_like(orig_labels).to(bool)
        for idx, label in enumerate(orig_labels):
            if label.item() in additional_mask_tokens:
                pad_mask[idx] = True
            if label.item() == rep_start_tokenid:
                pad_mask[:idx+1] = True
                
        batch["labels"][i][pad_mask] = pad_value
    return batch