import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
    
from defaults.featuring_defaults import *
from defaults.train_defaults import BATCH_SIZE
from .conll2003 import load_data
from .dataset import ConllDataset, PredictDataset
from .featuring import Featuring
from .tokenizer import tokenize

import numpy as np
import torch
from torch.nn.functional import pad
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

def _custom_collate_fn(batch):
    """
    Function for sorting and padding each batch data for packed_pad_sequence()
    in train, validation, test phases.
    """
    new_batch = {"features": {}}
    keys = batch[0]["features"].keys()
    sorted_batch = sorted(
        [sample for sample in batch],
        key=lambda x: x["features"]["global_vectors"].shape[0],
        reverse=True,
    )
    for key in keys:

        if key == "casing_vectors":
            new_batch["features"][key] = pad_sequence(
                [
                    torch.Tensor(np.array(sample["features"][key]))
                    for sample in sorted_batch
                ],
                batch_first=True,
            )

        elif key == "global_vectors":
            new_batch["features"][key] = pad_sequence(
                [
                    torch.Tensor(np.array(sample["features"][key]))
                    for sample in sorted_batch
                ],
                padding_value=1,
                batch_first=True,
            )
            
            new_batch["features"]["seq_lengths"] = [
                sample["features"][key].shape[0] for sample in sorted_batch
            ]

        elif key == "char_level_vectors":
            max_len_word = max([len(word) for word in sorted_batch[0]["features"][key]])
            max_len_sequence = sorted_batch[0]["features"]["global_vectors"].shape[0]

            pad_words = lambda words: pad_sequence(
                [torch.Tensor(word) for word in words], batch_first=True
            )
            padded_samples = [
                pad_words(sample["features"][key]) for sample in sorted_batch
            ]

            pad_tensor = lambda tensor: pad(
                tensor,
                (
                    0,
                    max_len_word - tensor.shape[1],
                    0,
                    max_len_sequence - tensor.shape[0],
                ),
                "constant",
                0,
            )
            new_batch["features"][key] = torch.stack(
                [pad_tensor(sample) for sample in padded_samples]
            )
            new_batch["features"]["seq_lengths"] = [
                sample.shape[0] for sample in padded_samples
            ]

    if sorted_batch[0].get("tags", -1) != -1:
        new_batch["tags"] = pad_sequence(
            [torch.Tensor(sample["tags"]) for sample in sorted_batch], batch_first=True
        )
    return new_batch


def load_transform_data(
    phase,
    use_char_level=USE_CHAR_LEVEL_EMBEDDING,
    use_casing=USE_CASING_FEATURE,
    batch_size=BATCH_SIZE,
):
    """
    Load raw data then transform it into features.
    """
    raw_dataset = load_data()[phase]

    datafeaturing = Featuring(use_char_level=use_char_level, use_casing=use_casing)

    dataset = ConllDataset(dataset=raw_dataset, preprocessing=datafeaturing)

    use_shuffle = phase == "train"

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=use_shuffle,
        collate_fn=_custom_collate_fn,
    )

    return dataloader


def load_from_array(
    sequences,
    use_char_level=USE_CHAR_LEVEL_EMBEDDING,
    use_casing=USE_CASING_FEATURE,
    batch_size=1,
):
    """
    Transform sequences to features.
    """
    sequences = tokenize(sequences)
    datafeaturing = Featuring(use_char_level=use_char_level, use_casing=use_casing)

    dataset = PredictDataset(dataset=sequences, preprocessing=datafeaturing)

    use_shuffle = False

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=use_shuffle,
        collate_fn=_custom_collate_fn,
    )
    return dataloader
