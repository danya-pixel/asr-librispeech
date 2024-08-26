import logging
from typing import List
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    result_batch = {}

    spectrograms = [item['spectrogram'] for item in dataset_items]
    max_len_spectrogram = max([s.shape[2] for s in spectrograms])

    padded_spectrograms = []
    for s in spectrograms:
        padded_spectrogram = F.pad(s, (0, max_len_spectrogram-s.shape[2]))
        padded_spectrograms.append(padded_spectrogram[0])
    
    result_batch['spectrogram'] = torch.stack(padded_spectrograms)
    

    text_encoded = [item['text_encoded'] for item in dataset_items]
    max_text_encoded_length = max(text.shape[1] for text in text_encoded)
    padded_text_encoded = []
    text_encoded_lengths = []
    for text in text_encoded:
        padded_text = torch.nn.functional.pad(text, (0, max_text_encoded_length - text.shape[1]), value=0)
        padded_text_encoded.append(padded_text[0])
        text_encoded_lengths.append(text.shape[1])
    result_batch['text_encoded'] = torch.stack(padded_text_encoded)
    result_batch['text_encoded_length'] = torch.tensor(text_encoded_lengths)

    result_batch['audio'] = [item['audio'] for item in dataset_items]
    result_batch['duration'] = [item['duration'] for item in dataset_items]
    result_batch['text'] = [item['text'] for item in dataset_items]
    result_batch['audio_path'] = [item['audio_path'] for item in dataset_items]

    return result_batch