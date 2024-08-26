from typing import List, NamedTuple

import torch

from .char_text_encoder import CharTextEncoder


class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str] = None):
        super().__init__(alphabet)
        vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def ctc_decode(self, inds: List[int]) -> str:
        chars = [self.ind2char[ind] for ind in inds]

        result = []
        prev_char = None
        for char in chars:
            if char != self.EMPTY_TOK and char != prev_char:
                result.append(char)
            prev_char = char

        return ''.join(result)
    
    def ctc_beam_search(self, probs: torch.tensor, probs_length: int,
                        beam_size: int = 100) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
  
        if probs.numel() == 0:
            return []
        
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
        hypos: List[Hypothesis] = []
        
        hypos = [Hypothesis("", 1.0)]

        for step in range(probs_length):
            new_hypos = []
            for hypo in hypos:
                for next_char_idx in range(voc_size):
                    next_char = self.ind2char[next_char_idx]
                    next_prob = probs[step, next_char_idx]
                    new_text = hypo.text + next_char
                    new_prob = hypo.prob * next_prob
                    new_hypos.append(Hypothesis(new_text, new_prob))
            hypos = sorted(new_hypos, key=lambda x: x.prob, reverse=True)[:beam_size]

        return sorted(hypos, key=lambda x: x.prob, reverse=True)
