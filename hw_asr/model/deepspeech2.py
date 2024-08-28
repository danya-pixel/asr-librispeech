from torch import nn, Tensor
from typing import Tuple
from hw_asr.base import BaseModel

import torch
from .convolution import DeepSpeech2Extractor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class DeepSpeech2(BaseModel):
    def __init__(
            self,
            input_dim: int,
            n_class: int,
            cnn_out_channels: int,
            num_rnn_layers: int = 5,
            rnn_hidden_dim: int = 512,
            bidirectional: bool = True,
            device: torch.device = 'cuda',
    ):
        super(DeepSpeech2, self).__init__(rnn_hidden_dim, n_class)
        self.device = device
        self.conv = DeepSpeech2Extractor(input_dim, cnn_out_channels)
        
        self.rnn = nn.GRU(
            input_size=1024,
            hidden_size=rnn_hidden_dim,
            num_layers=num_rnn_layers,
            batch_first=True,
            bidirectional=bidirectional
        )


        self.fc = nn.Sequential(
            nn.LayerNorm(2 * rnn_hidden_dim),
            nn.Linear(2 * rnn_hidden_dim, n_class, bias=False),
        )

    def forward(self, **batch) -> Tuple[Tensor, Tensor]:
        outputs, output_lengths = self.conv(**batch)
        packed_input = pack_padded_sequence(outputs, output_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.rnn(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        return {"logits": self.fc(output)}
    
    def transform_input_lengths(self, input_lengths):
        """
        Transform input lengths to output lengths after passing through the network.
        
        Args:
            input_lengths (Tensor): Original input lengths.
        
        Returns:
            Tensor: Transformed output lengths.
        """
        lengths = self.conv.conv._get_sequence_lengths(self.conv.conv.sequential[0],input_lengths)
        
        return lengths
