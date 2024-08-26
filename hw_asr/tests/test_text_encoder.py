import unittest
import torch

from hw_asr.text_encoder.ctc_char_text_encoder import CTCCharTextEncoder


class TestTextEncoder(unittest.TestCase):
    def test_ctc_decode(self):
        text_encoder = CTCCharTextEncoder()
        text = "i^^ ^w^i^sss^hhh^   i ^^^s^t^aaaar^teee^d " \
               "dddddd^oooo^in^g tttttttth^iiiis h^^^^^^^^w^ e^a^r^li^er"
        true_text = "i wish i started doing this hw earlier"
        inds = [text_encoder.char2ind[c] for c in text]
        decoded_text = text_encoder.ctc_decode(inds)
        self.assertIn(decoded_text, true_text)

    def test_beam_search_empty_input(self):
        text_encoder = CTCCharTextEncoder()
        probs = torch.tensor([])
        probs_length = 0
        beam_size = 10
        hypos = text_encoder.ctc_beam_search(probs, probs_length, beam_size)
        self.assertEqual(len(hypos), 0)

    def test_beam_search_single_char_input(self):
        text_encoder = CTCCharTextEncoder()
        voc_size = len(text_encoder.ind2char)
        probs = torch.tensor([[0.0] * voc_size])  
        probs[0, text_encoder.char2ind['d']] = 1.0  
        probs_length = 1
        beam_size = 1
        hypos = text_encoder.ctc_beam_search(probs, probs_length, beam_size)
        self.assertEqual(len(hypos), 1)
        self.assertEqual(hypos[0].text, "d")

    def test_beam_search_multi_char_input(self):
        text_encoder = CTCCharTextEncoder()
        voc_size = len(text_encoder.ind2char)
        probs = torch.tensor([[0.0] * voc_size, [0.0] * voc_size, [0.0] * voc_size]) 
        probs[0, text_encoder.char2ind['d']] = 1.0  
        probs[1, text_encoder.char2ind['d']] = 1.0  
        probs[2, text_encoder.char2ind['d']] = 1.0  
        probs_length = 3
        beam_size = 2
        hypos = text_encoder.ctc_beam_search(probs, probs_length, beam_size)
        self.assertEqual(len(hypos), 2)
        self.assertIn("ddd", [hypo.text for hypo in hypos])

    def test_beam_search_beam_size(self):
        text_encoder = CTCCharTextEncoder()
        voc_size = len(text_encoder.ind2char)
        probs = torch.tensor([[0.0] * voc_size, [0.0] * voc_size, [0.0] * voc_size])  
        probs[0, text_encoder.char2ind['d']] = 1.0 
        probs[1, text_encoder.char2ind['d']] = 1.0  
        probs[2, text_encoder.char2ind['d']] = 1.0 
        probs_length = 3
        beam_size = 1
        hypos = text_encoder.ctc_beam_search(probs, probs_length, beam_size)
        self.assertEqual(len(hypos), 1)
        self.assertEqual(hypos[0].text, "ddd")


if __name__ == "__main__":
    unittest.main()