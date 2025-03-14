import torch
from torch.utils.data import Dataset


class RandomTextDataset(Dataset):
    """
    A dummy dataset for language modeling.
    Each sample is a random sequence of token indices.
    """
    def __init__(self, num_samples=5000, seq_len=50, vocab_size=10000):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = torch.randint(0, self.vocab_size, (self.seq_len,))
        target = torch.cat([x[1:], torch.tensor([0])])
        return x, target
