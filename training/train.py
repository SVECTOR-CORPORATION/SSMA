import time

import torch
import torch.nn as nn
import torch.optim as optim
from ssma.model import SSMA_Model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class RandomTextDataset(Dataset):
    """
    Dummy dataset for language modeling.
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

def train_model(model, dataloader, optimizer, criterion, device, epoch):
    model.train()
    running_loss = 0.0
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}")
    for batch_idx, (inputs, targets) in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        logits, _ = model(inputs)
        logits = logits.view(-1, logits.size(-1))
        targets = targets.view(-1)
        loss = criterion(logits, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    avg_loss = running_loss / len(dataloader)
    return avg_loss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = 10000
    seq_len = 50
    batch_size = 32
    num_epochs = 5
    lr = 1e-3

    dataset = RandomTextDataset(num_samples=5000, seq_len=seq_len, vocab_size=vocab_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SSMA_Model(d_model=512, num_layers=4, r=64, m=256, top_k=32,
                       vocab_size=vocab_size, max_seq_len=seq_len)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        avg_loss = train_model(model, dataloader, optimizer, criterion, device, epoch)
        duration = time.time() - start_time
        print(f"\nEpoch {epoch}/{num_epochs} completed in {duration:.2f}s, Average Loss: {avg_loss:.4f}\n")
    torch.save(model.state_dict(), "ssma_model.pt")
    print("Model saved as ssma_model.pt")

if __name__ == "__main__":
    main()
