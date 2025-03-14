import torch
from ssma.model import SSMA_Model
from torch.utils.data import DataLoader, Dataset


class RandomTextDataset(Dataset):
    """
    Dummy evaluation dataset for language modeling.
    """
    def __init__(self, num_samples=1000, seq_len=50, vocab_size=10000):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = torch.randint(0, self.vocab_size, (self.seq_len,))
        target = torch.cat([x[1:], torch.tensor([0])])
        return x, target

def evaluate_model(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits, _ = model(inputs)
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
            loss = criterion(logits, targets)
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    print(f"Evaluation Loss: {avg_loss:.4f}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = 10000
    seq_len = 50
    batch_size = 32

    dataset = RandomTextDataset(num_samples=1000, seq_len=seq_len, vocab_size=vocab_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = SSMA_Model(d_model=512, num_layers=4, r=64, m=256, top_k=32,
                       vocab_size=vocab_size, max_seq_len=seq_len)
    model.to(device)
    model.load_state_dict(torch.load("ssma_model.pt", map_location=device))
    print("Loaded model checkpoint.")
    evaluate_model(model, dataloader, device)

if __name__ == "__main__":
    main()
