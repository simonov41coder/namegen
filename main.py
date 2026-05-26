"""
Char-level LSTM Name Generator
Usage:
  python namegen.py train --data names.txt --model model.pt
  python namegen.py generate --model model.pt --count 10
"""

import torch
import torch.nn as nn
import argparse
import random
import json
import os

# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class NameDataset:
    def __init__(self, names: list[str]):
        self.names = [n.strip().lower() for n in names if n.strip()]
        chars = sorted(set("".join(self.names)))
        self.vocab = ["<PAD>", "<SOS>", "<EOS>"] + chars
        self.c2i = {c: i for i, c in enumerate(self.vocab)}
        self.i2c = {i: c for c, i in self.c2i.items()}
        self.pad_idx = self.c2i["<PAD>"]
        self.sos_idx = self.c2i["<SOS>"]
        self.eos_idx = self.c2i["<EOS>"]

    def encode(self, name: str) -> list[int]:
        return [self.sos_idx] + [self.c2i[c] for c in name] + [self.eos_idx]

    def decode(self, indices: list[int]) -> str:
        special = {self.pad_idx, self.sos_idx, self.eos_idx}
        return "".join(self.i2c[i] for i in indices if i not in special)

    def get_pairs(self):
        """Return (input_seq, target_seq) pairs for all names."""
        pairs = []
        for name in self.names:
            enc = self.encode(name)
            x = torch.tensor(enc[:-1], dtype=torch.long)
            y = torch.tensor(enc[1:], dtype=torch.long)
            pairs.append((x, y))
        return pairs

    def vocab_size(self):
        return len(self.vocab)

    def save_vocab(self, path: str):
        with open(path, "w") as f:
            json.dump({"vocab": self.vocab, "names": self.names}, f)

    @classmethod
    def load_vocab(cls, path: str):
        with open(path) as f:
            data = json.load(f)
        obj = cls.__new__(cls)
        obj.names = data["names"]
        obj.vocab = data["vocab"]
        obj.c2i = {c: i for i, c in enumerate(obj.vocab)}
        obj.i2c = {i: c for c, i in obj.c2i.items()}
        obj.pad_idx = obj.c2i["<PAD>"]
        obj.sos_idx = obj.c2i["<SOS>"]
        obj.eos_idx = obj.c2i["<EOS>"]
        return obj


# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────

class NameLSTM(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 64, hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, x, hidden=None):
        x = self.embed(x)
        out, hidden = self.lstm(x, hidden)
        logits = self.fc(out)
        return logits, hidden


# ─────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────

def train(data_path: str, model_path: str, epochs: int = 200, lr: float = 0.003):
    with open(data_path) as f:
        names = f.readlines()

    dataset = NameDataset(names)
    pairs = dataset.get_pairs()
    vocab_path = model_path.replace(".pt", "_vocab.json")
    dataset.save_vocab(vocab_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥  Device: {device} | Vocab size: {dataset.vocab_size()} | Names: {len(dataset.names)}")

    model = NameLSTM(dataset.vocab_size()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1, epochs + 1):
        random.shuffle(pairs)
        total_loss = 0.0

        for x, y in pairs:
            x, y = x.unsqueeze(0).to(device), y.unsqueeze(0).to(device)
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits.view(-1, dataset.vocab_size()), y.view(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()

        if epoch % 20 == 0 or epoch == 1:
            avg = total_loss / len(pairs)
            print(f"Epoch {epoch:>4}/{epochs} | Loss: {avg:.4f}")

    torch.save(model.state_dict(), model_path)
    print(f"\n✅ Model saved → {model_path}")
    print(f"✅ Vocab saved → {vocab_path}")


# ─────────────────────────────────────────────
# Generate
# ─────────────────────────────────────────────

def generate(model_path: str, count: int = 10, temperature: float = 0.8, max_len: int = 12):
    vocab_path = model_path.replace(".pt", "_vocab.json")
    dataset = NameDataset.load_vocab(vocab_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = NameLSTM(dataset.vocab_size()).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    results = []
    with torch.no_grad():
        while len(results) < count:
            x = torch.tensor([[dataset.sos_idx]], dtype=torch.long).to(device)
            hidden = None
            chars = []

            for _ in range(max_len):
                logits, hidden = model(x, hidden)
                logits = logits[:, -1, :] / temperature
                probs = torch.softmax(logits, dim=-1)
                next_idx = torch.multinomial(probs, 1).item()

                if next_idx == dataset.eos_idx:
                    break
                if next_idx in (dataset.pad_idx, dataset.sos_idx):
                    continue

                chars.append(dataset.i2c[next_idx])
                x = torch.tensor([[next_idx]], dtype=torch.long).to(device)

            name = "".join(chars).capitalize()
            if len(name) >= 2 and name.lower() not in dataset.names:
                results.append(name)

    print(f"\n✨ Generated Names (temp={temperature}):")
    for i, name in enumerate(results, 1):
        print(f"  {i:>3}. {name}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Char-level LSTM Name Generator")
    sub = parser.add_subparsers(dest="cmd")

    t = sub.add_parser("train", help="Train the model")
    t.add_argument("--data", required=True, help="Path to names.txt (one name per line)")
    t.add_argument("--model", default="model.pt", help="Output model path")
    t.add_argument("--epochs", type=int, default=200)
    t.add_argument("--lr", type=float, default=0.003)

    g = sub.add_parser("generate", help="Generate names")
    g.add_argument("--model", default="model.pt", help="Trained model path")
    g.add_argument("--count", type=int, default=10)
    g.add_argument("--temp", type=float, default=0.8, help="Sampling temperature (0.5=conservative, 1.2=creative)")

    args = parser.parse_args()

    if args.cmd == "train":
        train(args.data, args.model, args.epochs, args.lr)
    elif args.cmd == "generate":
        generate(args.model, args.count, args.temp)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

