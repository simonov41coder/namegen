"""
Char-level LSTM Name Generator — Keras + TPU/GPU
Usage:
  python namegen_keras.py train --data names.txt --model model.keras
  python namegen_keras.py generate --model model.keras --count 10 --temp 0.8
"""

import os
import json
import argparse
import random
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress TF noise
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ─────────────────────────────────────────────
# TPU / GPU strategy
# ─────────────────────────────────────────────

def get_strategy():
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
        print(f"🚀 TPU detected — {strategy.num_replicas_in_sync} cores")
        return strategy
    except Exception:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            print(f"🖥  GPU detected — {len(gpus)} device(s)")
        else:
            print("💻 No TPU/GPU found — using CPU")
        return tf.distribute.get_strategy()  # default (CPU/GPU)


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class NameDataset:
    PAD, SOS, EOS = "<PAD>", "<SOS>", "<EOS>"

    def __init__(self, names: list[str]):
        self.names = [n.strip().lower() for n in names if n.strip()]
        chars = sorted(set("".join(self.names)))
        self.vocab = [self.PAD, self.SOS, self.EOS] + chars
        self.c2i = {c: i for i, c in enumerate(self.vocab)}
        self.i2c = {i: c for c, i in self.c2i.items()}
        self.pad_idx = self.c2i[self.PAD]
        self.sos_idx = self.c2i[self.SOS]
        self.eos_idx = self.c2i[self.EOS]

    def encode(self, name: str) -> list[int]:
        return [self.sos_idx] + [self.c2i[c] for c in name] + [self.eos_idx]

    def decode(self, indices: list[int]) -> str:
        special = {self.pad_idx, self.sos_idx, self.eos_idx}
        return "".join(self.i2c[i] for i in indices if i not in special)

    def vocab_size(self) -> int:
        return len(self.vocab)

    def build_dataset(self, batch_size: int = 32):
        """Build a padded tf.data.Dataset of (x, y) pairs."""
        xs, ys = [], []
        for name in self.names:
            enc = self.encode(name)
            xs.append(enc[:-1])
            ys.append(enc[1:])

        # pad sequences to same length
        max_len = max(len(s) for s in xs)
        xs = keras.preprocessing.sequence.pad_sequences(xs, maxlen=max_len, padding="post", value=self.pad_idx)
        ys = keras.preprocessing.sequence.pad_sequences(ys, maxlen=max_len, padding="post", value=self.pad_idx)

        ds = tf.data.Dataset.from_tensor_slices((xs, ys))
        ds = ds.shuffle(len(self.names)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds, max_len

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({"vocab": self.vocab, "names": self.names}, f)

    @classmethod
    def load(cls, path: str):
        with open(path) as f:
            data = json.load(f)
        obj = cls.__new__(cls)
        obj.names = data["names"]
        obj.vocab = data["vocab"]
        obj.c2i = {c: i for i, c in enumerate(obj.vocab)}
        obj.i2c = {i: c for c, i in obj.c2i.items()}
        obj.pad_idx = obj.c2i[cls.PAD]
        obj.sos_idx = obj.c2i[cls.SOS]
        obj.eos_idx = obj.c2i[cls.EOS]
        return obj


# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────

def build_model(vocab_size: int, embed_dim: int = 64, hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.3):
    inp = keras.Input(shape=(None,), dtype="int32", name="input")
    x = layers.Embedding(vocab_size, embed_dim, mask_zero=True, name="embed")(inp)

    for i in range(num_layers):
        return_seq = True  # always return sequences for stacked LSTM
        x = layers.LSTM(
            hidden_dim,
            return_sequences=return_seq,
            dropout=dropout,
            recurrent_dropout=0.0,  # recurrent dropout is slow on TPU, keep 0
            name=f"lstm_{i+1}"
        )(x)

    x = layers.Dropout(dropout)(x)
    out = layers.Dense(vocab_size, name="logits")(x)

    model = keras.Model(inputs=inp, outputs=out, name="NameLSTM")
    return model


# ─────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────

def train(data_path: str, model_path: str, epochs: int = 200, batch_size: int = 32, lr: float = 0.003):
    with open(data_path) as f:
        names = f.readlines()

    dataset = NameDataset(names)
    vocab_path = model_path.replace(".keras", "_vocab.json")
    dataset.save(vocab_path)

    ds, max_len = dataset.build_dataset(batch_size)
    print(f"📦 Names: {len(dataset.names)} | Vocab: {dataset.vocab_size()} | Max len: {max_len}")

    strategy = get_strategy()

    with strategy.scope():
        model = build_model(dataset.vocab_size())
        model.compile(
            optimizer=keras.optimizers.Adam(lr),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True, ignore_class=dataset.pad_idx),
            metrics=["accuracy"]
        )

    model.summary()

    callbacks = [
        keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=20, verbose=1),
        keras.callbacks.EarlyStopping(monitor="loss", patience=40, restore_best_weights=True, verbose=1),
    ]

    model.fit(ds, epochs=epochs, callbacks=callbacks)
    model.save(model_path)
    print(f"\n✅ Model saved → {model_path}")
    print(f"✅ Vocab saved → {vocab_path}")


# ─────────────────────────────────────────────
# Generate
# ─────────────────────────────────────────────

def generate(model_path: str, count: int = 10, temperature: float = 0.8, max_len: int = 12):
    vocab_path = model_path.replace(".keras", "_vocab.json")
    dataset = NameDataset.load(vocab_path)
    model = keras.models.load_model(model_path)

    results = []
    while len(results) < count:
        indices = [dataset.sos_idx]

        for _ in range(max_len):
            x = np.array([indices])
            logits = model.predict(x, verbose=0)[0, -1]  # last timestep
            logits = logits / temperature
            # softmax + sample
            logits -= np.max(logits)
            probs = np.exp(logits) / np.sum(np.exp(logits))
            next_idx = np.random.choice(len(probs), p=probs)

            if next_idx == dataset.eos_idx:
                break
            if next_idx in (dataset.pad_idx, dataset.sos_idx):
                continue

            indices.append(next_idx)

        name = dataset.decode(indices).capitalize()
        if len(name) >= 2:
            results.append(name)

    print(f"\n✨ Generated Names (temp={temperature}):")
    for i, name in enumerate(results, 1):
        print(f"  {i:>3}. {name}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Char-level LSTM Name Generator (Keras + TPU)")
    sub = parser.add_subparsers(dest="cmd")

    t = sub.add_parser("train")
    t.add_argument("--data", required=True, help="Path to names.txt")
    t.add_argument("--model", default="model.keras", help="Output model path")
    t.add_argument("--epochs", type=int, default=200)
    t.add_argument("--batch", type=int, default=32)
    t.add_argument("--lr", type=float, default=0.003)

    g = sub.add_parser("generate")
    g.add_argument("--model", default="model.keras", help="Trained model path")
    g.add_argument("--count", type=int, default=10)
    g.add_argument("--temp", type=float, default=0.8)

    args = parser.parse_args()

    if args.cmd == "train":
        train(args.data, args.model, args.epochs, args.batch, args.lr)
    elif args.cmd == "generate":
        generate(args.model, args.count, args.temp)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

