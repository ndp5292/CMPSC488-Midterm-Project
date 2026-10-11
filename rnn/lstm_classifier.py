"""
LSTM classifier for "Who Wrote It? Identifying LLMs from Their Responses"
CMPSC 448 midterm project -- RNN component (RQ1 + RQ2)

Reads the team's preprocessed splits (data/train.csv, data/validation.csv,
data/test.csv; columns: LLM_name, LLM_Input, LLM_output, source_id,
prompt_group) and trains a bidirectional LSTM to predict which LLM family
(Claude / GPT / Gemini) wrote a response.

RQ1  -> --mode output        (classify from LLM_output only)
RQ2  -> --mode input         (classify from the user's prompt only)
        --mode input_output  (prompt + [SEP] + response)
        --mode all           (runs all three and writes a comparison table)

Usage (local or Colab):
    pip install torch pandas scikit-learn matplotlib
    python lstm_classifier.py --data_dir data --mode all
    python lstm_classifier.py --data_dir data --mode output --epochs 20

Outputs go to --out_dir (default: results_lstm/):
    lstm_<mode>_metrics.json       accuracy, macro-F1, per-class report, confusion matrix
    lstm_<mode>_confusion.png      test-set confusion matrix
    lstm_<mode>_curves.png         train/val loss and val macro-F1 per epoch
    lstm_<mode>_best.pt            best checkpoint (by validation macro-F1)
    lstm_rq2_comparison.csv        only with --mode all
"""

import argparse
import json
import os
import random
import re
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

PAD, UNK, SEP = "<pad>", "<unk>", "<sep>"
NL = "<nl>"


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------
# Tokenization
# --------------------------------------------------------------------------
# Formatting is part of an LLM's "fingerprint" (e.g. Gemini uses **bold** far
# more often than Claude in this dataset), so the tokenizer keeps markdown
# markers, newlines and punctuation as their own tokens instead of stripping them.
TOKEN_RE = re.compile(
    r"\n"            # newlines (become <nl>)
    r"|```"          # code fences
    r"|\*\*"         # bold markers
    r"|#{1,6}"       # markdown headers
    r"|\d+\."        # numbered-list markers like "1."
    r"|[A-Za-z]+(?:'[A-Za-z]+)?"  # words, incl. contractions (don't, I'm)
    r"|\d+"          # numbers
    r"|[^\w\s]"      # any other single punctuation / symbol
)


def tokenize(text, lowercase=True):
    if not isinstance(text, str):
        return []
    if lowercase:
        text = text.lower()
    toks = TOKEN_RE.findall(text)
    out = []
    for t in toks:
        if t == "\n":
            out.append(NL)
        elif re.fullmatch(r"\d+\.", t):
            out.append("<num_item>")
        elif t.isdigit():
            out.append("<num>")
        else:
            out.append(t)
    return out


class Vocab:
    def __init__(self, token_lists, min_freq=2, max_size=30000):
        counter = Counter(tok for toks in token_lists for tok in toks)
        specials = [PAD, UNK, SEP]
        words = [w for w, c in counter.most_common() if c >= min_freq and w not in specials]
        words = words[: max_size - len(specials)]
        self.itos = specials + words
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, toks):
        unk = self.stoi[UNK]
        return [self.stoi.get(t, unk) for t in toks]


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
def build_tokens(df, mode, max_in, max_out, lowercase):
    """Return a list of token lists for the requested input mode."""
    seqs = []
    for inp, out in zip(df["LLM_Input"], df["LLM_output"]):
        if mode == "output":
            seqs.append(tokenize(out, lowercase)[:max_out])
        elif mode == "input":
            seqs.append(tokenize(inp, lowercase)[:max_in])
        elif mode == "input_output":
            seqs.append(tokenize(inp, lowercase)[:max_in] + [SEP] + tokenize(out, lowercase)[:max_out])
        else:
            raise ValueError(mode)
    return seqs


class TextDataset(Dataset):
    def __init__(self, token_lists, labels, vocab):
        self.ids = [vocab.encode(t) or [vocab.stoi[UNK]] for t in token_lists]  # never empty
        self.labels = labels

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return self.ids[i], self.labels[i]


def collate(batch):
    seqs, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    padded = torch.zeros(len(seqs), int(lengths.max()), dtype=torch.long)  # PAD id = 0
    for i, s in enumerate(seqs):
        padded[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return padded, lengths, torch.tensor(labels, dtype=torch.long)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
class LSTMClassifier(nn.Module):
    """
    Embedding -> BiLSTM -> [attention pooling ; max pooling] -> MLP -> logits

    * Bidirectional: each position sees both the left and right context.
    * Packed sequences: padding never enters the LSTM, so short and long
      responses are handled correctly within a batch.
    * Attention pooling: learns which time steps matter; the weights can be
      inspected for the RQ4 "what signals does the model use" analysis.
    * Max pooling: picks up strong local cues (e.g. a "**" or an apology
      phrase) wherever they occur in the response.
    """

    def __init__(self, vocab_size, num_classes, emb_dim=128, hidden=128,
                 layers=1, dropout=0.4, bidirectional=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.emb_drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            emb_dim, hidden, num_layers=layers, batch_first=True,
            bidirectional=bidirectional, dropout=dropout if layers > 1 else 0.0,
        )
        out_dim = hidden * (2 if bidirectional else 1)
        self.attn = nn.Linear(out_dim, 1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(out_dim * 2, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, num_classes),
        )

    def forward(self, x, lengths, return_attn=False):
        emb = self.emb_drop(self.embedding(x))
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.size(1))

        mask = (torch.arange(x.size(1), device=x.device)[None, :] < lengths.to(x.device)[:, None])

        scores = self.attn(out).squeeze(-1).masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=1)
        attn_pool = torch.bmm(weights.unsqueeze(1), out).squeeze(1)

        max_pool = out.masked_fill(~mask.unsqueeze(-1), float("-inf")).max(dim=1).values

        logits = self.head(torch.cat([attn_pool, max_pool], dim=1))
        return (logits, weights) if return_attn else logits


# --------------------------------------------------------------------------
# Train / evaluate
# --------------------------------------------------------------------------
def run_epoch(model, loader, device, criterion, optimizer=None, clip=1.0):
    training = optimizer is not None
    model.train(training)
    total_loss, preds, golds = 0.0, [], []
    with torch.set_grad_enabled(training):
        for x, lengths, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x, lengths)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
            total_loss += loss.item() * y.size(0)
            preds.extend(logits.argmax(1).cpu().tolist())
            golds.extend(y.cpu().tolist())
    n = len(golds)
    return total_loss / n, accuracy_score(golds, preds), f1_score(golds, preds, average="macro"), preds, golds


def plot_curves(history, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(ep, [h["train_loss"] for h in history], label="train")
    ax[0].plot(ep, [h["val_loss"] for h in history], label="validation")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("cross-entropy loss"); ax[0].legend()
    ax[1].plot(ep, [h["train_f1"] for h in history], label="train")
    ax[1].plot(ep, [h["val_f1"] for h in history], label="validation")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("macro-F1"); ax[1].legend()
    from matplotlib.ticker import MaxNLocator
    for a in ax:
        a.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(cm, labels, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i][j], ha="center", va="center",
                    color="white" if cm[i][j] > cm.max() / 2 else "black")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def train_one_mode(mode, train_df, val_df, test_df, args, device):
    set_seed(args.seed)
    print(f"\n{'=' * 70}\nLSTM | mode = {mode}\n{'=' * 70}")

    labels = sorted(train_df["LLM_name"].unique())
    lab2id = {l: i for i, l in enumerate(labels)}
    y_tr = [lab2id[l] for l in train_df["LLM_name"]]
    y_va = [lab2id[l] for l in val_df["LLM_name"]]
    y_te = [lab2id[l] for l in test_df["LLM_name"]]

    max_in = args.max_input_len
    max_out = args.max_output_len
    lc = not args.keep_case
    tok_tr = build_tokens(train_df, mode, max_in, max_out, lc)
    tok_va = build_tokens(val_df, mode, max_in, max_out, lc)
    tok_te = build_tokens(test_df, mode, max_in, max_out, lc)

    vocab = Vocab(tok_tr, min_freq=args.min_freq, max_size=args.max_vocab)  # train split only
    lens = [len(t) for t in tok_tr]
    print(f"classes: {labels}")
    print(f"vocab size: {len(vocab)} | train seq len median {int(np.median(lens))}, "
          f"95th pct {int(np.percentile(lens, 95))}, max {max(lens)}")

    g = torch.Generator().manual_seed(args.seed)
    dl_tr = DataLoader(TextDataset(tok_tr, y_tr, vocab), batch_size=args.batch_size,
                       shuffle=True, collate_fn=collate, generator=g)
    dl_va = DataLoader(TextDataset(tok_va, y_va, vocab), batch_size=args.batch_size, collate_fn=collate)
    dl_te = DataLoader(TextDataset(tok_te, y_te, vocab), batch_size=args.batch_size, collate_fn=collate)

    model = LSTMClassifier(len(vocab), len(labels), emb_dim=args.emb_dim, hidden=args.hidden,
                           layers=args.layers, dropout=args.dropout,
                           bidirectional=not args.unidirectional).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable parameters: {n_params:,} | device: {device}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, f"lstm_{mode}_best.pt")
    best_f1, best_epoch, bad_epochs, history = -1.0, 0, 0, []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc, tr_f1, _, _ = run_epoch(model, dl_tr, device, criterion, optimizer, args.clip)
        va_loss, va_acc, va_f1, _, _ = run_epoch(model, dl_va, device, criterion)
        scheduler.step(va_f1)
        history.append(dict(epoch=epoch, train_loss=tr_loss, train_acc=tr_acc, train_f1=tr_f1,
                            val_loss=va_loss, val_acc=va_acc, val_f1=va_f1,
                            lr=optimizer.param_groups[0]["lr"]))
        print(f"epoch {epoch:2d} | train loss {tr_loss:.4f} acc {tr_acc:.3f} f1 {tr_f1:.3f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.3f} f1 {va_f1:.3f} | {time.time() - t0:.1f}s")

        if va_f1 > best_f1:
            best_f1, best_epoch, bad_epochs = va_f1, epoch, 0
            torch.save({"model": model.state_dict(), "vocab": vocab.itos, "labels": labels,
                        "args": vars(args), "mode": mode}, ckpt_path)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"early stopping (no val macro-F1 improvement for {args.patience} epochs)")
                break

    # Evaluate the best checkpoint on the test set
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    te_loss, te_acc, te_f1, preds, golds = run_epoch(model, dl_te, device, criterion)
    cm = confusion_matrix(golds, preds, labels=list(range(len(labels))))
    report = classification_report(golds, preds, target_names=labels, digits=4, output_dict=True)

    print(f"\nbest epoch {best_epoch} (val macro-F1 {best_f1:.4f})")
    print(f"TEST  accuracy {te_acc:.4f} | macro-F1 {te_f1:.4f}")
    print(classification_report(golds, preds, target_names=labels, digits=4))
    print("confusion matrix (rows = true, cols = predicted):", labels)
    print(cm)

    metrics = {
        "model": "BiLSTM" if not args.unidirectional else "LSTM",
        "mode": mode,
        "labels": labels,
        "best_epoch": best_epoch,
        "val_macro_f1": best_f1,
        "test_accuracy": te_acc,
        "test_macro_f1": te_f1,
        "test_loss": te_loss,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "vocab_size": len(vocab),
        "trainable_params": n_params,
        "history": history,
        "hyperparameters": vars(args),
        "n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df),
    }
    with open(os.path.join(args.out_dir, f"lstm_{mode}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    if not args.no_plots:
        plot_curves(history, os.path.join(args.out_dir, f"lstm_{mode}_curves.png"), f"BiLSTM training ({mode})")
        plot_confusion(cm, labels, os.path.join(args.out_dir, f"lstm_{mode}_confusion.png"),
                       f"BiLSTM test ({mode})\nacc {te_acc:.3f}, macro-F1 {te_f1:.3f}")
    return metrics


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def load_split(data_dir, name):
    df = pd.read_csv(os.path.join(data_dir, name))
    df = df.dropna(subset=["LLM_name", "LLM_output"]).copy()
    df["LLM_Input"] = df["LLM_Input"].fillna("").astype(str)
    df["LLM_output"] = df["LLM_output"].astype(str)
    return df.reset_index(drop=True)


def parse_args():
    p = argparse.ArgumentParser(description="BiLSTM LLM-authorship classifier (CMPSC 448)")
    p.add_argument("--data_dir", default="data", help="folder with train.csv, validation.csv, test.csv")
    p.add_argument("--out_dir", default="results_lstm")
    p.add_argument("--mode", default="all", choices=["output", "input", "input_output", "all"])
    p.add_argument("--max_input_len", type=int, default=128, help="max prompt tokens")
    p.add_argument("--max_output_len", type=int, default=512, help="max response tokens")
    p.add_argument("--min_freq", type=int, default=2)
    p.add_argument("--max_vocab", type=int, default=30000)
    p.add_argument("--keep_case", action="store_true", help="do not lowercase text")
    p.add_argument("--emb_dim", type=int, default=128)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--unidirectional", action="store_true", help="plain LSTM instead of BiLSTM")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_plots", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = pick_device()
    train_df = load_split(args.data_dir, "train.csv")
    val_df = load_split(args.data_dir, "validation.csv")
    test_df = load_split(args.data_dir, "test.csv")
    print(f"train {len(train_df)} | val {len(val_df)} | test {len(test_df)}")
    print("train label counts:", train_df["LLM_name"].value_counts().to_dict())

    modes = ["output", "input", "input_output"] if args.mode == "all" else [args.mode]
    results = [train_one_mode(m, train_df, val_df, test_df, args, device) for m in modes]

    if len(results) > 1:
        names = {"output": "Output only (RQ1)", "input": "Input only", "input_output": "Input + Output"}
        table = pd.DataFrame([{
            "setting": names[r["mode"]],
            "test_accuracy": round(r["test_accuracy"], 4),
            "test_macro_f1": round(r["test_macro_f1"], 4),
            "val_macro_f1": round(r["val_macro_f1"], 4),
            "best_epoch": r["best_epoch"],
        } for r in results])
        table.to_csv(os.path.join(args.out_dir, "lstm_rq2_comparison.csv"), index=False)
        print("\nRQ2 comparison (BiLSTM, test set; chance = %.3f)" % (1 / len(results[0]["labels"])))
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
