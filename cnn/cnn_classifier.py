"""TextCNN for "Who Wrote It? Identifying LLMs from Their Responses".

Predicts LLM_name from (LLM_input, LLM_output).
  --mode output : RQ1 / RQ2 output only
  --mode input  : RQ2 input only
  --mode both   : RQ2 input + output (two CNN encoders, features concatenated)
Data: csv / json / jsonl / parquet with an LLM name, input and output column.
"""
import argparse, json, os, re, random, time
from collections import Counter
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold

NAME_COLS = ["LLM_name", "llm_name", "model", "model_name", "label", "llm"]
IN_COLS = ["LLM_Input", "LLM_input", "llm_input", "input", "prompt", "question"]
OUT_COLS = ["LLM_output", "llm_output", "output", "response", "answer"]
PAD, UNK = 0, 1
TOKEN_RE = re.compile(r"\n|\w+|[^\w\s]")


def pick(df, cands, given, what):
    if given:
        return given
    low = {c.lower(): c for c in df.columns}
    for c in cands:
        if c.lower() in low:
            return low[c.lower()]
    raise SystemExit(f"Could not find {what} column in {list(df.columns)}; pass it explicitly.")


def load(path, a):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv": df = pd.read_csv(path)
    elif ext == ".jsonl": df = pd.read_json(path, lines=True)
    elif ext == ".json": df = pd.read_json(path)
    elif ext == ".parquet": df = pd.read_parquet(path)
    else: raise SystemExit("unsupported file type")
    n, i, o = pick(df, NAME_COLS, a.name_col, "LLM name"), pick(df, IN_COLS, a.input_col, "input"), pick(df, OUT_COLS, a.output_col, "output")
    keep = [n, i, o] + ([a.task_col] if a.task_col else [])
    df = df[keep].dropna().copy()
    df.columns = ["name", "inp", "out"] + (["task"] if a.task_col else [])
    df["inp"], df["out"] = df["inp"].astype(str), df["out"].astype(str)
    before = len(df)
    df = df.drop_duplicates(["name", "inp", "out"]).reset_index(drop=True)
    print(f"loaded {before} rows, {len(df)} after dedupe; classes:\n{df['name'].value_counts().to_string()}")
    return df


def ablate(text, how):
    if how == "format":      # remove newlines + markdown symbols
        return re.sub(r"[*#`_|>\-]+", " ", text.replace("\n", " "))
    if how == "punct":       # remove all punctuation
        return re.sub(r"[^\w\s]", " ", text)
    return text


def tokenize(text, a):
    text = ablate(text, a.ablate)
    if a.lower: text = text.lower()
    toks = TOKEN_RE.findall(text)
    if a.ablate == "length": toks = toks[:50]
    return toks


class Vocab:
    def __init__(self, token_lists, max_size, min_freq):
        c = Counter(t for toks in token_lists for t in toks)
        words = [w for w, f in c.most_common(max_size) if f >= min_freq]
        self.itos = ["<pad>", "<unk>"] + words
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def encode(self, toks, max_len):
        ids = [self.stoi.get(t, UNK) for t in toks[:max_len]]
        return ids + [PAD] * (max_len - len(ids))


class Encoder(nn.Module):
    """Kim (2014) TextCNN: embedding -> parallel conv1d (several widths) -> max-over-time."""
    def __init__(self, vocab_size, emb, n_filters, widths):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=PAD)
        self.convs = nn.ModuleList([nn.Conv1d(emb, n_filters, k, padding=k // 2) for k in widths])
        self.out_dim = n_filters * len(widths)

    def forward(self, x):
        e = self.emb(x).transpose(1, 2)                      # B x emb x L
        return torch.cat([torch.relu(c(e)).max(dim=2).values for c in self.convs], dim=1)


class TextCNN(nn.Module):
    def __init__(self, vocab_sizes, n_classes, emb=128, n_filters=100, widths=(2, 3, 4, 5), dropout=0.5):
        super().__init__()
        self.encs = nn.ModuleList([Encoder(v, emb, n_filters, widths) for v in vocab_sizes])
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(sum(e.out_dim for e in self.encs), n_classes)

    def forward(self, xs):
        return self.fc(self.drop(torch.cat([e(x) for e, x in zip(self.encs, xs)], dim=1)))


def split(df, seed, a):
    if "split" in df.columns:                # fixed splits supplied by --train/--val/--test
        return tuple(np.where(df["split"] == s)[0] for s in ("train", "val", "test"))
    y = df["name"].values
    if a.task_col and a.test_task:           # RQ3: train on other tasks, test on held-out task
        te = (df["task"] == a.test_task).values
        rest = np.where(~te)[0]
        sub = df.iloc[rest]
        g = pd.factorize(sub["inp"])[0]
        sgk = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
        tr_i, va_i = next(sgk.split(sub, sub["name"], g))
        return rest[tr_i], rest[va_i], np.where(te)[0]
    groups = pd.factorize(df["inp"])[0] if a.group_split else np.arange(len(df))
    sgk = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    folds = [te for _, te in sgk.split(df, y, groups)]
    test, val = folds[0], folds[1]
    train = np.setdiff1d(np.arange(len(df)), np.concatenate([test, val]))
    return train, val, test


def run(df, seed, a, dev):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    classes = sorted(df["name"].unique()); cid = {c: i for i, c in enumerate(classes)}
    y = df["name"].map(cid).values
    tr, va, te = split(df, seed, a)
    fields = {"output": ["out"], "input": ["inp"], "both": ["inp", "out"]}[a.mode]
    lens = {"inp": a.max_len_input, "out": a.max_len_output}
    toks = {f: [tokenize(t, a) for t in df[f]] for f in fields}
    vocabs = {f: Vocab([toks[f][i] for i in tr], a.max_vocab, a.min_freq) for f in fields}   # train only
    X = {f: torch.tensor([vocabs[f].encode(t, lens[f]) for t in toks[f]]) for f in fields}
    Y = torch.tensor(y)
    model = TextCNN([len(vocabs[f].itos) for f in fields], len(classes), a.emb, a.filters, tuple(a.widths), a.dropout).to(dev)
    counts = np.bincount(y[tr], minlength=len(classes))
    w = torch.tensor(len(tr) / (len(classes) * np.maximum(counts, 1)), dtype=torch.float, device=dev)
    lossf = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.wd)

    def batches(idx, shuffle):
        idx = np.random.permutation(idx) if shuffle else idx
        for s in range(0, len(idx), a.batch):
            b = idx[s:s + a.batch]
            yield [X[f][b].to(dev) for f in fields], Y[b].to(dev)

    def predict(idx):
        model.eval(); ps = []
        with torch.no_grad():
            for xs, _ in batches(idx, False): ps.append(model(xs).argmax(1).cpu())
        return torch.cat(ps).numpy()

    best, best_state, bad, hist = -1, None, 0, []
    for ep in range(1, a.epochs + 1):
        model.train(); tot = 0
        for xs, yb in batches(tr, True):
            opt.zero_grad(); l = lossf(model(xs), yb); l.backward(); opt.step(); tot += l.item() * len(yb)
        vf1 = f1_score(y[va], predict(va), average="macro")
        hist.append({"epoch": ep, "train_loss": tot / len(tr), "val_macro_f1": vf1})
        print(f"  seed {seed} ep {ep:2d} loss {tot/len(tr):.4f} val macroF1 {vf1:.4f}")
        if vf1 > best: best, bad, best_state = vf1, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience: break
    model.load_state_dict(best_state)
    pred = predict(te)
    res = {"seed": seed, "n_train": len(tr), "n_val": len(va), "n_test": len(te),
           "test_acc": accuracy_score(y[te], pred), "test_macro_f1": f1_score(y[te], pred, average="macro"),
           "val_macro_f1": best, "history": hist,
           "report": classification_report(y[te], pred, target_names=classes, output_dict=True, zero_division=0),
           "confusion": confusion_matrix(y[te], pred, labels=range(len(classes))).tolist(), "classes": classes,
           "n_params": sum(p.numel() for p in model.parameters())}
    print(f"seed {seed}: test acc {res['test_acc']:.4f} macroF1 {res['test_macro_f1']:.4f}")
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", help="single file (random grouped split)")
    p.add_argument("--train"); p.add_argument("--val"); p.add_argument("--test")  # use the team's fixed splits
    p.add_argument("--mode", choices=["output", "input", "both"], default="output")
    p.add_argument("--name_col"); p.add_argument("--input_col"); p.add_argument("--output_col")
    p.add_argument("--task_col"); p.add_argument("--test_task", help="RQ3: hold this task out for testing")
    p.add_argument("--ablate", choices=["none", "format", "punct", "length"], default="none", help="RQ4")
    p.add_argument("--no_group_split", dest="group_split", action="store_false")
    p.add_argument("--lower", action="store_true")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=20); p.add_argument("--patience", type=int, default=4)
    p.add_argument("--batch", type=int, default=64); p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4); p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--emb", type=int, default=128); p.add_argument("--filters", type=int, default=100)
    p.add_argument("--widths", type=int, nargs="+", default=[2, 3, 4, 5])
    p.add_argument("--max_len_output", type=int, default=256); p.add_argument("--max_len_input", type=int, default=64)
    p.add_argument("--max_vocab", type=int, default=30000); p.add_argument("--min_freq", type=int, default=2)
    p.add_argument("--outdir", default="results")
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if a.train:
        parts = []
        for nm, f in (("train", a.train), ("val", a.val), ("test", a.test)):
            d = load(f, a); d["split"] = nm; parts.append(d)
        df = pd.concat(parts, ignore_index=True)
    else:
        df = load(a.data, a)
    tag = f"cnn_{a.mode}_{a.ablate}" + (f"_hold-{a.test_task}" if a.test_task else "")
    runs = [run(df, s, a, dev) for s in a.seeds]
    accs = [r["test_acc"] for r in runs]; f1s = [r["test_macro_f1"] for r in runs]
    summary = {"tag": tag, "args": vars(a), "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
               "macro_f1_mean": float(np.mean(f1s)), "macro_f1_std": float(np.std(f1s)),
               "chance": 1 / df["name"].nunique(), "runs": runs}
    os.makedirs(a.outdir, exist_ok=True)
    json.dump(summary, open(os.path.join(a.outdir, tag + ".json"), "w"), indent=1)
    print(f"\n[{tag}] acc {summary['acc_mean']:.4f}±{summary['acc_std']:.4f}  macroF1 {summary['macro_f1_mean']:.4f}±{summary['macro_f1_std']:.4f}  (chance {summary['chance']:.3f})")


if __name__ == "__main__":
    main()
