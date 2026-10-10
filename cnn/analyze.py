"""Builds the results table and figures from results/*.json."""
import json, glob, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = {json.load(open(f))["tag"]: json.load(open(f)) for f in sorted(glob.glob("results/*.json"))}
for t, d in R.items():
    print(f"{t:20s} acc {d['acc_mean']:.3f}±{d['acc_std']:.3f}  macroF1 {d['macro_f1_mean']:.3f}±{d['macro_f1_std']:.3f}")
d = R["cnn_output_none"]; cls = d["runs"][0]["classes"]
cm = np.sum([r["confusion"] for r in d["runs"]], axis=0); print(cls); print(cm)
for c in cls:
    g = lambda k: np.mean([r["report"][c][k] for r in d["runs"]])
    print(f"{c}: precision {g('precision'):.3f} recall {g('recall'):.3f} f1 {g('f1-score'):.3f}")
cmn = cm / cm.sum(1, keepdims=True)
fig, ax = plt.subplots(figsize=(4.2, 3.8)); ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(3), cls); ax.set_yticks(range(3), cls); ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("CNN, output only (3 seeds pooled)")
for i in range(3):
    for j in range(3): ax.text(j, i, f"{cmn[i,j]:.2f}", ha="center", va="center", color="white" if cmn[i, j] > .5 else "black")
plt.tight_layout(); plt.savefig("confusion_cnn_output.png", dpi=200)
tags = ["cnn_output_none", "cnn_input_none", "cnn_both_none", "cnn_output_format", "cnn_output_punct", "cnn_output_length"]
lab = ["Output", "Input", "Input+\nOutput", "Output,\nno format", "Output,\nno punct", "Output,\n50 tokens"]
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.bar(lab, [R[t]["acc_mean"] for t in tags], yerr=[R[t]["acc_std"] for t in tags], color=["#4477aa"] * 3 + ["#cc6677"] * 3, capsize=3)
ax.axhline(1 / 3, ls="--", color="gray"); ax.set_ylabel("Test accuracy"); ax.set_ylim(0, 1); plt.xticks(fontsize=8)
plt.tight_layout(); plt.savefig("accuracy_cnn_settings.png", dpi=200)
