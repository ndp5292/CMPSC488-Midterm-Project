# Who Wrote It? Identifying LLMs from Their Responses

CMPSC 448 midterm project. We train neural networks to predict which LLM family (GPT, Claude or Gemini) wrote a response, and study whether the user's prompt helps. The CNN is complete and evaluated. The RNN (LSTM) is in progress and its results are not yet in this repository.

## Team

- Nency Patel (team leader)
- Shreyansh
- Aarav Johri
- Jal Savani

## Repository layout

| Path | Contents |
|---|---|
| `data/` | `llm_dataset.csv` (full balanced set), `train.csv`, `validation.csv`, `test.csv` and the dataset README |
| `preprocessing/` | `Midterm_Dataset_Preprocessing.ipynb`, the code that builds the dataset |
| `cnn/` | TextCNN code, experiment scripts, results (`cnn/results/*.json`), figures and the CNN report section |

## Dataset

Source: Chatbot Arena Human Preference 55K, https://huggingface.co/datasets/lmarena-ai/arena-human-preference-55k (Apache 2.0).

- Each example is `(LLM_name, LLM_Input, LLM_output)`. `LLM_name` is the family (GPT, Claude or Gemini), taken from the model names in the Arena data. Each family therefore mixes several model versions.
- Cleaning: removed missing and empty prompts or responses and exact duplicates, then sampled 1,000 responses per family (3,000 total, balanced by construction).
- Split: train 2,400, validation 301, test 299, grouped by prompt so no prompt appears in more than one split.

## Research questions covered

- **RQ1:** Can we identify which LLM generated a response? (output only)
- **RQ2:** Does the prompt help? (input only, output only, input + output)
- **RQ4 (partial):** ablations that remove formatting, remove punctuation, or keep only the first 50 tokens of each response.

RQ3 (generalization across tasks) was not investigated because the dataset has no task labels.

## CNN model

A TextCNN (Kim, 2014) in `cnn/cnn_classifier.py`:

- Tokens are words, punctuation marks and newlines, so formatting is visible to the model. The vocabulary is built from the training split only (at most 30,000 tokens, minimum frequency 2). Responses are truncated or padded to 256 tokens and prompts to 64.
- 128-dimensional embedding trained from scratch, parallel 1D convolutions of widths 2, 3, 4 and 5 with 100 filters each, ReLU, max-over-time pooling, dropout 0.5 and a linear layer with 3 outputs. For input + output, one encoder per text and the features are concatenated.
- Adam, learning rate 1e-3, weight decay 1e-4, batch size 64, class-weighted cross-entropy, early stopping on validation macro-F1 (patience 4), at most 20 epochs. Every experiment uses seeds 0, 1 and 2.

## CNN results

Test set (299 examples), mean ± standard deviation over 3 seeds. Chance accuracy is 0.333.

| Setting | Accuracy | Macro-F1 |
|---|---|---|
| Output only (RQ1) | 0.803 ± 0.008 | 0.793 ± 0.009 |
| Input only (RQ2) | 0.313 ± 0.014 | 0.301 ± 0.022 |
| Input + output (RQ2) | 0.804 ± 0.004 | 0.796 ± 0.004 |
| Output, formatting removed (RQ4) | 0.749 ± 0.012 | 0.741 ± 0.012 |
| Output, punctuation removed (RQ4) | 0.766 ± 0.012 | 0.758 ± 0.011 |
| Output, first 50 tokens only (RQ4) | 0.773 ± 0.005 | 0.767 ± 0.005 |

Findings:

- The response alone identifies the LLM about 80% of the time. Per-class F1 for the output-only CNN is 0.92 for Claude, 0.74 for Gemini and 0.71 for GPT. Of the 177 errors in the pooled confusion matrix (3 seeds, 897 predictions), 125 are GPT and Gemini confused with each other.
- The prompt alone is at chance level and adding it to the response gives no gain. This holds in this dataset because splits share no prompts and the prompts come from a common pool of user questions.
- No single signal explains the performance. Removing formatting costs about 5 points, removing punctuation about 4 and keeping only 50 tokens about 3.
- With 299 test examples one standard error of accuracy is about 2.3 points, so small differences between ablations are within noise.

Figures: `cnn/confusion_cnn_output.png` and `cnn/accuracy_cnn_settings.png`.

## How to run

```
pip install -r cnn/requirements.txt
cd cnn
python cnn_classifier.py --train ../data/train.csv --val ../data/validation.csv --test ../data/test.csv --mode output
bash run_all.sh ../data     # every CNN experiment, about an hour on 2 CPU cores
python analyze.py           # results table and figures
```

Use `--mode input` or `--mode both` for the other RQ2 settings and `--ablate format`, `--ablate punct` or `--ablate length` for the RQ4 ablations.

To rebuild the dataset, open `preprocessing/Midterm_Dataset_Preprocessing.ipynb` and run all cells (it downloads the Arena data from Hugging Face).

## Use of LLM tools

AI assistants were used for coding help, debugging and drafting. All code, results and text were checked and are understood by the team.
