# Who Wrote It? Identifying LLMs from Their Responses

CMPSC 448 midterm project. We train neural networks to predict which LLM family (GPT, Claude or Gemini) wrote a response, and study whether the user's prompt helps. The CNN is complete and evaluated. The RNN (LSTM) is in progress and its results are not yet in this repository.

## Team

- Nency Patel (team leader)
- Shreyansh Agarwal
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

## LSTM model

A bidirectional LSTM in `rnn/lstm_classifier.py`:

- Text is lowercased. Tokens are words (including contractions), punctuation marks and newlines, and markdown markers (bold `**`, headers `#`, code fences, numbered-list markers such as `1.`) are kept as their own tokens, so formatting is visible to the model. Numbers map to a single `<num>` token. The vocabulary is built from the training split only (at most 30,000 tokens, minimum frequency 2). Responses are truncated to 512 tokens and prompts to 128; sequences are packed, so padding never enters the LSTM.
- 128-dimensional embedding trained from scratch, one-layer bidirectional LSTM with 128 hidden units per direction, attention pooling and max pooling over time (concatenated, 512 dimensions), then dropout 0.4, a 256-unit ReLU layer, dropout 0.4 and a linear layer with 3 outputs. For input + output, the prompt and response are joined as one sequence (`prompt <sep> response`) and read by the same LSTM.
- AdamW, learning rate 2e-3, weight decay 1e-4, batch size 32, unweighted cross-entropy, gradient clipping at 1.0, learning rate halved when validation macro-F1 stalls for one epoch, early stopping on validation macro-F1 (patience 4), at most 15 epochs. Each setting was run once, with seed 42.

## LSTM results

Test set (299 examples), single run (seed 42). Chance accuracy is 0.333.

| Setting | Accuracy | Macro-F1 |
|---|---|---|
| Output only (RQ1) | 0.799 | 0.790 |
| Input only (RQ2) | 0.348 | 0.345 |
| Input + output (RQ2) | 0.793 | 0.784 |

Findings:

- The response alone identifies the LLM about 80% of the time. Per-class F1 for the output-only LSTM is 0.95 for Claude, 0.72 for Gemini and 0.70 for GPT. Of the 60 test errors, 48 are GPT and Gemini confused with each other; 105 of the 114 Claude responses are classified correctly.
- The prompt alone is close to chance level and adding it to the response gives no gain (0.793 vs 0.799 accuracy, a difference of two test examples).
- The output-only model overfits after a few epochs: training macro-F1 reaches 0.98 while validation loss rises from 0.62 (epoch 3) to 1.18 (epoch 15). Validation macro-F1 still improves slowly, so early stopping keeps the last epoch.
- The LSTM results come from one seed, so they have no error bars; the 2.3-point standard error from the test set size still applies.

Figures: `rnn/results_lstm/lstm_output_confusion.png` and `rnn/results_lstm/lstm_output_curves.png` (confusion matrices and training curves for the other settings are in the same folder).

## CNN vs LSTM

The two models perform almost the same. For output only, the LSTM scores 0.799 accuracy and 0.790 macro-F1 against 0.803 ± 0.008 and 0.793 ± 0.009 for the CNN, and both mostly confuse GPT with Gemini. The LSTM is slightly better on Claude (0.95 vs 0.92 F1) and slightly worse on GPT and Gemini, but each difference is only a few test examples. Because the LSTM reads the whole response in order while the CNN only sees short n-gram windows, the similar results suggest that local cues such as word choice, phrasing and formatting carry most of the signal. The comparison is between the models as configured: the LSTM uses one seed, longer inputs (512 vs 256 response tokens) and lowercased text.

## How to run

CNN:

```
pip install -r cnn/requirements.txt
cd cnn
python cnn_classifier.py --train ../data/train.csv --val ../data/validation.csv --test ../data/test.csv --mode output
bash run_all.sh ../data     # every CNN experiment, about an hour on 2 CPU cores
python analyze.py           # results table and figures
```

Use `--mode input` or `--mode both` for the other RQ2 settings and `--ablate format`, `--ablate punct` or `--ablate length` for the RQ4 ablations.

LSTM (run from the top of the repo):

```
pip install -r rnn/requirements.txt
python rnn/lstm_classifier.py --data_dir data --out_dir rnn/results_lstm --mode all     # all three RQ2 settings, about 1.5 hours on 2 CPU cores
```

Use `--mode output`, `--mode input` or `--mode input_output` to run a single setting. Each run writes metrics, a confusion matrix and training curves to `rnn/results_lstm/`.

To rebuild the dataset, open `preprocessing/Midterm_Dataset_Preprocessing.ipynb` and run all cells (it downloads the Arena data from Hugging Face).

## Use of LLM tools

AI assistants were used for coding help, debugging and drafting. All code, results and text were checked and are understood by the team.
