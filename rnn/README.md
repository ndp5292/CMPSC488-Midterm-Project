# LLM fingerprint classification: LSTM

Bidirectional LSTM that predicts `LLM_name` from `LLM_Input` and/or `LLM_output`.

## Data format
Uses the team's fixed splits from the preprocessing notebook: `data/train.csv`, `data/validation.csv`, `data/test.csv`,
with columns `LLM_name`, `LLM_Input`, `LLM_output`, `source_id`, `prompt_group`.

## Run on the team's fixed splits
Run from the top of the repo:
```
pip install -r rnn/requirements.txt
python rnn/lstm_classifier.py --data_dir data --out_dir rnn/results_lstm --mode all   # ~1 hour on 2 CPU cores, a few minutes on a GPU
```
`--mode all` runs the three RQ2 settings one after another and writes `lstm_rq2_comparison.csv`.

## Run a single setting
```
# RQ1: output only
python rnn/lstm_classifier.py --data_dir data --out_dir rnn/results_lstm --mode output
# RQ2: input only / input + output
python rnn/lstm_classifier.py --data_dir data --out_dir rnn/results_lstm --mode input
python rnn/lstm_classifier.py --data_dir data --out_dir rnn/results_lstm --mode input_output
```
Results are written to the `--out_dir` folder (`rnn/results_lstm/`), one set per mode:
- `lstm_<mode>_metrics.json`: accuracy, macro-F1, per-class report, confusion matrix, per-epoch history, hyperparameters
- `lstm_<mode>_confusion.png`: test-set confusion matrix
- `lstm_<mode>_curves.png`: train and validation loss and macro-F1 per epoch
- `lstm_<mode>_best.pt`: best checkpoint (not committed; recreated by running the script)
- `lstm_rq2_comparison.csv`: summary table (with `--mode all`)
- `training_log.txt`: console output of the reported run

## Results (test set, 299 responses, seed 42)
| Setting | Accuracy | Macro-F1 |
|---|---|---|
| Output only (RQ1) | 0.799 | 0.790 |
| Input only (RQ2) | 0.348 | 0.345 |
| Input + output (RQ2) | 0.793 | 0.784 |

Chance accuracy is 0.333. These numbers come from a single run (seed 42), unlike the CNN, which reports mean and std over 3 seeds.

## Details
- Tokens: lowercased; words (including contractions), punctuation marks, newlines (`<nl>`), bold markers (`**`), header marks (`#`), code fences and numbered-list markers (`1.` becomes `<num_item>`) are separate tokens, because formatting is a signal. Numbers become `<num>`.
- Vocabulary built from the training split only (max 30000, min frequency 2).
- Lengths: responses truncated to 512 tokens, prompts to 128 tokens. Sequences are packed, so padding never enters the LSTM.
- Model: embedding 128, one-layer bidirectional LSTM with hidden size 128 per direction, attention pooling and max pooling over time (concatenated, 512 dimensions), then dropout 0.4, linear 512 to 256, ReLU, dropout 0.4, linear layer with 3 logits. `input_output` mode feeds `prompt <sep> response` as one sequence through a single LSTM.
- Training: AdamW lr 2e-3, weight decay 1e-4, batch 32, unweighted cross-entropy, gradient clipping at 1.0, learning rate halved when validation macro-F1 stalls for 1 epoch, at most 15 epochs, early stopping on validation macro-F1 (patience 4). The best epoch is evaluated on the test set once.
- Device: uses CUDA or Apple MPS if available, otherwise CPU.
