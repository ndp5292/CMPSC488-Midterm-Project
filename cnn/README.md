# LLM fingerprint classification: CNN

TextCNN (Kim 2014) that predicts `LLM_name` from `LLM_input` and/or `LLM_output`.

## Data format
One file (csv, json, jsonl or parquet) with one row per (LLM_name, LLM_Input, LLM_output).
Column names are auto-detected (`LLM_name`, `LLM_Input`, `LLM_output` and common aliases);
override with `--name_col --input_col --output_col`.

`synthetic_smoke_test.csv` is a fake stand-in used only to check that the code runs. Do not report its numbers.

## Run on the team's fixed splits
```
pip install -r requirements.txt
bash run_all.sh data     # all experiments, ~1 hour on 2 CPU cores
python analyze.py        # results table + figures
```
`data/` holds train.csv, validation.csv, test.csv, llm_dataset.csv from the preprocessing notebook.

## Run on any single file
```
pip install -r requirements.txt
# RQ1: output only
python cnn_classifier.py --data DATA.csv --mode output
# RQ2: input only / output only / input + output
python cnn_classifier.py --data DATA.csv --mode input
python cnn_classifier.py --data DATA.csv --mode both
# RQ3 (needs a task column): train on other tasks, test on one
python cnn_classifier.py --data DATA.csv --task_col task --test_task writing
# RQ4 ablations: remove formatting, punctuation, or truncate to 50 tokens
python cnn_classifier.py --data DATA.csv --ablate format
python cnn_classifier.py --data DATA.csv --ablate punct
python cnn_classifier.py --data DATA.csv --ablate length
```
Results (accuracy, macro-F1, per-class report, confusion matrix, training curves, mean and std over seeds 0,1,2) are written to `results/*.json`.

## Details
- Tokens: lowercase option, words, punctuation marks and newlines are separate tokens (formatting is a signal).
- Vocabulary built from the training split only (max 30000, min frequency 2).
- Model: embedding 128, conv1d widths 2,3,4,5 with 100 filters each, ReLU, max-over-time, dropout 0.5, linear layer. `both` mode uses one encoder for the input and one for the output, concatenated.
- Training: Adam lr 1e-3, weight decay 1e-4, batch 64, class-weighted cross-entropy, early stopping on validation macro-F1 (patience 4).
- Split: 80/10/10, stratified by LLM and grouped by prompt, so the same prompt never appears in both train and test. Turn off with `--no_group_split`.
