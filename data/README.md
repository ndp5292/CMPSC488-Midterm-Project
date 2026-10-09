# Dataset Information

## Project
Identifying LLMs from Their Responses Using CNN and RNN

## Dataset Source
We used the Chatbot Arena Human Preference 55K dataset from Hugging Face.

Source: https://huggingface.co/datasets/lmarena-ai/arena-human-preference-55k

License: Apache 2.0

## LLMs Selected
We selected three LLM families:
- GPT
- Claude
- Gemini

## Dataset Preparation
We cleaned the dataset by removing missing responses and duplicate examples. We then selected 1,000 responses from each LLM family to create a smaller, balanced dataset.

The final dataset contains 3,000 responses.

## Dataset Splitting
We divided the dataset into training, validation, and testing sets, ensuring that prompts do not overlap between the splits.

- Training: 2,400 responses
- Validation: 301 responses
- Testing: 299 responses

## Files
- `llm_dataset.csv`: Complete balanced dataset
- `train.csv`: Training data
- `validation.csv`: Validation data
- `test.csv`: Testing data

## Preprocessing Code
The preprocessing code is available in `preprocessing/Midterm_Dataset_Preprocessing.ipynb`.

This dataset will be used to train and evaluate CNN and RNN models for identifying which LLM generated a response.
