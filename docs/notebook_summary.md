# Notebook Quick Summary

## Purpose
Reference guide for the notebook [rnn_lstm_transformer_exploration.ipynb](../rnn_lstm_transformer_exploration.ipynb).

Goal: compare sequence models (`RNN`, `LSTM`, `Transformer Encoder`) on a synthetic task, and learn structure + optimization basics.

## What the notebook does
1. Creates a dummy sequence dataset with long-range dependency:
   - label `1` if first token equals last token
   - label `0` otherwise
   - balanced classes (50/50)
2. Builds 3 models:
   - `RNNClassifier`
   - `LSTMClassifier`
   - `TransformerClassifier` (with sinusoidal positional encoding)
3. Trains each model using shared utilities:
   - `CrossEntropyLoss`
   - `AdamW`
   - gradient clipping
4. Compares results on:
   - best validation accuracy
   - test accuracy
   - parameter count
   - training time
5. Runs a small hyperparameter sweep:
   - learning rate: `5e-4`, `1e-3`, `2e-3`
   - hidden size: `64`, `96`

## Expected learning outcomes
- Understand how recurrent and attention-based architectures differ.
- See why `LSTM` can handle long dependencies better than vanilla `RNN`.
- Understand how `Transformer` trades compute for stronger sequence interaction.
- Practice basic optimization and model comparison.

## Setup used
- Dependencies are listed in [requirements.txt](../requirements.txt).
- Local environment: `.venv`
- Registered kernel: **Python (.venv nlp_cores)**

## Suggested workflow
1. Open notebook and select kernel **Python (.venv nlp_cores)**.
2. Run all cells top-to-bottom once.
3. Inspect summary table and validation curves.
4. Change one variable at a time (sequence length, hidden size, learning rate).

## Next experiments
- Increase sequence length to stress memory (e.g., 80, 120).
- Add noisy rules or multiple dependency patterns.
- Try scheduler + early stopping.
- Add confusion matrix and per-class metrics.
