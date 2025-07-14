# Methodology

## Pre‑processing

- Apply low‑pass (< 500 Hz) or modulation filter (< 20 Hz) to raw waveform.

- Extract prosodic feature vectors (F0, energy, duration).

## Model Architectures

- Baseline – 1D CNN over concatenated feature streams.

- Seq2Vec – Bi‑GRU with attention pooling.

S- elf‑Supervised – Fine‑tune wav2vec 2.0 on prosody‑filtered audio.

## Training

- Cross‑entropy loss, AdamW, cosine LR schedule.

- Data augmentation: Gaussian noise, speed perturbation ±5 %, pitch shift ±1 semitone.

## Evaluation Metrics

- Accuracy, macro F1, Matthews correlation.

- Confusion analysis by language pair and utterance length.

