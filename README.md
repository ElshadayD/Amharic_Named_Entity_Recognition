# Amharic Named Entity Recognition (NER) Using XLM-RoBERTa


## Overview

This project explores Named Entity Recognition (NER) for Amharic using the multilingual transformer model **XLM-RoBERTa-base**.

The task involves identifying and classifying named entities in Amharic text using the BIO tagging scheme. The project also investigates the impact of class imbalance on NER performance by comparing standard cross-entropy loss with class-weighted cross-entropy loss.

## Dataset

- **Source:** EthioNLP
- **Language:** Amharic
- **Format:** Token-level annotations
- **Annotation scheme:** BIO

### Entity Labels

| Label | Description |
|---|---|
| PER | Person |
| ORG | Organization |
| LOC | Location |
| TIME | Time |
| TTL | Title |
| O | Outside any entity |

### Dataset Splits

| Split | Tokens |
|---|---:|
| Training | 87,961 |
| Validation | 11,062 |
| Test | 10,605 |

## Methodology

**Model:** [XLM-RoBERTa-base](https://huggingface.co/FacebookAI/xlm-roberta-base)

Since XLM-RoBERTa uses subword tokenization while NER annotations are word-level, labels were aligned as follows:

- The first subword receives the original word label.
- Remaining subwords receive `-100` and are ignored during loss calculation.

### Experiments

Two experiments were conducted using the same model and training configuration.

| Configuration | Experiment 1 | Experiment 2 |
|---|---|---|
| Loss function | Standard cross-entropy | Weighted cross-entropy |
| Class weights | No | Yes |
| Learning rate | 0.00002 | 0.00002 |
| Batch size | 16 | 16 |
| Epochs | 10 | 10 |
| Dropout | 0.3 | 0.3 |
| Frozen layers | First 3 | First 3 |

For the second experiment, inverse-frequency class weights were calculated to give greater importance to underrepresented entity labels.

## Results

### Overall Performance

| Metric | Without Class Weights | With Class Weights |
|---|---:|---:|
| Test Accuracy | 0.94 | 0.93 |
| Test Macro F1 | ~0.65 | **0.76** |
| Training F1 | 0.5248 | 0.677 |

Class weighting improved test macro F1 by approximately 11 percentage points, despite a slight decrease in accuracy.

### Key Findings

- Class weighting significantly improved recognition of minority entity classes.
- B-PER F1 increased from 0.73 to 0.93.
- I-PER F1 increased from 0.68 to 0.91.
- Entity boundary errors between B- and I-tags were reduced.
- Rare labels, particularly title-related labels, remained challenging.

## Conclusion

The experiments demonstrate that multilingual transformer models can be effectively applied to Amharic NER.

However, class imbalance has a significant impact on entity recognition performance. Introducing class-weighted loss improved macro F1 and recognition of minority classes, highlighting the importance of addressing data imbalance and using appropriate evaluation metrics.

The results also show why accuracy alone can be misleading for highly imbalanced sequence-labeling tasks.

## Technologies

- Python
- PyTorch
- Hugging Face Transformers
- XLM-RoBERTa
- Scikit-learn
- NumPy

## Future Work

- Improve recognition of rare entity types.
- Explore alternative class-weighting strategies.
- Investigate additional Amharic NER datasets.
- Experiment with different multilingual and language-specific transformer models.
