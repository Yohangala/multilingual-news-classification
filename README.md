# 🌐 Multilingual News Headline Classification

> **Published Research** · International Conference on Smart Systems for Sustainable Development · April 2026

[![Conference](https://img.shields.io/badge/Conference-Smart%20Systems%202026-blue?style=flat-square)]()
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-orange?style=flat-square&logo=huggingface)](https://huggingface.co)
[![Python](https://img.shields.io/badge/Python-3.9+-green?style=flat-square&logo=python)](https://python.org)
[![Languages](https://img.shields.io/badge/Languages-English%20|%20Hindi%20|%20Gujarati%20|%20Marathi-purple?style=flat-square)]()

---

## 🎯 Problem Statement

Classifying Indian news headlines across **4 languages** — English, Hindi, Gujarati, and Marathi — into categories such as Elections, Sports, Entertainment, Technology, Business, Health, Education, and more. The challenge: each language has different morphology, script, and linguistic patterns, requiring language-aware models and a data-centric pipeline.

---

## 📊 Dataset

| Language | Script | Train | Validation | Test | After Augmentation |
|---|---|---|---|---|---|
| Hindi | Devanagari | 30,851 | 3,835 | 3,836 | 47,472 |
| English | Latin | 36,877 | 4,610 | 4,610 | 35,956 |
| Gujarati | Gujarati | 26,472 | 3,416 | 3,346 | 35,859 |
| Marathi | Devanagari | 22,014 | 2,750 | 2,761 | 30,301 |

**Categories:** Elections, Sports, Entertainment, Technology, Business, Health, Education, Lifestyle, World, Science, Auto, India/National

Sample data is in `data/samples/`. Full datasets available on request.

---

## 🏗️ Pipeline

```
Raw News Headlines (4 languages)
        │
        ▼
Data Cleaning & Deduplication
        │
        ▼
Data Augmentation (back-translation, paraphrase)
        │
        ▼
Language-Specific Fine-tuning
┌─────────────────────────────────┐
│  English      → DistilBERT      │
│  English      → DistilRoBERTa   │
│  All langs    → MuRIL           │
│  Hindi/Gu/Mr  → IndicBERT       │
│  Baseline     → TF-IDF + LogReg │
└─────────────────────────────────┘
        │
        ▼
Probability Extraction (soft outputs)
        │
        ▼
Meta-Learner: Stacking Ensemble (Logistic Regression on probabilities)
        │
        ▼
Final Predictions
```

---

## 📈 Results

### English Baselines
| Model | Test Accuracy | Macro F1 |
|---|---|---|
| Logistic Regression (TF-IDF) | 70.5% | 0.650 |
| Linear SVC (TF-IDF) | 69.2% | 0.641 |

### Transformer Models (per language, fine-tuned)
| Model | Language | Notes |
|---|---|---|
| MuRIL | All 4 | Google multilingual Indian language model |
| DistilBERT | English | Distilled BERT, efficient inference |
| DistilRoBERTa | English | Distilled RoBERTa |
| IndicBERT | Hindi, Gujarati, Marathi | AI4Bharat Indian language model |

### Final Ensemble
| Component | Role |
|---|---|
| MuRIL probabilities | Multilingual signal |
| RoBERTa probabilities | English signal |
| TF-IDF + LogReg probabilities | Lexical signal |
| Meta Logistic Regression | Combines all signals |

Full classification reports are in `results/`.

---

## 🔑 Key Techniques

- **Data-centric approach**: Heavy focus on data quality, deduplication, and augmentation before modelling
- **Cross-lingual transfer**: MuRIL trained jointly across all 4 languages
- **Stacking ensemble**: Three diverse models → meta-learner for final prediction
- **Language-aware models**: IndicBERT for Indic scripts, DistilBERT/RoBERTa for English

---

## 📂 Repository Structure

```
multilingual-news-classification/
├── data/
│   └── samples/          # 200-row sample CSVs per language
├── results/
│   ├── classif_LogReg_test.txt
│   ├── classif_LinearSVC_test.txt
│   ├── train_history.csv
│   ├── ensemble_preds_test.csv
│   ├── dataset_summary_from_scan.csv
│   └── augmentation_summary_from_scan.csv
├── notebooks/            # Training notebooks (add your Colab .ipynb files here)
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

```bash
pip install -r requirements.txt
```

To fine-tune MuRIL on your own data:
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

model_name = "google/muril-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=12)
```

---

## 📚 Citation

```bibtex
@inproceedings{gala2026multilingual,
  title     = {Data-Centric Multilingual News Headline Classification with
               Transformer Models and Hybrid Ensemble Learning},
  author    = {Gala, Yohan and others},
  booktitle = {International Conference on Smart Systems for Sustainable Development},
  year      = {2026}
}
```

---

## 👥 Authors

Yohan Gala · K J Somaiya Institute of Technology, Mumbai
