# Multilabel Emotion Classification

This project focuses on building a multilingual, multilabel emotion classifier using Transformer-based models like BERT. The goal is to predict multiple emotions (e.g., joy, fear, sadness) from a given piece of text. The project supports English and non-English languages (e.g., Hindi).

---

## 🧠 Problem Statement
Given a piece of text, the model should predict one or more emotion labels from the following list:

**English Labels**:
```
[anger, fear, joy, sadness, surprise]
```

**Non-English Labels** (e.g., Hindi):
```
[anger, fear, joy, sadness, surprise, disgust]
```

### 🔢 Output Format
For each input, output a binary vector (e.g., [0,1,0,1,1]) indicating the presence of each emotion.

### 📊 Evaluation Metrics
- F1-macro
- Precision-macro
- Recall-macro

---

## 📁 Project Structure
```
multilabel-emotion-classification/
├── README.md
├── requirements.txt
├── config.yaml
├── train.py
├── inference.py
├── utils.py
├── interpretability/
│   ├── lime_explainer.py
│   └── attention_viz.ipynb
├── data/
│   ├── README.md
│   └── sample.json
└── notebooks/
    ├── EDA.ipynb
    └── baseline_model.ipynb
```

---

## 🧠 How to Use

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python train.py --config config.yaml
```

### 3. Run inference
```bash
python inference.py --text "I can't move, my hand is stuck, and my mom is screaming"
```

### 4. Run EDA
Open `notebooks/EDA.ipynb` in Jupyter/Colab.

---

## 📈 Future Work
- Add BERTViz and LIME-based interpretability
- Try different loss functions
- Add knowledge-based contextual support using COMET

---

## 📚 References
- [HuggingFace Transformers](https://huggingface.co/transformers)
- [LIME](https://github.com/marcotcr/lime)
- [COMET](https://github.com/atcbosselut/comet-commonsense)
