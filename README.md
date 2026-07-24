# 📈 Financial Tweet Sentiment Analysis & Agentic AI Workflow

An end-to-end Natural Language Processing (NLP) framework and Agentic AI workflow for classifying financial market sentiment on Twitter/X into **Bearish (0)**, **Bullish (1)**, or **Neutral (2)** categories. 

---

## 📸 Overview & Architecture

Rather than relying on a single static model or plain LLM prompt, this repository combines:
1. **Traditional ML Baselines**: TF-IDF, Word2Vec, Logistic Regression, LinearSVM, and MLP architectures.
2. **Transformer Encoders**: Fine-tuned **FinBERT** (`ProsusAI/finbert`), Sentence-BERT, and RoBERTa models.
3. **Agentic AI Decision Protocol**: A conversational agent that orchestrates classification models as tools, automatically routing queries based on intent, checking classifier agreement, and explaining predictions in plain financial language.

## 📊 Dataset & Exploratory Findings

* **Training Set**: 9,543 labelled tweets.
* **Test Set**: 2,388 unlabelled tweets (`pred_10.csv`).
* **Class Balance**:
  * 🔴 **Bearish (0)**: 15.11% (1,442 samples)
  * 🟢 **Bullish (1)**: 20.15% (1,923 samples)
  * ⚪ **Neutral (2)**: 64.74% (6,178 samples)

> **Key Insight**: Because the dataset is heavily skewed toward Neutral news and announcements, models are evaluated primarily using **Macro-averaged F1** to ensure balanced performance across minority classes.

---

## 🏆 Model Performance Benchmark

Models evaluated on the held-out validation set ($n = 1,909$, 80/20 stratified split):

| Rank | Model Architecture | Feature Representation | Accuracy | Precision | Recall | **Macro F1** |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| 🥇 **1** | **Fine-tuned FinBERT (Full)** | **Fine-tuned Transformer** | **0.862** | **0.805** | **0.844** | **0.822** |
| 🥈 **2** | Sentence-BERT + LogReg | Frozen Encoder Embeddings | 0.811 | 0.754 | 0.763 | **0.758** |
| 🥉 **3** | RoBERTa + LogReg | Frozen Encoder Embeddings | 0.817 | 0.760 | 0.756 | **0.757** |
| 4 | FinBERT Embeddings + LogReg | Frozen Domain Encoder | 0.808 | 0.747 | 0.736 | **0.741** |
| 5 | TF-IDF + Logistic Regression | Sparse Lexical (Balanced) | 0.799 | 0.724 | 0.763 | **0.741** |
| 6 | TF-IDF + LinearSVM | Sparse Lexical (Balanced) | 0.803 | 0.731 | 0.738 | **0.734** |
| 7 | Word2Vec + MLP (Deep) | Dense Static Embeddings | 0.782 | 0.705 | 0.680 | **0.690** |
| 8 | FinBERT (Head-only fine-tune) | Frozen Encoder + Head | 0.743 | 0.674 | 0.694 | **0.681** |
| 9 | Word2Vec + LogReg | Dense Static Embeddings | 0.747 | 0.662 | 0.583 | **0.609** |
| 10 | DeepSeek-V4 (Zero-Shot) | Decoder LLM Prompting | 0.587 | 0.653 | 0.686 | **0.602** |

---

## 🤖 Agentic AI Workflow (`agent_10.ipynb`)

The repository includes a stateful conversational agent capable of tool calling and automated decision-making:

### Registered Tools
1. `classify_fast(tweet)`: Evaluates tweets instantly (<1 ms) using class-balanced TF-IDF + Logistic Regression.
2. `classify_best(tweet)`: Lazy-loads fine-tuned **FinBERT** on GPU/MPS accelerator to output high-accuracy classifications with per-class confidence probabilities.
3. `compare_models(tweet)`: Executes both models, cross-checks predictions, and generates human-readable reconciliation.
4. `get_model_info()`: Exposes model evaluation metrics and speed trade-offs for user introspection.

### Decision Protocol
* **Fast Path**: Exploratory requests default to `classify_fast`.
* **High Precision**: High-stakes queries route to `classify_best`.
* **Auto-Escalation**: Ambiguous signals or model disagreements trigger `compare_models` automatically.

---

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
2. Environment VariablesCreate a .env file in the root directory if you wish to run the Agentic AI Workflow (agent_10.ipynb):
```plaintext
DEEPSEEK_API_KEY=your_deepseek_api_key_here
OPENAI_API_BASE=[https://api.deepseek.com/v1](https://api.deepseek.com/v1)
OPENAI_MODEL=deepseek-chat
```
# Start interactive CLI session
chat()
👥 Authors (Group 10 - NOVA IMS)  Alexander Batista - [20250419]  Mehmet Karaca - [20250344]  Luís Mendes - [20221949]  Verónica Mendes - [20221945] 
