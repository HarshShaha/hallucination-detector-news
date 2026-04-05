# 🧠 Hallucination Detection System (News Domain)

## 🚀 Overview

This project presents a **Context-Aware Hallucination Detection System** designed to identify factual inconsistencies in LLM-generated text.

Unlike generic approaches, this system uses a **news-domain knowledge base** built from the CNN/DailyMail dataset, making it more robust for real-world scenarios.

The system combines:

* 🔎 Retrieval (FAISS + semantic embeddings)
* 🧠 Natural Language Inference (NLI)
* ⚡ GPU-accelerated batch processing

---

## 🎯 Problem Statement

Large Language Models (LLMs) often generate fluent but **factually incorrect (hallucinated)** content.
This project aims to:

> Detect whether a given text contains hallucinated claims by verifying it against trusted knowledge sources.

---

## 🏗️ System Architecture

```
LLM Output
    ↓
Claim Extraction
    ↓
Evidence Retrieval (FAISS + Sentence Transformers)
    ↓
NLI Verification (Transformer Model)
    ↓
Hallucination Classification
```

---

## 🧠 Key Features

* ✅ Domain-specific (News-based verification)
* ✅ Hybrid retrieval using semantic embeddings
* ✅ Transformer-based fact verification (NLI)
* ✅ GPU optimization for fast inference
* ✅ Separation of training and inference pipelines
* ✅ Flask-based web interface
* ✅ Modular and scalable design

---

## 📊 Dataset

* Primary Dataset:

  * CNN/DailyMail

The dataset is used to:

* Build a **knowledge base of factual sentences**
* Train a retrieval system for grounding claims

---

## ⚙️ Tech Stack

* **Language:** Python
* **Frameworks:** PyTorch, HuggingFace Transformers
* **Embeddings:** SentenceTransformers
* **Retrieval:** FAISS
* **Backend:** Flask
* **NLP Tools:** NLTK

---

## 📁 Project Structure

```
hallucination_detector/
│
├── training/
│   └── build_kb_and_index.py
│
├── inference/
│   ├── verifier_pipeline.py
│   └── app.py
│
├── saved/
│   ├── faiss.index
│   └── documents.pkl
│
├── templates/
│   └── index.html
│
├── requirements.txt
└── README.md
```

---

## 🚀 Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/hallucination-detector-news.git
cd hallucination-detector-news
```

---

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3️⃣ Build Knowledge Base (One-Time Training)

```bash
python training/build_kb_and_index.py
```

This will:

* Load news dataset
* Create sentence-level knowledge base
* Build FAISS index
* Save artifacts in `/saved/`

---

### 4️⃣ Run the Application

```bash
cd inference
python app.py
```

Open in browser:

```
http://127.0.0.1:5000
```

---

## 🌐 Web Interface

* Enter any text (e.g., LLM output)
* System detects:

  * ✅ FACTUAL
  * ❌ HALLUCINATED
  * ⚠️ UNCERTAIN

---

## 📸 Demo

*(Add screenshots here)*

```
Example:
"The Eiffel Tower is in Berlin"
→ HALLUCINATED ❌
```

---

## ⚡ Performance Optimizations

* GPU-based inference using PyTorch
* Batch processing for NLI
* Precomputed FAISS index
* Efficient sentence embeddings

---

## 🧪 Future Work

* 🔥 Fine-tune NLI model on FEVER
* 🌐 Real-time news API integration
* 📊 Evaluation metrics (F1, ROC curves)
* 🧠 Multi-hop reasoning
* 🎨 Advanced UI (React/Angular)

---

## 📌 Applications

* Fact-checking systems
* AI content validation
* News verification platforms
* Responsible AI systems

---

## 👨‍💻 Author

**Harsh Shaha**

* Final Year Computer Engineering Student
* Interested in NLP, Deep Learning, and High-Performance Computing

---

## ⭐ Contribute

Feel free to fork, improve, and submit pull requests!

---

## 📜 License

This project is for academic and research purposes.
