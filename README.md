# Fine-Tuning & Benchmarking Small Open-Source LLMs for Low-Resource Nepali NLP

> **Final-Year Project - BE Computer Engineering**
> **Binod Raj Pant**
> Exploring whether parameter-efficient fine-tuning can make small open-source LLMs more capable in **Nepali Question Answering, Translation, and Summarization**.

[![Kaggle Showcase](https://img.shields.io/badge/Kaggle-Project%20Showcase-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code/binodrajpant13/final-proj-showcasellama-3-2-3b)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Adapters-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/iwasbinod)
[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Unsloth](https://img.shields.io/badge/Training-Unsloth-orange)](https://github.com/unslothai/unsloth)
[![PEFT](https://img.shields.io/badge/Fine--Tuning-QLoRA%20%2B%20PEFT-green)](https://github.com/huggingface/peft)

---

## What is this project?

This repository documents my experiments with **parameter-efficient fine-tuning (PEFT)** of open-source language models for **low-resource Nepali NLP**.

The project started with larger 7 - 8B-class models and progressively expanded across several model families. After experimenting with different architectures and model sizes, I am moving forward with the **Llama 3.2 3B family** as the main model family for my final-year project.

The main research question is:

> **Can a carefully prepared Nepali dataset + QLoRA fine-tuning make a small LLM substantially better at specific Nepali NLP tasks without requiring massive computational resources?**

The project currently evaluates three downstream tasks:

| Task | Direction / Setting | Evaluation |
|---|---|---|
| **Question Answering** | Context-based Nepali QA | EM, F1, Refusal Rate |
| **Translation** | Nepali → English | BLEU, chrF++, Exact Match |
| **Summarization** | Nepali → Nepali | ROUGE, chrF++, BERTScore, length & quality diagnostics |

---

# Current Final-Year Direction

## Llama 3.2 3B

For the final-year project, I am standardizing the main experiments around:

**`unsloth/Llama-3.2-3B-Instruct-bnb-4bit`**

This gives the project a practical balance between:

- model capability
- low VRAM requirements
- Kaggle/T4 feasibility
- QLoRA compatibility
- Nepali adaptation
- reproducible benchmarking

### Current evaluation set

**30 questions per task - 90 questions total**

```text

       FINAL EVALUATION SET

  Context-Based QA       30
  Nepali → English Translation 30
  Nepali → Nepali Summary   30

 TOTAL              90

```

### Model configuration

```python
CONFIG = {
  "qa": {
    "llama": {
      "base": "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
      "adapter": "iwasbinod/llama-3.2-3b_qa_v2",
    },
  },

  "translation": {
    "llama": {
      "base": "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
      "adapter": "iwasbinod/llama-3.2-3b-nepali-english-translation-with_syn_v4",
    },
  },

  "summarization": {
    "llama": {
      "base": "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
      "adapter": "iwasbinod/summarization-llama3.2-3b-real-syn-highquality",
    },
  },
}

FAMILY_LABEL = {
  "llama": "Llama-3.2-3B"
}

GEN_PARAMS = {
  "qa": {
    "max_new_tokens": 96,
    "temperature": 0.0
  },

  "translation": {
    "max_new_tokens": 220,
    "temperature": 0.15
  },

  "summarization": {
    "max_new_tokens": 280,
    "temperature": 0.25
  },
}
```

---

# Results at a Glance

## Question Answering

**Adapter:** `iwasbinod/llama-3.2-3b_qa_v2`

| Metric | Base | Fine-Tuned | Improvement |
|---|---:|---:|---:|
| **Exact Match (EM)** | 6.02 | **11.65** | **+5.64** |
| **F1** | 30.37 | **36.18** | **+5.81** |
| **Refusal Rate** | 1.1% | **0.0%** | Improved |

### QA training configuration

| Parameter | Value |
|---|---:|
| LoRA rank (`r`) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Learning rate | 3e-5 |
| Epochs | 2 |
| Max sequence length | 768 |
| Batch size/device | 1 |
| Gradient accumulation | 16 |
| Effective batch size | 16 |
| Warmup | 8% of total steps |
| Optimizer | `adamw_8bit` |
| Scheduler | `cosine` |
| Weight decay | 0.01 |
| Packing | False |

### QA data

**Primary dataset**

`iwasbinod/QA_nepali_5000_pairs_syn` - approximately 5,000 cleaned synthetic Nepali QA pairs, span-aligned against their own contexts.

**Additional supervision**

- SQuAD v1.1 → Nepali, machine-translated and fuzzy-aligned.
- Approximately 2,500 - 3,000 aligned examples when enabled.
- Hindi `ai4bharat/IndicQA` was deliberately excluded to keep the final result focused on Nepali.
- Hard negatives were disabled (`N_NEGATIVES = 0`) because earlier experiments showed better F1 on the all-answerable evaluation setting.

### Held-out evaluation

`Yunika/Nepali-QA` - 266 examples, kept completely separate from training.

The evaluation set is human-written and all-answerable, allowing refusal rate to be monitored without introducing unnecessary negative examples into training.

**Adapter:**
https://huggingface.co/iwasbinod/llama-3.2-3b_qa_v2

---

# Summarization

**Adapter:** `iwasbinod/summarization-llama3.2-3b-real-syn-highquality`

### REAL XLSum-Nepali evaluation

| Metric | Base | Fine-Tuned |
|---|---:|---:|
| ROUGE-1 | 0.1147 | **0.1683** |
| ROUGE-2 | 0.0247 | **0.0525** |
| ROUGE-L | 0.0904 | **0.1591** |
| chrF++ | 22.9222 | **23.4638** |
| BERTScore F1 | 0.6755 | **0.7363** |
| Average prediction length | 48.8833 | **14.7500** |
| Average reference length | 18.7833 | 18.7833 |
| Average length ratio | 2.6025 | **0.7853** |
| Empty rate | 0.0000 | 0.0000 |
| Devanagari rate | 1.0000 | 1.0000 |
| Likely truncated rate | 0.2833 | **0.0000** |
| Composite score / 100 | 31.4498 | **35.8687** |
| Source-copy Jaccard | 0.1255 | **0.0497** |

### What changed?

The fine-tuned model produced:

- higher ROUGE-1
- higher ROUGE-2
- higher ROUGE-L
- higher chrF++
- higher BERTScore
- much more controlled output length
- **0% likely truncation**
- lower source-copy overlap

This is particularly useful because the goal is not simply to generate longer Nepali text, but to produce **shorter, more focused summaries that better match the reference summaries**.

**Adapter:**
https://huggingface.co/iwasbinod/summarization-llama3.2-3b-real-syn-highquality

---

# Translation

**Adapter:** `iwasbinod/llama-3.2-3b-nepali-english-translation-with_syn_v4`

### Evaluation

| Metric | Base | Fine-Tuned | Improvement |
|---|---:|---:|---:|
| **BLEU** | 8.5613 | **15.7620** | **+7.2007** |
| **chrF++** | 35.4504 | **40.8992** | **+5.4489** |
| **Exact Match %** | 2.00% | **3.33%** | **+1.33 pp** |

### Data sources

**Main organic corpus**

`iamTangsang/Nepali-to-English-Translation-Dataset`

**Additional high-quality data**

FLORES-200 development set:

`facebook/flores - npi_Deva-eng_Latn`

The combination provides both domain-specific Nepali-English translation data and a smaller high-quality human-translated benchmark.

**Adapter:**
https://huggingface.co/iwasbinod/llama-3.2-3b-nepali-english-translation-with_syn_v4

---

# Model Journey

The project did **not** start with Llama 3.2 3B.

During the experimentation phase, I worked with multiple model families and sizes, including:

```text
         MODEL EXPLORATION

    ┼

  Llama 3.1    Mistral 7B/8B   Qwen 3B/8B

    ┼

        Kaggle Experiments

       Architecture Comparison

       Llama 3.2 3B selected

       FINAL-YEAR PROJECT
```

I also experimented with **Qwen 3B and 8B models in Kaggle notebooks**.

These experiments helped me understand the practical trade-offs between model size, VRAM usage, dataset quality, training stability, and downstream performance.

### Why Llama 3.2 3B?

For the final-year project, I am moving forward with the **Llama 3.2 3B family** rather than maintaining many model families in the final pipeline.

This gives the final evaluation a more consistent setup:

- more focused
- easier to reproduce
- more computationally practical
- easier to benchmark consistently
- suitable for constrained GPU environments such as Kaggle T4

> **Important:** The initial repository files primarily contain the earlier **Llama and Mistral experiments**. The Qwen experiments were also conducted in Kaggle. If you are interested in those experiments, I can provide/add the corresponding Kaggle links and artifacts separately.

---

# Kaggle Showcase

The main Llama 3.2 3B showcase notebook is available here:

**Final Project Showcase - Llama 3.2 3B**

https://www.kaggle.com/code/binodrajpant13/final-proj-showcasellama-3-2-3b

The notebook demonstrates the final-year evaluation pipeline, including:

```text
Base Model

   QA Base vs Fine-Tuned

   Translation Base vs Fine-Tuned

   Summarization Base vs Fine-Tuned

             Metrics + Analysis
```

---

# Hugging Face Adapter Registry

All currently published adapters are available through the `iwasbinod` Hugging Face profile.

## Current Llama 3.2 3B adapters

| Task | Base Model | Adapter |
|---|---|---|
| QA | `unsloth/Llama-3.2-3B-Instruct-bnb-4bit` | `iwasbinod/llama-3.2-3b_qa_v2` |
| Translation | `unsloth/Llama-3.2-3B-Instruct-bnb-4bit` | `iwasbinod/llama-3.2-3b-nepali-english-translation-with_syn_v4` |
| Summarization | `unsloth/Llama-3.2-3B-Instruct-bnb-4bit` | `iwasbinod/summarization-llama3.2-3b-real-syn-highquality` |

Profile:

https://huggingface.co/iwasbinod

---

# Earlier 7 - 8B Experiments

Before moving to the Llama 3.2 3B final-year setup, I conducted experiments primarily with **Llama and Mistral 7 - 8B-class models**.

These earlier adapters remain useful for comparison and for understanding the evolution of the project.

## Llama

| Task | Adapter |
|---|---|
| Text Summarization - Scraped News | `iwasbinod/nepali-llama-summarization-news-scraped-data-qlora` |
| Text Summarization - General Corpus | `iwasbinod/nepali-llama-summarization-qlora` |
| Machine Translation | `iwasbinod/nepali-llama-translation-qlora` |
| Question Answering | `iwasbinod/nepali-llama-qa-qlora` |

## Mistral

| Task | Adapter |
|---|---|
| Text Summarization - Scraped News | `iwasbinod/nepali-mistral-summarization-news-scraped-data-qlora` |
| Text Summarization - General Corpus | `iwasbinod/nepali-mistral-summarization-qlora` |
| Machine Translation | `iwasbinod/nepali-mistral-translation-qlora` |
| Question Answering | `iwasbinod/nepali-mistral-qa-qlora` |

These are mostly open/public resources and can be used to inspect the earlier experiments.

---

# Training Stack

The project uses a lightweight PEFT-based training pipeline:

```text
         Nepali Dataset

       Data Cleaning / QA

       Tokenization / Formatting

       4-bit Quantized Base LLM

           QLoRA

          PEFT

          Unsloth

         LoRA Adapter

       Benchmark Base vs FT
```

### Core technologies

- **Python**
- **PyTorch**
- **Hugging Face Transformers**
- **Hugging Face PEFT**
- **Unsloth**
- **QLoRA**
- **BitsAndBytes**
- **Kaggle GPU / NVIDIA T4**
- **Streamlit**
- **BLEU / chrF++**
- **ROUGE**
- **BERTScore**
- **Exact Match / F1**

---

# 🧪 Why Benchmark the Base Model?

A major part of this project is that **fine-tuning is not automatically assumed to improve every metric**.

Instead, the project follows:

```text
       BASE MODEL

       Evaluation

       Fine-Tuning   Same Test Set

       Fine-Tuned Model ◄

       Comparative Analysis
```

This allows the project to answer a more meaningful question:

> **Does PEFT actually improve task-specific Nepali performance compared with the original base model?**

That question eventually became the motivation for the research work associated with this project.

---

# Research Outcome

One of the most valuable outcomes of this project was the opportunity to develop a **research paper from the experiments**.

## The PEFT Paradox

### **“The PEFT Paradox: Why Fine-Tuning Does Not Guarantee Absolute Dominance Over Base LLMs in Low-Resource Languages”**

**Binod Raj Pant∗**
*Independent Researcher*

The research examines an observation that emerged during the low-resource language fine-tuning experiments:

> **A fine-tuned model can improve on some task-specific metrics while still failing to achieve absolute dominance over its base model across every evaluation dimension.**

The work grew directly from the experimental process documented in this repository.

### From project → experiments → research

```text
Final-Year Project

Dataset Construction

Multiple LLM Experiments

Llama / Mistral / Qwen

QLoRA + PEFT

Base vs Fine-Tuned Evaluation

Unexpected / Mixed Results

Research Question

    THE PEFT PARADOX
```

I am especially grateful to all the **teachers, supervisors, mentors, and everyone who supported the project**, whose guidance made it possible to take this work beyond a final-year implementation and into research.

**Thank you. **

---

# 🗂️ Repository Structure

```text
Fine_Tuning_and_Benchmarking_Small_Open-Source_LLMs/

 data/         # Dataset and processed data
 evaluation/      # Evaluation scripts and metric utilities
 logs/         # Training / execution logs
 outputs/        # Generated outputs and intermediate artifacts
 results/        # Benchmark results

 baseline_eval.py    # Base-model benchmarking
 main.py        # Main pipeline entry point
 qlora_trainer.py    # QLoRA fine-tuning pipeline
 scrap.py        # Data collection / scraping utilities
 streamlit_app.py    # Interactive inference interface

 pyproject.toml     # Project configuration
 requirements.txt    # Dependency list
 uv.lock        # Locked dependencies
 .python-version    # Python version
 .gitignore
 README.md
```

---

# Installation

The project uses `uv` for dependency management.

### 1. Clone

```bash
git clone https://github.com/Just-Binod/Fine_Tuning_and_Benchmarking_Small_Open-Source_LLMs.git

cd Fine_Tuning_and_Benchmarking_Small_Open-Source_LLMs
```

### 2. Create environment

```bash
uv venv

source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
uv sync
```

For GPU training, ensure that the CUDA, PyTorch, Unsloth, and BitsAndBytes versions are compatible with the available GPU environment.

---

# Running the Pipeline

### Data collection

```bash
python scrap.py
```

### Fine-tuning

```bash
python qlora_trainer.py
```

### Benchmarking

```bash
python baseline_eval.py
```

### Interactive application

```bash
streamlit run streamlit_app.py
```

---

# Reproducibility Notes

For meaningful comparison:

- Keep the evaluation set fixed.
- Evaluate the **base model and fine-tuned model using the same prompts**.
- Keep generation parameters task-specific but identical between base and fine-tuned runs.
- Do not train on the final evaluation examples.
- Report multiple metrics rather than relying on a single score.
- Record model version, adapter version, dataset version, and generation configuration.

---

# Project Philosophy

The project is not limited to improving a single benchmark score.

It is about understanding **what actually happens when small open-source models are adapted to a low-resource language**.

Nepali has fewer high-quality resources than many high-resource languages. Therefore, improvements cannot be judged only by model size or training loss.

The project focuses on:

```text
DATA QUALITY
   +
EFFICIENT FINE-TUNING
   +
CAREFUL EVALUATION
   +
BASELINE COMPARISON
   +
LOW-RESOURCE LANGUAGE

REPRODUCIBLE NLP RESEARCH
```

---

# Acknowledgements

I would like to sincerely thank my **teachers, project supervisors, mentors, colleagues, open-source contributors, dataset creators, and everyone who supported this work**.

This project would not have reached its current stage without the open-source ecosystem and the people who make research and experimentation accessible.

Special thanks to the communities and tools around:

- Hugging Face
- Unsloth
- PEFT
- QLoRA
- Kaggle
- PyTorch
- Open-source LLM research
- Nepali NLP datasets and researchers

---

# Author

### **Binod Raj Pant**

**BE Computer Engineering - Final-Year Project**

Research interest:

> **Low-Resource NLP · Nepali Language Models · PEFT · QLoRA · LLM Evaluation**

---

## If this project is useful

If you are researching **Nepali NLP, low-resource language modeling, PEFT, QLoRA, or small LLMs**, feel free to explore the experiments and adapters.

The repository is an evolving research project, and additional **Llama, Mistral, and Qwen Kaggle experiments** may be added as the work progresses.

**Thank you for visiting this project. **
