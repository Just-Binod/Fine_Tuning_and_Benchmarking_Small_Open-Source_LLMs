

```markdown
# Fine-Tuning and Benchmarking Small Open-Source LLMs for Low-Resource Nepali NLP Tasks

This repository contains the source code, dataset configurations, and evaluation pipelines used to fine-tune and benchmark 7-8B parameter open-source large language models (LLMs) for the Nepali language. 

Using **Unsloth**, **QLoRA**, and **PEFT**, Llama and Mistral architectures were optimized across multiple downstream tasks, including text summarization, machine translation, and question answering (QA).

---

## Model Registry (Hugging Face Adapters)

All fine-tuned LoRA adapters resulting from these experiments are publicly hosted on Hugging Face under the profile `iwasbinod`.

### Llama-Based Adapters
* `iwasbinod/nepali-llama-summarization-news-scraped-data-qlora` - News summarization (scraped datasets)
* `iwasbinod/nepali-llama-summarization-qlora` - General text summarization
* `iwasbinod/nepali-llama-translation-qlora` - English-to-Nepali / Nepali-to-English translation
* `iwasbinod/nepali-llama-qa-qlora` - Question answering and context extraction

### Mistral-Based Adapters
* `iwasbinod/nepali-mistral-summarization-news-scraped-data-qlora` - News summarization (scraped datasets)
* `iwasbinod/nepali-mistral-summarization-qlora` - General text summarization
* `iwasbinod/nepali-mistral-translation-qlora` - English-to-Nepali / Nepali-to-English translation
* `iwasbinod/nepali-mistral-qa-qlora` - Question answering and context extraction

---

## Setup and Installation

This project utilizes `uv` for lightning-fast environment setup and deterministic dependency tracking via `uv.lock` and `pyproject.toml`.

### 1. Clone the Repository
```bash
git clone [https://github.com/Just-Binod/Fine_Tuning_and_Benchmarking_Small_Open-Source_LLMs.git](https://github.com/Just-Binod/Fine_Tuning_and_Benchmarking_Small_Open-Source_LLMs.git)
cd Fine_Tuning_and_Benchmarking_Small_Open-Source_LLMs

```

### 2. Environment Setup

Create a virtual environment matching the repository's `.python-version` declaration and activate it:

```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

```

### 3. Install Dependencies

Sync the exact lockfile state to ensure reproducible environments:

```bash
uv sync

```

*Note: Ensure your local CUDA configuration aligns with Unsloth's execution kernels prior to training.*

---

## How to Reproduce

### 1. Data Processing & Scraping

Raw data handling or scraping operations are initiated via:

```bash
python scrap.py

```

### 2. Fine-Tuning (QLoRA)

To start the training run using Unsloth and PEFT adapters, run the trainer script:

```bash
python qlora_trainer.py

```

*Parameters can be adjusted inside `qlora_trainer.py` to toggle between Mistral and Llama architectures.*

### 3. Evaluation & Benchmarking

To benchmark the base models or your fine-tuned checkpoints against performance baselines:

```bash
python baseline_eval.py

```

### 4. Interactive UI

A Streamlit interface is provided to interact with the models or view results dynamically:

```bash
streamlit run streamlit_app.py

```

---

## Repository Structure

As shown in `image_bf65a2.jpg`, the repository layout is structured as follows:

```text
├── data/               # Improved training datasets (e.g., textbook + health pairs)
├── evaluation/         # Evaluator scripts, formatters, and metrics calculation
├── logs/               # Streamlit application logs and run metadata
├── outputs/            # Generated outputs and cached files
├── results/            # Performance metrics for fine-tuned checkpoints
├── .gitignore          # Git exclusion rules
├── .python-version     # Target Python version file
├── README.md           # Project overview and documentation
├── baseline_eval.py    # Baseline evaluation execution pipeline
├── main.py             # Principal project execution entry point
├── pyproject.toml      # Project metadata and tool configuration declarations
├── qlora_trainer.py    # Unsloth-accelerated QLoRA training loop script
├── requirements.txt    # Fallback legacy pip dependency manifest
├── scrap.py            # Data scraping and preprocessing utility
├── streamlit_app.py    # Interactive Streamlit application
└── uv.lock             # Deterministic package manager lockfile

```

```

```
