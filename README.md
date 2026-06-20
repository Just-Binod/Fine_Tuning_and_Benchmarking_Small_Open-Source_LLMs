
```markdown
# Fine-Tuning and Benchmarking Small Open-Source LLMs for Low-Resource Nepali NLP Tasks

This repository provides an end-to-end framework for data ingestion, parameter-efficient fine-tuning (PEFT), and baseline benchmarking of 7-8B parameter large language models optimized for the Nepali language. 

By leveraging **Unsloth**, **QLoRA**, and **PEFT**, this project minimizes VRAM overhead while introducing domain-specific adaptations for Mistral and Llama architectures across critical NLP downstreams: Text Summarization, Machine Translation, and Question Answering (QA).

---

## Model Matrix & Adapter Registry

All fine-tuned LoRA adapters are serialized and hosted on the Hugging Face Hub under the `iwasbinod` organization profile.

| Architecture | Target Downstream Task | Dataset Domain / Context | Hugging Face Hub Repository ID |
| :--- | :--- | :--- | :--- |
| **Llama (7B/8B)** | Text Summarization | Scraped News Datasets | [`iwasbinod/nepali-llama-summarization-news-scraped-data-qlora`](https://huggingface.co/iwasbinod/nepali-llama-summarization-news-scraped-data-qlora) |
| **Llama (7B/8B)** | Text Summarization | General Text Corpus | [`iwasbinod/nepali-llama-summarization-qlora`](https://huggingface.co/iwasbinod/nepali-llama-summarization-qlora) |
| **Llama (7B/8B)** | Machine Translation | English $\leftrightarrow$ Nepali | [`iwasbinod/nepali-llama-translation-qlora`](https://huggingface.co/iwasbinod/nepali-llama-translation-qlora) |
| **Llama (7B/8B)** | Question Answering | Context-Grounded QA Pairs | [`iwasbinod/nepali-llama-qa-qlora`](https://huggingface.co/iwasbinod/nepali-llama-qa-qlora) |
| **Mistral (7B)** | Text Summarization | Scraped News Datasets | [`iwasbinod/nepali-mistral-summarization-news-scraped-data-qlora`](https://huggingface.co/iwasbinod/nepali-mistral-summarization-news-scraped-data-qlora) |
| **Mistral (7B)** | Text Summarization | General Text Corpus | [`iwasbinod/nepali-mistral-summarization-qlora`](https://huggingface.co/iwasbinod/nepali-mistral-summarization-qlora) |
| **Mistral (7B)** | Machine Translation | English $\leftrightarrow$ Nepali | [`iwasbinod/nepali-mistral-translation-qlora`](https://huggingface.co/iwasbinod/nepali-mistral-translation-qlora) |
| **Mistral (7B)** | Question Answering | Context-Grounded QA Pairs | [`iwasbinod/nepali-mistral-qa-qlora`](https://huggingface.co/iwasbinod/nepali-mistral-qa-qlora) |

---

## Installation & Environment Configuration

This project enforces deterministic package management via `uv`. Ensure your host environment has `uv` installed before executing setup steps.

### 1. Repository Initialization
```bash
git clone [https://github.com/Just-Binod/Fine_Tuning_and_Benchmarking_Small_Open-Source_LLMs.git](https://github.com/Just-Binod/Fine_Tuning_and_Benchmarking_Small_Open-Source_LLMs.git)
cd Fine_Tuning_and_Benchmarking_Small_Open-Source_LLMs

```

### 2. Hermetic Virtual Environment Setup

Initialize a localized python environment mapped to the native `.python-version` declaration:

```bash
uv venv
source .venv/bin/activate  # OS X / Linux
# On Windows use: .venv\Scripts\activate

```

### 3. Dependency Synchronization

Instantiate exact project states using the lockfile:

```bash
uv sync

```

*Note: For GPU acceleration configurations, verify that your CUDA Toolkit version aligns closely with your Unsloth wheel specifications before starting the training runs.*

---

## Execution & Pipeline Workflow

### 1. Data Scraping & Preprocessing

To invoke pipeline data ingestion routines or collect upstream corpus components:

```bash
python scrap.py

```

### 2. Model Fine-Tuning Pipeline

Fine-tuning execution relies on Unsloth optimization kernels. Execute the QLoRA training loop with:

```bash
python qlora_trainer.py

```

### 3. Quantitative Evaluation & Benchmarking

To run model checkpoints or base representations through validation suites (e.g., computing evaluation metrics across test splits):

```bash
python baseline_eval.py

```

### 4. Interactive User Interface

Launch the localized Streamlit web application dashboard to inspect outputs or query individual adapters in real time:

```bash
streamlit run streamlit_app.py

```

---

## Directory Architecture

The system file hierarchy follows the explicit blueprint captured in `image_bf65a2.jpg`:

```text
├── data/               # Localized data stores (Textbook parsing & Healthcare text pairs)
├── evaluation/         # Performance evaluation routines, formatters, and metrics logic
├── logs/               # Telemetry and application execution logs
├── outputs/            # Mid-process serialization caches and artifacts
├── results/            # Checkpoint performance metrics matrices
├── .gitignore          # Version control file exclusion patterns
├── .python-version     # Strictly enforced python runtime version specification
├── README.md           # Core documentation file
├── baseline_eval.py    # Main script for benchmarking baseline performance
├── main.py             # System execution orchestration pipeline entrypoint
├── pyproject.toml      # Modern tool configurations and metadata properties
├── qlora_trainer.py    # Unsloth-accelerated QLoRA training configuration
├── requirements.txt    # Legacy requirements specification manifest
├── scrap.py            # Targeted upstream scraping and collection utility
├── streamlit_app.py    # Streamlit presentation and interaction interface layer
└── uv.lock             # Deterministic resolution lockfile for project dependencies

```

---


```

```
