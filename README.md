# Quant-Chem Lecture Processing & Transcription Pipeline

A robust, multi-provider free LLM router and automated vision pipeline designed to convert quantum chemistry PDF lecture slides into clean Markdown containing precise LaTeX equations and structured diagram descriptions.

---

## Features

- **Multi-Provider LLM Free Tier Router**: Automatically discovers, filters (omitting broken/404 or preview models), ranks, and queries free endpoints across **Google Gemini**, **Groq**, **OpenRouter**, **NVIDIA NIM**, **Hugging Face**, and local **Ollama** instances.
- **Smart Fallback Engine**: If a model throws a rate-limit or error mid-pipeline, it automatically fails over to the next best ranked model.
- **High-DPI Slide Extraction**: Renders lecture PDF pages into pristine 300-DPI PNG assets via `PyMuPDF`.
- **Vision OCR & Transcriber**: Converts complex chemical structures, mathematical matrices, wavefunctions, and graphs into rigorous Markdown + LaTeX (`$` inline, `$$` block equations).
- **Dockerized & Non-Interactive Ready**: Easily containerized for local execution with full support for headless, automated batch tasks via CLI flags.

---

## Project Structure

```text
├── pipeline.py            # Main automation script
├── Dockerfile             # Container definition
├── docker-compose.yml     # Compose configuration and volume mounts
├── requirements.txt       # Python package dependencies
├── .env.example           # Template for API keys
└── quant_chem/            # Local data directory (mounted in Docker)
    ├── notes/             # Place raw lecture PDFs here
    ├── assets/            # Rendered 300-DPI page PNGs
    └── transcription/     # Generated Markdown lecture notes
```

---

## Getting Started

### 1. Clone & Configure Environment
Clone your repository, then copy the environment template and insert your free API keys:
```bash
cp .env.example .env
```
Edit `.env` and fill in your keys (e.g., `GEMINI_API_KEY`, `GROQ_API_KEY`, `OPENROUTER_API_KEY`).

### 2. Docker Setup
Ensure Docker Desktop is running, then initialize the model cache file placeholder and build the image:
```bash
touch .free_models_cache.json
docker compose build
```

---

## Usage Guide

### Interactive Mode (Default)
Run commands interactively to choose your preferred free LLM provider or routing strategy from the terminal menu (`1–9`):
```bash
# 1. Extract PDFs to high-DPI slide images
docker compose run --rm pipeline extract quant_chem/notes

# 2. Test Vision OCR on a sample slide
docker compose run --rm pipeline verify

# 3. Transcribe all extracted lecture slide folders
docker compose run --rm pipeline transcribe quant_chem/assets
```

### Automated / Non-Interactive Batch Mode
You can bypass the interactive menu by specifying the `--option` and optional `--model` flags:

- **Option 3 (Groq Free Tier)**:
  ```bash
  docker compose run --rm pipeline transcribe quant_chem/assets --option 3
  ```
- **Option 7 (Top-10 Token-Ranked Fallback)**:
  ```bash
  docker compose run --rm pipeline transcribe quant_chem/assets --option 7
  ```
- **Option 9 (Custom Model Specification)**:
  ```bash
  docker compose run --rm pipeline transcribe quant_chem/assets --option 9 --model groq:llama-3.3-70b-versatile
  ```

---
