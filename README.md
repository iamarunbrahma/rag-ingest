# RAG-Ingest: PDF to Markdown Extraction and Indexing for RAG

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Author: Arun Brahma](https://img.shields.io/badge/Author-Arun%20Brahma-blue)](https://github.com/iamarunbrahma)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://linkedin.com/in/iamarunbrahma)
[![Medium](https://img.shields.io/badge/Medium-Follow-black)](https://medium.com/@iamarunbrahma)

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Usage](#usage)
7. [Project Structure](#project-structure)
8. [How It Works](#how-it-works)
9. [Troubleshooting](#troubleshooting)
10. [Host Ollama locally](#host-ollama-locally)
11. [Disclaimer](#disclaimer)

## Introduction
RAG-Ingest is an advanced tool designed to seamlessly convert PDF documents into markdown format while preserving the original layout and formatting. It leverages state-of-the-art technologies to extract text, images, tables, and code blocks, making them ready for sophisticated natural language processing tasks. The extracted content is indexed in a vector database (Qdrant) to enhance Retrieval Augmented Generation (RAG) capabilities, enabling efficient and context-aware information retrieval.


## Features

- **Intelligent PDF Processing**: Advanced text extraction with layout preservation using PyMuPDF
- **Smart Content Structuring**: Automatic detection of headers, lists, code blocks with language identification
- **Context-Aware Processing**: Uses LLaMA model for contextual enhancement of document chunks
- **Hybrid Search Capabilities**: Combines dense and sparse embeddings via Qdrant for improved retrieval
- **Image Processing**: Automated image extraction and captioning using ViT-GPT2
- **Table Extraction**: Converts complex PDF tables to markdown using pdfplumber
- **Code Block Detection**: Language-specific code block identification for multiple programming languages
- **Vector Storage**: Efficient document indexing with Qdrant vector store
- **Cloud Integration**: AWS S3 integration for persistent storage and document management


## Prerequisites

- **Python 3.8+**: Ensure you have Python version 3.8 or higher installed.
- **PyTorch**: Required for deep learning tasks. Install via `pip install torch`.
- **CUDA-compatible GPU**: Recommended for faster processing. Ensure CUDA is installed and configured.
- **Tesseract OCR**: Necessary for image text extraction. Install using:
  - Ubuntu: `sudo apt-get install tesseract-ocr`
  - macOS: `brew install tesseract`
  - Windows: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
- **Ollama**: For context-aware processing. Install using:
  - Linux: `curl -fsSL https://ollama.com/install.sh | sh`
  - macOS: Download from [Ollama](https://ollama.com/download/mac)
  - Windows: Check [Ollama GitHub](https://github.com/ollama/ollama) for updates
- **Qdrant**: Ensure Qdrant service is running for vector storage.


## Installation

- **Clone the Repository:**
  - Run: `git clone https://github.com/iamarunbrahma/rag-ingest.git`
  - Navigate: `cd rag-ingest`

- **Create a Virtual Environment:**
  - Using `venv`:
    - Run: `python -m venv rag-ingest-env`
    - Activate:
      - On macOS/Linux: `source rag-ingest-env/bin/activate`
      - On Windows: `rag-ingest-env\Scripts\activate`
  - Using `conda`:
    - Run: `conda create --name rag-ingest-env python`
    - Activate: `conda activate rag-ingest-env`

- **Install Required Packages:**
  - Run: `pip install -r requirements.txt`

- **Install Tesseract OCR:**
  - On Ubuntu: `sudo apt-get install tesseract-ocr`
  - On macOS: `brew install tesseract`
  - On Windows: Download and install from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

- **Install Ollama:**
  - On Linux: `curl -fsSL https://ollama.com/install.sh | sh`
  - On macOS: Download from [Ollama](https://ollama.com/download/mac)
  - On Windows: Currently in beta, check [Ollama GitHub](https://github.com/ollama/ollama) for updates


## Configuration

- **Environment Variables:**
  - Create a `.env` file in the project root.
  - Add the following variables:
    ```shell
    QDRANT_URL=<your_qdrant_url>
    QDRANT_API_KEY=<your_qdrant_api_key>
    ```

- **Configuration File:**
  - Modify `config/config.yaml` to adjust settings:
    - **Extraction and Indexing:**
      - Set `OLLAMA_CTX_LENGTH` for context length.
      - Set `OLLAMA_PREDICT_LENGTH` for prediction length.
    - **Model and Embedding:**
      - Choose embedding and LLM models.
    - **Output and Storage:**
      - Configure output directory paths.
      - Set page delimiter format.

- **Prompt Customization:**
  - Edit `config/prompts.json` for context-aware chunk modification:
    - Define system roles.
    - Set document processing instructions.
    - Customize chunk modification templates.
    - Establish context-aware processing rules.


## Usage

To extract markdown from a PDF and index it:
```python 
python index.py --input <path> --file_category <category> --collection_name <name> --persist_dir <dir> [--md_flag] [--debug_mode]
```

Arguments:
- `--input`: Path to the input PDF file or directory
- `--file_category`: Category of the file (finance, healthcare, or oil_gas)
- `--collection_name`: Name of the Qdrant collection (default: rag_llm)
- `--persist_dir`: Directory to persist the index (default: persist)
- `--md_flag`: Flag to process markdown files instead of PDFs
- `--debug_mode`: Flag to enable debugging mode


## Project Structure

```bash
├── config/
│ ├── config.yaml
│ └── prompts.json
├── extract.py
├── index.py
├── requirements.txt
└── README.md
```


## How It Works

1. **PDF Extraction (extract.py)**
The extraction process begins by analyzing PDF documents using PyMuPDF for core content extraction. The system intelligently identifies document structure including headers, lists, and code blocks while preserving the original layout. Images are extracted and processed using VisionEncoderDecoderModel for automated captioning, while tables are converted to markdown format using pdfplumber. The process maintains document fidelity by preserving formatting, handling special characters, and processing mathematical formulas.

2. **Indexing (index.py)**
The indexing process starts by splitting documents into semantic chunks using SentenceSplitter while maintaining document metadata. These chunks are enhanced using Ollama for context-aware processing, ensuring semantic relationships between sections are preserved. The processed chunks are then vectorized using HuggingFace embeddings and stored in Qdrant vector store, enabling efficient hybrid search capabilities. The system supports both PDF and markdown inputs, with automatic version control and persistent storage management.


## Troubleshooting

- **Memory Issues**
  - Reduce `OLLAMA_CTX_LENGTH` in config.yaml
  - Process large PDFs in smaller batches
  - Enable debug mode to track memory usage: `--debug_mode`

- **Processing Errors**
  - Verify PDF isn't password-protected
  - Check Tesseract OCR installation
  - Ensure sufficient disk space for image extraction
  - Monitor `logs/extract.log` and `logs/index.log`

- **Vector Store Issues**
  - Verify Qdrant connection settings
  - Check collection name uniqueness
  - Monitor Qdrant logs for indexing errors

- **Cloud Storage**
  - Verify AWS credentials in Secrets Manager
  - Check S3 bucket permissions
  - Ensure sufficient S3 bucket space

- **Model Loading**
  - Verify Ollama installation
  - Check model availability: `ollama list`
  - Monitor GPU memory usage


## Host Ollama locally

- **Installation Options:**
  - Linux: Execute `curl -fsSL https://ollama.com/install.sh | sh`
  - macOS: Download from `https://ollama.com/download/mac`
  - Windows: Currently in beta, check [Ollama GitHub](https://github.com/ollama/ollama) for updates

- **System Requirements:**
  - Minimum 8GB RAM
  - 4GB free disk space per model
  - NVIDIA GPU (optional, but recommended for better performance)

- **Model Management:**
  - Pull models using: `ollama pull <model_name>`
  - List available models: `ollama list`
  - Remove models: `ollama rm <model_name>`
  - Common models:
    ```bash
    ollama pull llama2:7b    # Smaller, faster
    ollama pull llama3.1:8b  # Better quality
    ollama pull mistral:7b   # Good balance
    ```

- **Running the Service:**
  - Linux/WSL: Start with `ollama serve`
  - macOS: Launch Ollama application
  - Verify service: `curl http://localhost:11434/api/version`

- **Environment Setup:**
  - Adjust model parameters in `config/config.yaml`:
    - Context length (OLLAMA_CTX_LENGTH)
    - Prediction length (OLLAMA_PREDICT_LENGTH)

- **Security Considerations:**
  - Run behind firewall/reverse proxy
  - Avoid exposing to public internet
  - Use latest version for security updates


## Disclaimer

This codebase is provided under the MIT License. While efforts have been made to ensure reliability, the codebase is provided "as is" without warranty of any kind. The authors are not responsible for any damages or liabilities arising from its use. This project involves memory-intensive operations and requires appropriate computational resources.