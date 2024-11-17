# RAG-Ingest: PDF to Markdown Extraction and Indexing for RAG

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
10. [Host OLLAMA locally](#host-ollama-locally)

## Introduction

RAG-Ingest is an advanced tool designed to seamlessly convert PDF documents into markdown format while preserving the original layout and formatting. It leverages state-of-the-art technologies to extract text, images, tables, and code blocks, making them ready for sophisticated natural language processing tasks. The extracted content is indexed in a vector database (Qdrant) to enhance Retrieval Augmented Generation (RAG) capabilities, enabling efficient and context-aware information retrieval.

## Features

- **PDF to Markdown Conversion**: Preserves layout and formatting.
- **Image Handling**: Extracts images with captions using VisionEncoderDecoderModel.
- **Table Processing**: Detects and converts tables to markdown using pdfplumber.
- **Header and Code Block Detection**: Identifies headers and code blocks with language detection.
- **Vector Indexing**: Utilizes Qdrant for efficient retrieval and storage.
- **Multi-PDF Support**: Processes multiple PDFs simultaneously.
- **Configurable Settings**: Allows fine-tuning of extraction parameters.
- **Debug Mode**: Saves extracted content to a temporary directory for debugging.

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

### PDF Extraction (extract.py)
- **Text Processing**
  - Uses PyMuPDF (fitz) for primary text and layout extraction
  - Implements intelligent font size detection for header levels
  - Preserves document structure and formatting
  - Handles bullet points and numbered lists conversion

- **Layout Analysis**
  - Maintains original document layout and spacing
  - Detects and processes multi-column layouts
  - Preserves paragraph structure and indentation
  - Identifies and converts horizontal lines and special characters

- **Special Elements Processing**
  - Extracts and processes tables using pdfplumber
  - Detects and formats code blocks with language identification
  - Handles hyperlinks and cross-references
  - Processes mathematical formulas and special symbols

- **Image Handling**
  - Extracts embedded images with position preservation
  - Uses VisionEncoderDecoderModel for image captioning
  - Implements OCR using Tesseract for text in images
  - Saves images with contextual filenames

### Indexing (index.py)
- **Document Processing**
  - Splits documents into semantic chunks using SentenceSplitter
  - Maintains document metadata and structure
  - Processes multiple files in parallel
  - Handles both PDF and markdown inputs

- **Context Enhancement**
  - Uses OLLAMA for context-aware chunk processing
  - Implements retry mechanism for model requests
  - Maintains document context across chunks
  - Preserves semantic relationships between sections

- **Vector Storage**
  - Integrates with Qdrant for vector storage
  - Implements efficient document refresh strategies
  - Enables hybrid search capabilities
  - Maintains persistent storage with version control

- **Model Integration**
  - Uses HuggingFace embeddings for vector representation
  - Implements OLLAMA for context processing
  - Supports multiple LLM models
  - Handles model loading and resource management

## Troubleshooting

1. **CUDA Memory Issues**
   - If encountering CUDA out-of-memory errors:
     - Reduce `OLLAMA_CTX_LENGTH` in config.yaml (default: 65536)
     - Lower batch size for processing
     - Consider using a smaller model like `llama2:7b` instead of `llama3.1:8b`
     - Clear CUDA cache using `torch.cuda.empty_cache()` before running

2. **OCR and Image Processing**
   - For Tesseract OCR issues:
     - Verify Tesseract installation: `tesseract --version`
     - Set correct path in system environment variables
     - For Windows: Add Tesseract installation directory to PATH
     - Check image quality and DPI settings if OCR accuracy is low

3. **Vector Database Connection**
   - If Qdrant connection fails:
     - Verify `QDRANT_URL` and `QDRANT_API_KEY` in .env file
     - Check if Qdrant service is running and accessible
     - Ensure collection name matches in configuration
     - Try local Qdrant instance for testing: `docker run -p 6333:6333 qdrant/qdrant`

4. **PDF Processing Errors**
   - When PDF extraction fails:
     - Check PDF file permissions and encryption
     - Verify PDF is not corrupted (try opening in different viewers)
     - For large PDFs (>100MB), increase system memory allocation

5. **OLLAMA Model Issues**
   - If OLLAMA model fails to load:
     - Verify OLLAMA service is running: `ollama list`
     - Check model availability: `ollama pull llama3.1:8b`
     - Monitor system resources during model loading
     - Consider reducing model size or using quantized versions

6. **Performance Optimization**
   - For slow processing:
     - Increase worker threads in parallel processing
     - Adjust chunk size in `config.yaml`
     - Use SSD storage for better I/O performance
     - Enable GPU acceleration if available

7. **Environment Setup**
   - Common setup issues:
     - Use Python 3.8+ (verify with `python --version`)
     - Install all dependencies: `pip install -r requirements.txt`
     - Create isolated environment: `python -m venv venv`
     - For M1/M2 Macs, use Miniforge for ARM-compatible packages

8. **Storage and Output**
   - When facing storage issues:
     - Clear output directory regularly: `rm -rf outputs/*`
     - Monitor disk space for vector store
     - Check log files in `logs/` directory for errors
     - Use `persist_dir` argument to manage index storage location

## Host OLLAMA locally

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