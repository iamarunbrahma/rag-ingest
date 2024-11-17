import os
import json
import yaml
import time
import argparse
import shutil
import datetime
import traceback
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from tqdm.auto import tqdm
import logging
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
import qdrant_client
import re
import torch
import nest_asyncio
import warnings

from secrets_manager import get_secret
from extract import MarkdownPDFExtractor
from llama_index.core import (
    Document,
    VectorStoreIndex,
    StorageContext,
    Settings,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage

warnings.filterwarnings("ignore")
torch.cuda.empty_cache()
nest_asyncio.apply()

# Load environment variables and configuration
load_dotenv()

with open(Path("config/config.yaml").resolve(), "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

with open(Path("config/prompts.json").resolve(), "r", encoding="utf-8") as f:
    prompts = json.load(f)

secret = get_secret(config)


class Logger:
    """Handles logging setup and configuration."""

    @staticmethod
    def setup():
        """Initialize logging configuration and return logger instance."""
        log_dir = Path(__file__).parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{Path(__file__).stem}.log"

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file, encoding="utf-8"),
                logging.StreamHandler(),
            ],
        )
        logger = logging.getLogger(__name__)
        logger.info("Logger initialized successfully")
        return logger


logger = Logger.setup()


class TextProcessor:
    """Handles text processing and markdown conversion operations."""

    @staticmethod
    def markdown_to_text(markdown_text):
        """Convert markdown formatted text to plain text."""
        # Remove code blocks
        text = re.sub(r"```[\s\S]*?```", "", markdown_text)
        text = re.sub(r"`[^`\n]+`", "", text)

        # Remove headers
        text = re.sub(r"^#+\s+(.*?)$", r"\1", text, flags=re.MULTILINE)

        # Remove list markers
        text = re.sub(r"^\s*[-*+]\s+(.*?)$", r"\1", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*\d+\.\s+(.*?)$", r"\1", text, flags=re.MULTILINE)

        # Remove images and links
        text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
        text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)

        # Remove formatting
        text = re.sub(r"(\*\*|__)(.*?)\1", r"\2", text)
        text = re.sub(r"(\*|_)(.*?)\1", r"\2", text)
        text = re.sub(r"\^([^\s^]+)(?:\^|(?=\s|$))", r"\1", text)
        text = re.sub(r"~([^\s~]+)(?:~|(?=\s|$))", r"\1", text)

        # Remove horizontal rules and tables
        text = re.sub(r"^\s*[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\|.*\|$", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\+[-+]+\+$", "", text, flags=re.MULTILINE)

        # Remove blockquotes and HTML tags
        text = re.sub(r"^>\s+(.*?)$", r"\1", text, flags=re.MULTILINE)
        text = re.sub(r"<[^>]+>", "", text)

        return text.strip()


class Index:
    """Handles document indexing and vector store operations."""

    # Class-level model instances for reuse
    embed_model = None
    llm_model = None
    qdrant_client = None
    qdrant_aclient = None
    s3_client = None

    def __init__(
        self, persist_dir: str, collection_name: str, debug_mode: bool
    ) -> None:
        """Initialize Index instance with storage and model settings."""
        logger.info(f"Initializing Index with collection: {collection_name}")
        self.persist = Path(persist_dir) / collection_name
        self.collection_name = collection_name
        self.date = str(datetime.date.today())
        self.debug_mode = debug_mode

        if self.debug_mode:
            logger.setLevel(logging.DEBUG)
            logger.info(
                f"Debug mode enabled. Outputs will be saved to {config['DEBUG_DIR']}"
            )
            Path(config["DEBUG_DIR"]).mkdir(parents=True, exist_ok=True)

        self._initialize_components()
        logger.info("Index initialization completed")

    @classmethod
    def _load_embed_model(cls):
        """Load and return the embedding model."""
        if cls.embed_model is None:
            logger.info(f"Loading embedding model: {config['EMBED_MODEL']}")
            cls.embed_model = HuggingFaceEmbedding(model_name=config["EMBED_MODEL"])
        return cls.embed_model

    @classmethod
    @retry(
        stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=60)
    )
    def _load_llm_model(cls):
        """Load and return the language model with retry mechanism."""
        if cls.llm_model is None:
            logger.info(f"Loading LLM model: {config['OLLAMA_MODEL']}")
            cls.llm_model = Ollama(
                model=config["OLLAMA_MODEL"],
                temperature=0.7,
                request_timeout=600.0,
                additional_kwargs={
                    "num_ctx": config["OLLAMA_CTX_LENGTH"],
                    "num_predict": config["OLLAMA_PREDICT_LENGTH"],
                    "cache": False,
                },
            )
        return cls.llm_model

    @classmethod
    def _get_qdrant_client(cls):
        """Initialize and return Qdrant client."""
        if cls.qdrant_client is None:
            logger.info("Initializing Qdrant client")
            cls.qdrant_client = qdrant_client.QdrantClient(
                url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY")
            )
        return cls.qdrant_client

    @classmethod
    def _get_qdrant_aclient(cls):
        """Initialize and return async Qdrant client."""
        if cls.qdrant_aclient is None:
            logger.info("Initializing async Qdrant client")
            cls.qdrant_aclient = qdrant_client.AsyncQdrantClient(
                url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY")
            )
        return cls.qdrant_aclient

    @classmethod
    def _get_s3_client(cls):
        """Initialize and return S3 client."""
        if cls.s3_client is None:
            cls.s3_client = boto3.client(
                "s3",
                aws_access_key_id=secret["AWS_ACCESS_KEY_ID"],
                aws_secret_access_key=secret["AWS_SECRET_ACCESS_KEY"],
                region_name=config["AWS_REGION"],
            )

        return cls.s3_client

    def _initialize_components(self):
        """Initialize all required components for indexing."""
        logger.info("Initializing components")
        self.splitter = SentenceSplitter(
            chunk_size=config["CHUNK_SIZE"], chunk_overlap=config["CHUNK_OVERLAP"]
        )
        self.embed = self._load_embed_model()
        self.llm = self._load_llm_model()

        Settings.llm = self.llm
        Settings.embed_model = self.embed

        if not self.debug_mode:
            self.client = self._get_qdrant_client()
            self.aclient = self._get_qdrant_aclient()
            self.s3_client = self._get_s3_client()
            self._setup_storage_context()

    def _setup_storage_context(self):
        """Set up storage context for vector store."""
        logger.info(
            f"Setting up storage context for collection: {self.collection_name}"
        )
        if os.path.exists(self.persist) and [
            "docstore.json",
            "index_store.json",
        ] in os.listdir(self.persist):
            vector_store = QdrantVectorStore(
                client=self.client,
                aclient=self.aclient,
                collection_name=self.collection_name,
                enable_hybrid=config["QDRANT_ENABLE_HYBRID"],
                fastembed_sparse_model=config["FASTEMBED_SPARSE_MODEL"],
                prefer_grpc=False,
            )

            self.storage_context = StorageContext.from_defaults(
                persist_dir=self.persist, vector_store=vector_store
            )
            logger.info("Loaded existing storage context")
        else:
            if self.client.collection_exists(collection_name=self.collection_name):
                logger.info(f"Deleting existing collection: {self.collection_name}")
                self.client.delete_collection(collection_name=self.collection_name)

            vector_store = QdrantVectorStore(
                client=self.client,
                aclient=self.aclient,
                collection_name=self.collection_name,
                enable_hybrid=config["QDRANT_ENABLE_HYBRID"],
                fastembed_sparse_model=config["FASTEMBED_SPARSE_MODEL"],
                prefer_grpc=False,
            )

            self.storage_context = StorageContext.from_defaults(
                vector_store=vector_store
            )
            logger.info("Created new storage context")

    def _process_chunk(
        self, chunk, highlighted_chunk, file_name, file_extension, category, idx, i
    ):
        """Process and create document object from chunk with metadata."""
        metadata = {
            "file_name": file_name,
            "extension": file_extension,
            "category": category,
            "created_at": self.date,
            "page_num": str(idx + 1),
            "chunk_num": str(i + 1),
            "highlighted_chunk": highlighted_chunk,
        }

        return Document(
            id_=f"{file_name}_{str(idx+1)}_{str(i+1)}",
            text=chunk,
            metadata=metadata,
            excluded_llm_metadata_keys=[
                "file_name",
                "created_at",
                "extension",
                "page_num",
                "chunk_num",
                "highlighted_chunk",
            ],
            excluded_embed_metadata_keys=[
                "created_at",
                "extension",
                "page_num",
                "chunk_num",
                "highlighted_chunk",
            ],
            metadata_seperator="\n",
            metadata_template="{key}: {value}",
            text_template="<METADATA>: {metadata_str}\n-----\n<CONTENT>: {content}",
        )

    def _process_page(self, args):
        """Process a single page and return its chunks."""
        idx, curr, page, file_name, file_extension, category, document = args
        chunks = self.splitter.split_text(page)

        page_chunks = []
        for i, chunk in enumerate(chunks):
            highlighted_chunk = TextProcessor.markdown_to_text(chunk)

            if len(re.findall(r"\b\w+\b", highlighted_chunk.strip())) > 128:
                modified_chunk = self._contextual_retrieval(
                    document, highlighted_chunk, curr
                )
                modified_chunk = re.sub(
                    r"assistant:",
                    r"contextual snippet:",
                    modified_chunk,
                    flags=re.IGNORECASE,
                )
                modified_chunk = (
                    modified_chunk
                    + "\n\nmain chunk:\n"
                    + re.sub(r"!\[.*?\]\(.*?\)", "", chunk)
                )

            else:
                modified_chunk = re.sub(r"!\[.*?\]\(.*?\)", "", chunk)

            if self.debug_mode:
                with open(
                    f"{config['DEBUG_DIR']}/{file_name}.md", "a", encoding="utf-8"
                ) as f:
                    if f.tell() == 0:
                        f.write(modified_chunk)
                    else:
                        f.write(f"\n\n{config['PAGE_DELIMITER']}\n\n{modified_chunk}")

            page_chunks.append(
                self._process_chunk(
                    modified_chunk,
                    highlighted_chunk,
                    file_name,
                    file_extension,
                    category,
                    idx,
                    i,
                )
            )

        return page_chunks

    def _contextual_retrieval(self, document, chunk, curr):
        """Retrieve contextual information for a chunk using LLM."""
        start_time = time.time()
        messages = [
            ChatMessage(
                role="system", content=prompts["prompts"][0]["prompt_template"]
            ),
            ChatMessage(
                role="user",
                content=prompts["prompts"][1]["prompt_template"].format(
                    WHOLE_DOCUMENT=document[
                        curr : curr
                        + (config["OLLAMA_CTX_LENGTH"] - config["CHUNK_SIZE"])
                    ]
                ),
            ),
            ChatMessage(
                role="user",
                content=prompts["prompts"][2]["prompt_template"].format(
                    CHUNK_CONTENT=chunk
                ),
            ),
        ]

        try:
            modified_chunk = self.llm.chat(messages)
            elapsed_time = time.time() - start_time
            logger.info(f"LLM request completed in {elapsed_time:.2f} seconds")
            return str(modified_chunk)

        except Exception as err:
            elapsed_time = time.time() - start_time
            logger.warning(
                f"Error during chunk modification after {elapsed_time:.2f} seconds: {str(err)}"
            )
            raise

    def _save_persist_dir(self):
        """Upload persist directory to S3 bucket."""
        try:
            logger.info(
                f"Uploading files to S3 bucket: {config['S3_PERSIST_BUCKET']}/{self.persist}"
            )

            for local_file in Path(self.persist).iterdir():
                if local_file.is_file():
                    s3_key = f"{str(self.persist)}/{local_file.name}"

                    try:
                        logger.info(f"Uploading {local_file.name} to {s3_key}")
                        self.s3_client.upload_file(
                            Filename=str(local_file),
                            Bucket=config["S3_PERSIST_BUCKET"],
                            Key=s3_key,
                            ExtraArgs={"ContentType": "application/json"},
                        )
                    except ClientError as e:
                        logger.error(f"Error uploading file {local_file.name}: {e}")
                        raise

            logger.info(
                f"Successfully uploaded persist directory to S3: {self.persist}"
            )

        except Exception as e:
            logger.error(f"Error in _load_persist_dir: {str(e)}")
            logger.exception(traceback.format_exc())
            raise

    def _save_docs(self, file_paths: list):
        """Save documents to S3 bucket."""
        logger.info(f"Saving documents to S3 bucket: {config['S3_DOCS_BUCKET']}")

        for file_path in file_paths:
            try:
                if not file_path.suffix.lower() == ".pdf":
                    continue

                # Read the file content
                with open(file_path, "rb") as f:
                    file_content = f.read()

                # Upload the file to S3
                logger.info(
                    f"Uploading {Path(file_path).name} to {config['S3_DOCS_BUCKET']}"
                )
                self.s3_client.put_object(
                    Bucket=config["S3_DOCS_BUCKET"],
                    Key=Path(file_path).name,
                    Body=file_content,
                    ContentType="application/pdf",
                    ContentDisposition="inline; filename=" + Path(file_path).name,
                )
                logger.info("Document upload to S3 completed successfully")

            except ClientError as e:
                logger.error(f"Error uploading file {Path(file_path).name}: {e}")
                logger.exception(traceback.format_exc())

    def persist_docs(self, file_paths: list, category: str, md_flag: bool) -> None:
        """Process and persist documents to vector store."""
        logger.info(f"Starting document persistence for {len(file_paths)} files")
        all_chunks = []

        with tqdm(
            file_paths,
            desc="Processing files...",
            initial=1,
            total=len(file_paths),
            leave=False,
        ) as main_progress_bar:
            for index, file_path in enumerate(main_progress_bar, start=1):
                file_name = Path(file_path).stem
                main_progress_bar.set_description(
                    f"File {index}/{len(file_paths)}: {file_name}"
                )
                logger.info(f"Processing file {index}/{len(file_paths)}: {file_name}")

                file_extension = Path(file_path).suffix.lower()

                if md_flag:
                    # Process markdown files
                    with open(file_path, "r", encoding="utf-8") as f:
                        md_content = f.read()
                    markdown_pages = md_content.rstrip(config["PAGE_DELIMITER"]).split(
                        config["PAGE_DELIMITER"]
                    )
                    document = re.sub(r"config['PAGE_DELIMITER']", "", md_content)
                    logger.info(
                        f"Loaded markdown file with {len(markdown_pages)} pages"
                    )
                else:
                    # Process PDF files
                    extractor = MarkdownPDFExtractor(file_path)
                    markdown_content, markdown_pages = extractor.extract()
                    document = re.sub(r"config['PAGE_DELIMITER']", "", markdown_content)
                    logger.info(
                        f"Extracted PDF content with {len(markdown_pages)} pages"
                    )

                curr = 0
                with tqdm(
                    desc=f"Ingesting pages from {file_name}",
                    initial=1,
                    total=len(markdown_pages),
                    leave=False,
                ) as progress_bar:
                    for idx, page in enumerate(markdown_pages):
                        try:
                            result = self._process_page(
                                (
                                    idx,
                                    curr,
                                    page,
                                    file_name,
                                    file_extension,
                                    category,
                                    document,
                                )
                            )
                            all_chunks.extend(result)
                            logger.debug(
                                f"Processed page {idx+1} with {len(result)} chunks"
                            )

                            if idx > 5:
                                curr = sum(
                                    len(text) for text in markdown_pages[: idx - 5]
                                )

                        except Exception as e:
                            logger.error(f"Error processing page {idx+1}: {e}")
                            logger.exception(traceback.format_exc())
                        progress_bar.update(1)

        if not self.debug_mode:
            logger.info("Updating vector store index")
            if self.client.collection_exists(collection_name=self.collection_name):
                index = load_index_from_storage(self.storage_context)
                index.refresh_ref_docs(
                    all_chunks,
                    update_kwargs={"delete_kwargs": {"delete_from_docstore": True}},
                )
                logger.info("Updated existing index with new documents")
            else:
                shutil.rmtree(self.persist, ignore_errors=True)
                index = VectorStoreIndex.from_documents(
                    documents=all_chunks, storage_context=self.storage_context
                )
                logger.info("Created new index with documents")

            Path(self.persist).mkdir(parents=True, exist_ok=True)
            index.storage_context.persist(persist_dir=self.persist)
            logger.info(f"Persisted index to {self.persist}")
            self._save_persist_dir()

            if not md_flag:
                self._save_docs(file_paths)


def main():
    """Main function to handle command line arguments and initiate indexing."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", help="Input file or directory to Index", required=True
    )
    parser.add_argument("--file_category", help="File Category", required=True)
    parser.add_argument("--collection_name", default="rag_llm", help="Collection Name")
    parser.add_argument("--persist_dir", default="persist", help="Persistent Directory")
    parser.add_argument(
        "--md_flag", action="store_true", default=False, help="Process markdown content"
    )
    parser.add_argument(
        "--debug_mode", action="store_true", default=False, help="Debugging Mode"
    )

    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    logger.info(f"Processing input path: {input_path}")

    # Validate file category
    assert args.file_category in [
        "finance",
        "healthcare",
        "oil_gas",
    ], "File category must be either `finance`, `healthcare`, or `oil_gas`"

    # Handle markdown files
    if args.md_flag:
        if input_path.is_file():
            assert (
                input_path.suffix.lower() == ".md"
            ), "Input file must be a Markdown file"
            file_paths = [input_path]
        elif input_path.is_dir():
            file_paths = list(input_path.glob("*.md"))
            assert (
                len(file_paths) > 0
            ), "No Markdown files found in the specified directory"
        else:
            raise ValueError(
                "Invalid input: must be a Markdown file or a directory containing Markdown files"
            )
    # Handle PDF files
    else:
        if input_path.is_file():
            assert input_path.suffix.lower() == ".pdf", "Input file must be a PDF"
            file_paths = [input_path]
        elif input_path.is_dir():
            file_paths = list(input_path.glob("*.pdf"))
            assert len(file_paths) > 0, "No PDF files found in the specified directory"
        else:
            raise ValueError(
                "Invalid input: must be a PDF file or a directory containing PDF files"
            )

    try:
        logger.info("Initializing indexing process")
        index_obj = Index(args.persist_dir, args.collection_name, args.debug_mode)
        index_obj.persist_docs(file_paths, args.file_category, args.md_flag)
        logger.info(f"Successfully indexed {len(file_paths)} file(s)")
    except Exception as e:
        logger.error(f"An error occurred during indexing: {str(e)}")
        logger.exception(traceback.format_exc())


if __name__ == "__main__":
    main()
