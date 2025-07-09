"""
Configuration settings for the Medical Knowledge Assistant RAG system.
"""

import os
from typing import List, Optional
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    # Fallback to loading from current directory
    load_dotenv()


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # OpenAI Configuration
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_model: str = Field("gpt-4", env="OPENAI_MODEL")
    openai_temperature: float = Field(0.1, env="OPENAI_TEMPERATURE")
    openai_max_tokens: int = Field(1000, env="OPENAI_MAX_TOKENS")
    
    # Vector Database Configuration
    chroma_db_path: str = Field("./data/vector_db", env="CHROMA_DB_PATH")
    embedding_model: str = Field("all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    
    # RAGAS Configuration
    ragas_cache_dir: str = Field("./ragas_cache", env="RAGAS_CACHE_DIR")
    ragas_evaluation_batch_size: int = Field(10, env="RAGAS_EVALUATION_BATCH_SIZE")
    ragas_faithfulness_threshold: float = Field(0.90, env="RAGAS_FAITHFULNESS_THRESHOLD")
    ragas_context_precision_threshold: float = Field(0.85, env="RAGAS_CONTEXT_PRECISION_THRESHOLD")
    ragas_context_recall_threshold: float = Field(0.80, env="RAGAS_CONTEXT_RECALL_THRESHOLD")
    ragas_answer_relevancy_threshold: float = Field(0.85, env="RAGAS_ANSWER_RELEVANCY_THRESHOLD")
    
    # API Configuration
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")
    api_workers: int = Field(4, env="API_WORKERS")
    api_reload: bool = Field(True, env="API_RELOAD")
    
    # Document Processing
    max_document_size: str = Field("50MB", env="MAX_DOCUMENT_SIZE")
    supported_formats: List[str] = Field(["pdf", "docx", "txt"], env="SUPPORTED_FORMATS")
    chunk_size: int = Field(1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(200, env="CHUNK_OVERLAP")
    
    # Monitoring and Logging
    log_level: str = Field("INFO", env="LOG_LEVEL")
    metrics_port: int = Field(9090, env="METRICS_PORT")
    enable_prometheus: bool = Field(True, env="ENABLE_PROMETHEUS")
    
    # Safety Configuration
    enable_safety_filter: bool = Field(True, env="ENABLE_SAFETY_FILTER")
    medical_safety_threshold: float = Field(0.95, env="MEDICAL_SAFETY_THRESHOLD")
    blocked_terms_file: str = Field("./data/blocked_terms.txt", env="BLOCKED_TERMS_FILE")
    
    # Performance Configuration
    max_concurrent_requests: int = Field(10, env="MAX_CONCURRENT_REQUESTS")
    request_timeout: int = Field(30, env="REQUEST_TIMEOUT")
    vector_search_top_k: int = Field(5, env="VECTOR_SEARCH_TOP_K")
    
    # Data Storage
    upload_dir: str = Field("./data/uploads", env="UPLOAD_DIR")
    processed_dir: str = Field("./data/processed", env="PROCESSED_DIR")
    evaluation_results_dir: str = Field("./evaluation_results", env="EVALUATION_RESULTS_DIR")
    
    # Development
    debug: bool = Field(False, env="DEBUG")
    environment: str = Field("production", env="ENVIRONMENT")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings 