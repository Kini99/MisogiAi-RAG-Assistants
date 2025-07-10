import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    # OpenAI Configuration
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_model: str = Field("gpt-4-turbo-preview", env="OPENAI_MODEL")
    
    # Database Configuration
    database_url: str = Field(..., env="DATABASE_URL")
    
    # Redis Configuration
    redis_url: str = Field("redis://localhost:6379", env="REDIS_URL")
    redis_db: int = Field(0, env="REDIS_DB")
    
    # Vector Database Configuration
    pinecone_api_key: Optional[str] = Field(None, env="PINECONE_API_KEY")
    pinecone_environment: Optional[str] = Field(None, env="PINECONE_ENVIRONMENT")
    pinecone_index_name: str = Field("financial-documents", env="PINECONE_INDEX_NAME")
    
    # Weaviate Configuration
    weaviate_url: Optional[str] = Field(None, env="WEAVIATE_URL")
    weaviate_api_key: Optional[str] = Field(None, env="WEAVIATE_API_KEY")
    
    # API Configuration
    api_rate_limit: int = Field(100, env="API_RATE_LIMIT")
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")
    debug: bool = Field(False, env="DEBUG")
    
    # Cache Configuration
    cache_ttl_realtime: int = Field(3600, env="CACHE_TTL_REALTIME")  # 1 hour
    cache_ttl_historical: int = Field(86400, env="CACHE_TTL_HISTORICAL")  # 24 hours
    cache_ttl_popular: int = Field(604800, env="CACHE_TTL_POPULAR")  # 7 days
    
    # Celery Configuration
    celery_broker_url: str = Field("redis://localhost:6379/1", env="CELERY_BROKER_URL")
    celery_result_backend: str = Field("redis://localhost:6379/2", env="CELERY_RESULT_BACKEND")
    
    # Security
    secret_key: str = Field(..., env="SECRET_KEY")
    access_token_expire_minutes: int = Field(30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # Monitoring
    enable_metrics: bool = Field(True, env="ENABLE_METRICS")
    metrics_port: int = Field(9090, env="METRICS_PORT")
    
    # Logging
    log_level: str = Field("INFO", env="LOG_LEVEL")
    log_format: str = Field("json", env="LOG_FORMAT")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()


# Vector DB selection
def get_vector_db_config():
    """Get vector database configuration based on available settings"""
    if settings.pinecone_api_key and settings.pinecone_environment:
        return {
            "type": "pinecone",
            "api_key": settings.pinecone_api_key,
            "environment": settings.pinecone_environment,
            "index_name": settings.pinecone_index_name
        }
    elif settings.weaviate_url:
        return {
            "type": "weaviate",
            "url": settings.weaviate_url,
            "api_key": settings.weaviate_api_key
        }
    else:
        raise ValueError("Either Pinecone or Weaviate configuration is required") 