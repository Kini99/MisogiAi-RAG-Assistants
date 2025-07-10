# Financial Intelligence RAG System

A production-scale Financial Intelligence RAG (Retrieval-Augmented Generation) system designed to handle concurrent queries on corporate financial reports and earnings data with Redis caching and OpenAI API integration.

## Features

- **Concurrent Request Handling**: Supports 100+ concurrent requests with <2s response time
- **Redis Caching**: Smart caching with TTL (1h real-time, 24h historical) targeting >70% cache hit ratio
- **Document Processing**: Handles corporate annual reports, quarterly earnings, and financial statements
- **Vector Search**: Pinecone/Weaviate integration for semantic search
- **Background Processing**: Celery queue for document ingestion and processing
- **Monitoring**: Real-time system metrics and performance dashboards
- **Load Testing**: Built-in load testing with Locust

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI App   │    │   Redis Cache   │    │  Vector DB      │
│   (Async)       │◄──►│   (Cluster)     │    │  (Pinecone)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PostgreSQL    │    │   Celery Queue  │    │   OpenAI API    │
│   (Metadata)    │    │   (Background)  │    │   (Generation)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.11+
- Redis Server
- PostgreSQL
- OpenAI API Key

### Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. Start Redis and PostgreSQL services

5. Run the application:
   ```bash
   python main.py
   ```

### Environment Variables

Create a `.env` file with the following variables:

```env
# OpenAI
OPENAI_API_KEY=your_openai_api_key

# Database
DATABASE_URL=postgresql://user:password@localhost/financial_rag

# Redis
REDIS_URL=redis://localhost:6379

# Vector DB
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
PINECONE_INDEX_NAME=financial-documents

# Weaviate (Alternative)
WEAVIATE_URL=http://localhost:8080

# API Configuration
API_RATE_LIMIT=100
CACHE_TTL_REALTIME=3600
CACHE_TTL_HISTORICAL=86400
```

## API Endpoints

### Query Endpoints
- `POST /api/v1/query` - Submit financial queries
- `GET /api/v1/query/{query_id}` - Get query results
- `GET /api/v1/companies` - List available companies
- `GET /api/v1/metrics` - Get financial metrics

### Document Management
- `POST /api/v1/documents/upload` - Upload financial documents
- `GET /api/v1/documents` - List processed documents
- `DELETE /api/v1/documents/{doc_id}` - Delete document

### Monitoring
- `GET /api/v1/health` - Health check
- `GET /api/v1/metrics` - System metrics
- `GET /api/v1/cache/stats` - Cache statistics

## Load Testing

Run load tests using Locust:

```bash
locust -f load_tests/locustfile.py --host=http://localhost:8000
```

### Test Scenarios
- **Burst Testing**: 200 concurrent users × 10 minutes
- **Performance Validation**: Response time and cache hit ratio measurement

## Performance Targets

- **Response Time**: <2 seconds for concurrent requests
- **Cache Hit Ratio**: >70%
- **Concurrent Users**: 100+ simultaneous requests
- **Throughput**: 1000+ requests per minute

## Monitoring

Access monitoring dashboards:
- System Metrics: `http://localhost:8000/metrics`
- Cache Statistics: `http://localhost:8000/api/v1/cache/stats`
- Health Check: `http://localhost:8000/api/v1/health`

## Project Structure

```
├── app/
│   ├── api/                 # FastAPI routes
│   ├── core/               # Core configuration
│   ├── models/             # Database models
│   ├── services/           # Business logic
│   ├── cache/              # Redis caching
│   └── rag/                # RAG pipeline
├── load_tests/             # Load testing scripts
├── tests/                  # Unit tests
├── docs/                   # Sample documents
├── main.py                 # Application entry point
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## Development

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black app/
isort app/
```

## License

MIT License 