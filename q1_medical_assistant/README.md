# Medical Knowledge Assistant RAG Pipeline

A production-ready Medical Knowledge Assistant that uses Retrieval-Augmented Generation (RAG) with comprehensive RAGAS evaluation for healthcare professionals.

## Features

- **Medical RAG Pipeline**: Document ingestion → Vector DB → Retrieval → OpenAI generation
- **RAGAS Evaluation**: Real-time monitoring of Context Precision, Context Recall, Faithfulness, and Answer Relevancy
- **Safety System**: RAGAS-validated response filtering to prevent harmful medical advice
- **Production API**: RESTful endpoints with monitoring dashboard
- **Medical Data Sources**: PDF processing, drug databases, clinical protocols

## System Architecture

```
Medical Documents → Document Processor → Vector DB (Chroma)
                                        ↓
User Query → API Gateway → RAG Pipeline → OpenAI Generation
                                        ↓
                                    RAGAS Evaluation → Safety Filter → Response
```

## RAGAS Metrics

- **Faithfulness**: >0.90 (medical accuracy)
- **Context Precision**: >0.85
- **Context Recall**: Measures retrieval completeness
- **Answer Relevancy**: Ensures response relevance

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Environment Variables**:
   ```bash
   cp .env.example .env
   # Add your OpenAI API key and other configurations
   ```

3. **Run the System**:
   ```bash
   # Start the API server
   python app.py
   
   # Run RAGAS evaluation
   python ragas_evaluation.py
   ```

4. **Access the Dashboard**:
   - API: http://localhost:8000
   - Monitoring Dashboard: http://localhost:8000/dashboard
   - API Documentation: http://localhost:8000/docs

## API Endpoints

- `POST /query` - Submit medical queries
- `GET /metrics` - Get RAGAS metrics
- `POST /upload` - Upload medical documents
- `GET /dashboard` - Monitoring dashboard

## Docker Deployment

```bash
docker build -t medical-rag .
docker run -p 8000:8000 medical-rag
```

## Project Structure

```
├── app.py                 # FastAPI application
├── ragas_evaluation.py    # RAGAS evaluation pipeline
├── medical_rag/          # Core RAG system
│   ├── document_processor.py
│   ├── vector_store.py
│   ├── retrieval.py
│   └── generation.py
├── ragas_framework/      # RAGAS implementation
│   ├── metrics.py
│   ├── evaluation.py
│   └── monitoring.py
├── data/                 # Medical datasets
├── tests/               # Test suite
└── docker/              # Docker configuration
```

## Safety Features

- RAGAS-validated response filtering
- Medical accuracy thresholds
- Zero harmful medical advice guarantee
- Real-time monitoring and alerting

## Performance Targets

- Response latency p95 < 3 seconds
- Faithfulness >0.90
- Context Precision >0.85

## License

MIT License - See LICENSE file for details. 