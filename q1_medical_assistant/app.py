"""
Medical Knowledge Assistant RAG API with RAGAS evaluation.

FastAPI application providing RESTful endpoints for medical queries,
document upload, and real-time RAGAS monitoring.
"""

# Load environment variables first
from dotenv import load_dotenv
from pathlib import Path

# Load .env file from the project root
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    # Fallback to loading from current directory
    load_dotenv()

import logging
import time
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import structlog

from medical_rag.config import get_settings
from medical_rag.document_processor import MedicalDocumentProcessor
from medical_rag.vector_store import MedicalVectorStore
from medical_rag.generation import MedicalResponseGenerator
from ragas_framework.evaluation import RAGASEvaluationPipeline
from ragas_framework.monitoring import RAGASMonitor

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()
settings = get_settings()

# Initialize FastAPI app
app = FastAPI(
    title="Medical Knowledge Assistant RAG API",
    description="Production-ready medical RAG system with RAGAS evaluation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for UI
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Initialize components
document_processor = MedicalDocumentProcessor()
vector_store = MedicalVectorStore()
response_generator = MedicalResponseGenerator()
evaluation_pipeline = RAGASEvaluationPipeline()
ragas_monitor = RAGASMonitor()

# Start monitoring
ragas_monitor.start_monitoring()

# Pydantic models
class QueryRequest(BaseModel):
    query: str = Field(..., description="Medical query")
    include_sources: bool = Field(True, description="Include source information")
    evaluate_response: bool = Field(True, description="Evaluate response with RAGAS")

class QueryResponse(BaseModel):
    response: str
    sources: List[Dict[str, Any]]
    generation_time: float
    safety_score: float
    ragas_metrics: Optional[Dict[str, float]] = None
    quality_check: Optional[Dict[str, Any]] = None

class UploadResponse(BaseModel):
    message: str
    documents_processed: int
    chunks_created: int
    processing_time: float

class MetricsResponse(BaseModel):
    current_metrics: Dict[str, Any]
    status: str
    message: Optional[str] = None

class EvaluationRequest(BaseModel):
    questions: List[str]
    contexts: List[List[str]]
    answers: List[str]
    ground_truths: Optional[List[str]] = None
    batch_name: Optional[str] = None

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.0.0",
        "components": {
            "document_processor": "active",
            "vector_store": "active",
            "response_generator": "active",
            "evaluation_pipeline": "active",
            "ragas_monitor": "active"
        }
    }

# Main UI endpoint
@app.get("/", response_class=HTMLResponse)
async def main_ui():
    """
    Main UI interface for Medical Knowledge Assistant.
    
    Provides a comprehensive interface for:
    - Document upload and processing
    - Medical query submission
    - Real-time RAGAS monitoring
    - Response visualization
    """
    return HTMLResponse(content=main_ui_html, status_code=200)

# Main query endpoint
@app.post("/query", response_model=QueryResponse)
async def query_medical_knowledge(request: QueryRequest):
    """
    Submit a medical query and get a RAG-generated response with RAGAS evaluation.
    
    This endpoint processes medical queries through the complete RAG pipeline:
    1. Retrieves relevant medical documents
    2. Generates response using OpenAI
    3. Evaluates response quality with RAGAS
    4. Returns response with safety validation
    """
    try:
        start_time = time.time()
        
        logger.info("Processing medical query", query=request.query[:100])
        
        # Step 1: Retrieve relevant documents
        retrieval_start = time.time()
        retrieved_docs = vector_store.similarity_search(request.query)
        retrieval_time = time.time() - retrieval_start
        
        if not retrieved_docs:
            raise HTTPException(
                status_code=404, 
                detail="No relevant medical documents found for the query"
            )
        
        # Step 2: Generate response
        generation_start = time.time()
        generation_result = response_generator.generate_response(
            request.query, 
            retrieved_docs, 
            request.include_sources
        )
        generation_time = time.time() - generation_start
        
        # Step 3: RAGAS evaluation (if requested)
        ragas_metrics = None
        quality_check = None
        
        if request.evaluate_response:
            evaluation_start = time.time()
            
            # Prepare context for evaluation
            contexts = [[doc.page_content for doc, _ in retrieved_docs]]
            
            # Run RAGAS evaluation
            evaluation_result = evaluation_pipeline.evaluate_single_query(
                request.query,
                contexts[0],
                generation_result["response"]
            )
            
            ragas_metrics = evaluation_result["metrics"]
            quality_check = evaluation_result["quality_check"]
            
            evaluation_time = time.time() - evaluation_start
            
            # Add to monitoring
            ragas_monitor.add_evaluation_result(evaluation_result)
            
            # Handle both context_precision and context_utilization
            context_metric = ragas_metrics.get("context_precision", ragas_metrics.get("context_utilization", 0))
            
            logger.info(
                "RAGAS evaluation completed",
                faithfulness=ragas_metrics.get("faithfulness", 0),
                context_metric=context_metric,
                quality_pass=quality_check.get("overall_pass", False)
            )
        
        # Calculate total processing time
        total_time = time.time() - start_time
        
        # Prepare response
        response = QueryResponse(
            response=generation_result["response"],
            sources=generation_result["sources"],
            generation_time=generation_time,
            safety_score=generation_result["safety_score"],
            ragas_metrics=ragas_metrics,
            quality_check=quality_check
        )
        
        logger.info(
            "Query processed successfully",
            total_time=total_time,
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            safety_score=generation_result["safety_score"]
        )
        
        return response
        
    except Exception as e:
        logger.error("Error processing query", error=str(e))
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# Document upload endpoint
@app.post("/upload", response_model=UploadResponse)
async def upload_medical_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...)
):
    """
    Upload medical documents for processing and indexing.
    
    Supports PDF, DOCX, and TXT files. Documents are processed asynchronously
    and added to the vector store for retrieval.
    """
    try:
        start_time = time.time()
        
        logger.info("Processing document upload", num_files=len(files))
        
        # Validate files
        for file in files:
            if not file.filename:
                raise HTTPException(status_code=400, detail="Invalid file")
            
            file_extension = Path(file.filename).suffix.lower().lstrip('.')
            if file_extension not in settings.supported_formats:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Unsupported file format: {file_extension}"
                )
        
        # Process documents
        total_documents = 0
        total_chunks = 0
        
        for file in files:
            try:
                # Save uploaded file
                upload_path = Path(settings.upload_dir) / file.filename
                upload_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(upload_path, "wb") as buffer:
                    content = await file.read()
                    buffer.write(content)
                
                # Process document
                documents = document_processor.process_document(str(upload_path))
                
                # Add to vector store
                vector_store.add_documents(documents)
                
                total_documents += 1
                total_chunks += len(documents)
                
                logger.info(
                    "Document processed",
                    filename=file.filename,
                    chunks=len(documents)
                )
                
            except Exception as e:
                logger.error(f"Error processing file {file.filename}: {e}")
                continue
        
        processing_time = time.time() - start_time
        
        response = UploadResponse(
            message=f"Successfully processed {total_documents} documents",
            documents_processed=total_documents,
            chunks_created=total_chunks,
            processing_time=processing_time
        )
        
        logger.info(
            "Document upload completed",
            documents_processed=total_documents,
            chunks_created=total_chunks,
            processing_time=processing_time
        )
        
        return response
        
    except Exception as e:
        logger.error("Error in document upload", error=str(e))
        raise HTTPException(status_code=500, detail=f"Error uploading documents: {str(e)}")

# RAGAS evaluation endpoint
@app.post("/evaluate")
async def evaluate_rag_system(request: EvaluationRequest):
    """
    Evaluate RAG system performance using RAGAS metrics.
    
    This endpoint runs comprehensive evaluation on a batch of queries
    and returns detailed metrics including faithfulness, context precision,
    context recall, and answer relevancy.
    """
    try:
        logger.info(
            "Starting RAGAS evaluation",
            num_queries=len(request.questions),
            batch_name=request.batch_name
        )
        
        # Run evaluation
        evaluation_result = evaluation_pipeline.evaluate_batch(
            request.questions,
            request.contexts,
            request.answers,
            request.ground_truths,
            request.batch_name
        )
        
        # Add to monitoring
        ragas_monitor.add_evaluation_result(evaluation_result)
        
        logger.info(
            "RAGAS evaluation completed",
            batch_name=evaluation_result["batch_name"],
            evaluation_time=evaluation_result["evaluation_time"],
            quality_pass=evaluation_result["quality_check"]["overall_pass"]
        )
        
        return evaluation_result
        
    except Exception as e:
        logger.error("Error in RAGAS evaluation", error=str(e))
        raise HTTPException(status_code=500, detail=f"Error in evaluation: {str(e)}")

# Metrics endpoint
@app.get("/metrics", response_model=MetricsResponse)
async def get_ragas_metrics():
    """
    Get current RAGAS metrics and system status.
    
    Returns real-time metrics from the monitoring system including
    current performance indicators and quality scores.
    """
    try:
        current_metrics = ragas_monitor.get_current_metrics()
        
        response = MetricsResponse(
            current_metrics=current_metrics,
            status=current_metrics.get("status", "unknown"),
            message=current_metrics.get("message")
        )
        
        return response
        
    except Exception as e:
        logger.error("Error getting metrics", error=str(e))
        raise HTTPException(status_code=500, detail=f"Error getting metrics: {str(e)}")

# Metrics history endpoint
@app.get("/metrics/history")
async def get_metrics_history(hours: int = 24):
    """
    Get RAGAS metrics history for specified time period.
    
    Args:
        hours: Number of hours to look back (default: 24)
    
    Returns:
        Historical metrics data grouped by time intervals
    """
    try:
        history = ragas_monitor.get_metrics_history(hours)
        return history
        
    except Exception as e:
        logger.error("Error getting metrics history", error=str(e))
        raise HTTPException(status_code=500, detail=f"Error getting history: {str(e)}")

# Alerts endpoint
@app.get("/alerts")
async def get_alerts(hours: int = 24):
    """
    Get system alerts for specified time period.
    
    Args:
        hours: Number of hours to look back (default: 24)
    
    Returns:
        List of system alerts and warnings
    """
    try:
        alerts = ragas_monitor.get_alerts(hours)
        return {"alerts": alerts, "count": len(alerts)}
        
    except Exception as e:
        logger.error("Error getting alerts", error=str(e))
        raise HTTPException(status_code=500, detail=f"Error getting alerts: {str(e)}")

# Vector store statistics
@app.get("/vector-store/stats")
async def get_vector_store_stats():
    """Get vector store statistics and collection information."""
    try:
        stats = vector_store.get_collection_stats()
        return stats
        
    except Exception as e:
        logger.error("Error getting vector store stats", error=str(e))
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

# Monitoring dashboard
@app.get("/dashboard", response_class=HTMLResponse)
async def monitoring_dashboard():
    """
    Real-time monitoring dashboard for RAGAS metrics.
    
    Provides a web interface for monitoring system performance,
    quality metrics, and alerts.
    """
    dashboard_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Medical RAG - RAGAS Monitoring Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; }
            .header { background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
            .metric-card { background: white; padding: 20px; margin: 10px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .metric-value { font-size: 2em; font-weight: bold; color: #3498db; }
            .metric-label { color: #7f8c8d; margin-bottom: 10px; }
            .status-good { color: #27ae60; }
            .status-warning { color: #f39c12; }
            .status-error { color: #e74c3c; }
            .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
            .refresh-btn { background: #3498db; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; }
            .refresh-btn:hover { background: #2980b9; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Medical Knowledge Assistant - RAGAS Monitoring Dashboard</h1>
                <p>Real-time monitoring of RAG system performance and quality metrics</p>
            </div>
            
            <button class="refresh-btn" onclick="refreshMetrics()">Refresh Metrics</button>
            
            <div class="grid" id="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">Faithfulness</div>
                    <div class="metric-value" id="faithfulness">--</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Context Precision/Utilization</div>
                    <div class="metric-value" id="context-precision">--</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Context Recall</div>
                    <div class="metric-value" id="context-recall">--</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Answer Relevancy</div>
                    <div class="metric-value" id="answer-relevancy">--</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Safety Score</div>
                    <div class="metric-value" id="safety-score">--</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Total Evaluations</div>
                    <div class="metric-value" id="total-evaluations">--</div>
                </div>
            </div>
            
            <div class="metric-card">
                <h3>System Status</h3>
                <div id="system-status">Loading...</div>
            </div>
            
            <div class="metric-card">
                <h3>Recent Alerts</h3>
                <div id="alerts">Loading...</div>
            </div>
            </div>
    
    <script src="/static/app.js"></script>
    <script>
            async function refreshMetrics() {
                try {
                    const response = await fetch('/metrics');
                    const data = await response.json();
                    
                    if (data.status === 'active' && data.current_metrics) {
                        const metrics = data.current_metrics;
                        
                        document.getElementById('faithfulness').textContent = 
                            (metrics.faithfulness || 0).toFixed(3);
                        document.getElementById('context-precision').textContent = 
                            (metrics.context_precision || metrics.context_utilization || 0).toFixed(3);
                        document.getElementById('context-recall').textContent = 
                            (metrics.context_recall || 0).toFixed(3);
                        document.getElementById('answer-relevancy').textContent = 
                            (metrics.answer_relevancy || 0).toFixed(3);
                        document.getElementById('safety-score').textContent = 
                            (metrics.safety_score || 0).toFixed(3);
                        document.getElementById('total-evaluations').textContent = 
                            data.status_info?.total_evaluations || 0;
                        
                        // Update status
                        document.getElementById('system-status').innerHTML = 
                            `<p><strong>Status:</strong> ${data.status}</p>
                             <p><strong>Recent Evaluations:</strong> ${data.status_info?.recent_evaluations || 0}</p>
                             <p><strong>Last Evaluation:</strong> ${data.status_info?.last_evaluation || 'N/A'}</p>`;
                    } else {
                        document.getElementById('system-status').innerHTML = 
                            `<p class="status-warning">${data.message || 'No data available'}</p>`;
                    }
                    
                    // Get alerts
                    const alertsResponse = await fetch('/alerts?hours=1');
                    const alertsData = await alertsResponse.json();
                    
                    if (alertsData.alerts && alertsData.alerts.length > 0) {
                        const alertsHtml = alertsData.alerts.map(alert => 
                            `<div class="status-warning">
                                <strong>${alert.severity.toUpperCase()}</strong> - ${alert.timestamp}<br>
                                ${alert.warnings?.join(', ') || alert.message || 'Alert triggered'}
                            </div>`
                        ).join('<br>');
                        document.getElementById('alerts').innerHTML = alertsHtml;
                    } else {
                        document.getElementById('alerts').innerHTML = '<p class="status-good">No alerts in the last hour</p>';
                    }
                    
                } catch (error) {
                    console.error('Error refreshing metrics:', error);
                    document.getElementById('system-status').innerHTML = 
                        '<p class="status-error">Error loading metrics</p>';
                }
            }
            
            // Refresh metrics on load and every 30 seconds
            refreshMetrics();
            setInterval(refreshMetrics, 30000);
        </script>
    </body>
    </html>
    """
    
    return dashboard_html

# Main UI HTML content
main_ui_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Medical Knowledge Assistant - RAG System</title>
    <link rel="stylesheet" href="/static/styles.css">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            text-align: center;
        }
        
        .header h1 {
            color: #2c3e50;
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 700;
        }
        
        .header p {
            color: #7f8c8d;
            font-size: 1.1em;
            margin-bottom: 20px;
        }
        
        .status-indicator {
            display: inline-block;
            padding: 8px 16px;
            background: #27ae60;
            color: white;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: 600;
        }
        
        .main-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }
        
        .card {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
        }
        
        .card h2 {
            color: #2c3e50;
            margin-bottom: 20px;
            font-size: 1.5em;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        
        .upload-area {
            border: 3px dashed #bdc3c7;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            transition: all 0.3s ease;
            cursor: pointer;
            margin-bottom: 20px;
        }
        
        .upload-area:hover {
            border-color: #3498db;
            background: rgba(52, 152, 219, 0.05);
        }
        
        .upload-area.dragover {
            border-color: #27ae60;
            background: rgba(39, 174, 96, 0.1);
        }
        
        .upload-icon {
            font-size: 3em;
            color: #bdc3c7;
            margin-bottom: 15px;
        }
        
        .upload-text {
            color: #7f8c8d;
            font-size: 1.1em;
            margin-bottom: 10px;
        }
        
        .file-input {
            display: none;
        }
        
        .upload-btn {
            background: #3498db;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 1em;
            font-weight: 600;
            transition: all 0.3s ease;
            margin-top: 15px;
        }
        
        .upload-btn:hover {
            background: #2980b9;
            transform: translateY(-2px);
        }
        
        .query-form {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        
        .form-group {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        
        .form-group label {
            font-weight: 600;
            color: #2c3e50;
        }
        
        .form-group textarea {
            padding: 15px;
            border: 2px solid #ecf0f1;
            border-radius: 10px;
            font-size: 1em;
            resize: vertical;
            min-height: 120px;
            transition: border-color 0.3s ease;
        }
        
        .form-group textarea:focus {
            outline: none;
            border-color: #3498db;
        }
        
        .form-options {
            display: flex;
            gap: 20px;
            align-items: center;
        }
        
        .checkbox-group {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .checkbox-group input[type="checkbox"] {
            width: 18px;
            height: 18px;
        }
        
        .submit-btn {
            background: linear-gradient(135deg, #27ae60, #2ecc71);
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 1.1em;
            font-weight: 600;
            transition: all 0.3s ease;
            margin-top: 10px;
        }
        
        .submit-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(39, 174, 96, 0.3);
        }
        
        .submit-btn:disabled {
            background: #bdc3c7;
            cursor: not-allowed;
            transform: none;
        }
        
        .response-section {
            grid-column: 1 / -1;
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            margin-bottom: 30px;
        }
        
        .response-content {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
            border-left: 4px solid #3498db;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        
        .metric-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }
        
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #3498db;
            margin-bottom: 5px;
        }
        
        .metric-label {
            color: #7f8c8d;
            font-size: 0.9em;
        }
        
        .sources-section {
            margin-top: 20px;
        }
        
        .source-item {
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
            border-left: 3px solid #e74c3c;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 40px;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #3498db;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .alert {
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
            font-weight: 600;
        }
        
        .alert-success {
            background: rgba(39, 174, 96, 0.1);
            color: #27ae60;
            border: 1px solid #27ae60;
        }
        
        .alert-error {
            background: rgba(231, 76, 60, 0.1);
            color: #e74c3c;
            border: 1px solid #e74c3c;
        }
        
        .alert-warning {
            background: rgba(243, 156, 18, 0.1);
            color: #f39c12;
            border: 1px solid #f39c12;
        }
        
        .query-suggestions {
            position: absolute;
            top: 100%;
            left: 0;
            right: 0;
            background: white;
            border: 1px solid #ddd;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            z-index: 1000;
            max-height: 200px;
            overflow-y: auto;
            display: none;
        }
        
        .suggestion-item {
            padding: 10px 15px;
            cursor: pointer;
            border-bottom: 1px solid #f0f0f0;
            transition: background-color 0.2s;
        }
        
        .suggestion-item:hover {
            background-color: #f8f9fa;
        }
        
        .suggestion-item:last-child {
            border-bottom: none;
        }
        
        .form-group {
            position: relative;
        }
        
        .dashboard-link {
            position: fixed;
            top: 20px;
            right: 20px;
            background: #e74c3c;
            color: white;
            padding: 12px 20px;
            border-radius: 25px;
            text-decoration: none;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(231, 76, 60, 0.3);
        }
        
        .dashboard-link:hover {
            background: #c0392b;
            transform: translateY(-2px);
        }
        
        @media (max-width: 768px) {
            .main-grid {
                grid-template-columns: 1fr;
            }
            
            .header h1 {
                font-size: 2em;
            }
            
            .form-options {
                flex-direction: column;
                align-items: flex-start;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏥 Medical Knowledge Assistant</h1>
            <p>Advanced RAG System with RAGAS Evaluation for Healthcare Professionals</p>
            <div class="status-indicator" id="system-status">System Online</div>
        </div>
        
        <a href="/dashboard" class="dashboard-link">📊 RAGAS Dashboard</a>
        
        <div class="main-grid">
            <!-- Document Upload Section -->
            <div class="card">
                <h2>📄 Document Upload</h2>
                <div class="upload-area" id="upload-area">
                    <div class="upload-icon">📁</div>
                    <div class="upload-text">Drag & drop medical documents here or click to browse</div>
                    <div class="upload-text">Supported: PDF, DOCX, TXT</div>
                    <input type="file" id="file-input" class="file-input" multiple accept=".pdf,.docx,.txt">
                    <button class="upload-btn" onclick="document.getElementById('file-input').click()">
                        Choose Files
                    </button>
                </div>
                <div id="upload-status"></div>
            </div>
            
            <!-- Query Section -->
            <div class="card">
                <h2>🔍 Medical Query</h2>
                <form class="query-form" id="query-form">
                    <div class="form-group">
                        <label for="query-input">Enter your medical question:</label>
                        <textarea 
                            id="query-input" 
                            placeholder="e.g., What are the contraindications for warfarin therapy?"
                            required
                        ></textarea>
                    </div>
                    
                    <div class="form-options">
                        <div class="checkbox-group">
                            <input type="checkbox" id="include-sources" checked>
                            <label for="include-sources">Include Sources</label>
                        </div>
                        <div class="checkbox-group">
                            <input type="checkbox" id="evaluate-response" checked>
                            <label for="evaluate-response">RAGAS Evaluation</label>
                        </div>
                    </div>
                    
                    <button type="submit" class="submit-btn" id="submit-btn">
                        🔍 Submit Query
                    </button>
                </form>
            </div>
        </div>
        
        <!-- Response Section -->
        <div class="response-section" id="response-section" style="display: none;">
            <h2>📋 Response & Analysis</h2>
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Processing your query with RAGAS evaluation...</p>
            </div>
            <div id="response-content"></div>
        </div>
    </div>
    
    <script>
        // Global variables
        let isProcessing = false;
        
        // Initialize the application
        document.addEventListener('DOMContentLoaded', function() {
            initializeUploadArea();
            initializeQueryForm();
            checkSystemHealth();
        });
        
        // Initialize upload area
        function initializeUploadArea() {
            const uploadArea = document.getElementById('upload-area');
            const fileInput = document.getElementById('file-input');
            
            // Drag and drop functionality
            uploadArea.addEventListener('dragover', function(e) {
                e.preventDefault();
                uploadArea.classList.add('dragover');
            });
            
            uploadArea.addEventListener('dragleave', function(e) {
                e.preventDefault();
                uploadArea.classList.remove('dragover');
            });
            
            uploadArea.addEventListener('drop', function(e) {
                e.preventDefault();
                uploadArea.classList.remove('dragover');
                const files = e.dataTransfer.files;
                handleFileUpload(files);
            });
            
            // File input change
            fileInput.addEventListener('change', function(e) {
                handleFileUpload(e.target.files);
            });
        }
        
        // Handle file upload
        async function handleFileUpload(files) {
            if (files.length === 0) return;
            
            const formData = new FormData();
            for (let file of files) {
                formData.append('files', file);
            }
            
            showAlert('Uploading documents...', 'info');
            
            try {
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    showAlert(`Successfully uploaded ${result.documents_processed} documents with ${result.chunks_created} chunks created.`, 'success');
                } else {
                    showAlert(`Upload failed: ${result.detail}`, 'error');
                }
            } catch (error) {
                showAlert(`Upload error: ${error.message}`, 'error');
            }
        }
        
        // Initialize query form
        function initializeQueryForm() {
            const form = document.getElementById('query-form');
            
            form.addEventListener('submit', async function(e) {
                e.preventDefault();
                
                if (isProcessing) return;
                
                const query = document.getElementById('query-input').value.trim();
                const includeSources = document.getElementById('include-sources').checked;
                const evaluateResponse = document.getElementById('evaluate-response').checked;
                
                if (!query) {
                    showAlert('Please enter a medical query.', 'warning');
                    return;
                }
                
                await submitQuery(query, includeSources, evaluateResponse);
            });
        }
        
        // Submit query
        async function submitQuery(query, includeSources, evaluateResponse) {
            isProcessing = true;
            const submitBtn = document.getElementById('submit-btn');
            const responseSection = document.getElementById('response-section');
            const loading = document.getElementById('loading');
            const responseContent = document.getElementById('response-content');
            
            // Show loading state
            submitBtn.disabled = true;
            submitBtn.textContent = '🔄 Processing...';
            responseSection.style.display = 'block';
            loading.style.display = 'block';
            responseContent.innerHTML = '';
            
            try {
                const response = await fetch('/query', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        query: query,
                        include_sources: includeSources,
                        evaluate_response: evaluateResponse
                    })
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    displayResponse(result);
                } else {
                    showAlert(`Query failed: ${result.detail}`, 'error');
                }
            } catch (error) {
                showAlert(`Query error: ${error.message}`, 'error');
            } finally {
                // Reset loading state
                isProcessing = false;
                submitBtn.disabled = false;
                submitBtn.textContent = '🔍 Submit Query';
                loading.style.display = 'none';
            }
        }
        
        // Display response
        function displayResponse(result) {
            const responseContent = document.getElementById('response-content');
            
            let html = `
                <div class="response-content">
                    <h3>📝 Medical Response</h3>
                    <p>${result.response}</p>
                </div>
            `;
            
            // Display RAGAS metrics if available
            if (result.ragas_metrics) {
                html += `
                    <h3>📊 RAGAS Evaluation Metrics</h3>
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <div class="metric-value">${(result.ragas_metrics.faithfulness || 0).toFixed(3)}</div>
                            <div class="metric-label">Faithfulness</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">${(result.ragas_metrics.context_precision || 0).toFixed(3)}</div>
                            <div class="metric-label">Context Precision</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">${(result.ragas_metrics.context_recall || 0).toFixed(3)}</div>
                            <div class="metric-label">Context Recall</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">${(result.ragas_metrics.answer_relevancy || 0).toFixed(3)}</div>
                            <div class="metric-label">Answer Relevancy</div>
                        </div>
                    </div>
                `;
                
                // Quality check results
                if (result.quality_check) {
                    const qualityStatus = result.quality_check.overall_pass ? 'PASS' : 'FAIL';
                    const qualityClass = result.quality_check.overall_pass ? 'success' : 'error';
                    
                    html += `
                        <div class="alert alert-${qualityClass}">
                            <strong>Quality Check:</strong> ${qualityStatus}
                            ${result.quality_check.warnings ? '<br>Warnings: ' + result.quality_check.warnings.join(', ') : ''}
                        </div>
                    `;
                }
            }
            
            // Display sources if available
            if (result.sources && result.sources.length > 0) {
                html += `
                    <div class="sources-section">
                        <h3>📚 Sources</h3>
                        ${result.sources.map(source => `
                            <div class="source-item">
                                <strong>${source.title || 'Medical Document'}</strong><br>
                                <small>Page: ${source.page || 'N/A'} | Score: ${(source.score || 0).toFixed(3)}</small>
                            </div>
                        `).join('')}
                    </div>
                `;
            }
            
            // Performance metrics
            html += `
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value">${(result.generation_time || 0).toFixed(2)}s</div>
                        <div class="metric-label">Generation Time</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">${(result.safety_score || 0).toFixed(3)}</div>
                        <div class="metric-label">Safety Score</div>
                    </div>
                </div>
            `;
            
            responseContent.innerHTML = html;
        }
        
        // Show alert
        function showAlert(message, type) {
            const statusDiv = document.getElementById('upload-status');
            const alertClass = `alert alert-${type}`;
            
            statusDiv.innerHTML = `<div class="${alertClass}">${message}</div>`;
            
            // Auto-remove after 5 seconds
            setTimeout(() => {
                statusDiv.innerHTML = '';
            }, 5000);
        }
        
        // Check system health
        async function checkSystemHealth() {
            try {
                const response = await fetch('/health');
                const health = await response.json();
                
                const statusIndicator = document.getElementById('system-status');
                if (health.status === 'healthy') {
                    statusIndicator.textContent = 'System Online';
                    statusIndicator.style.background = '#27ae60';
                } else {
                    statusIndicator.textContent = 'System Issues';
                    statusIndicator.style.background = '#e74c3c';
                }
            } catch (error) {
                const statusIndicator = document.getElementById('system-status');
                statusIndicator.textContent = 'System Offline';
                statusIndicator.style.background = '#e74c3c';
            }
        }
        
        // Auto-refresh system health every 30 seconds
        setInterval(checkSystemHealth, 30000);
    </script>
</body>
</html>
"""

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    logger.info("Starting Medical Knowledge Assistant RAG API")
    
    # Ensure directories exist
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.processed_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.evaluation_results_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info("Medical Knowledge Assistant RAG API started successfully")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Medical Knowledge Assistant RAG API")
    ragas_monitor.stop_monitoring()

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        workers=settings.api_workers
    ) 