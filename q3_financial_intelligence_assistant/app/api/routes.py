import time
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from app.models.database import get_db
from app.models.models import Company, Document, Query
from app.api.models import (
    QueryRequest, QueryResponse, CompanyCreate, CompanyResponse,
    DocumentResponse, QueryHistoryResponse, SystemStatsResponse,
    HealthResponse, ErrorResponse
)
from app.rag.rag_service import rag_service
from app.cache.redis_client import redis_client
from app.core.config import settings
import structlog

logger = structlog.get_logger()

# Create router
router = APIRouter()

# Rate limiting storage (in production, use Redis for distributed rate limiting)
request_counts = {}


def check_rate_limit(client_id: str = "default"):
    """Simple rate limiting check"""
    current_time = time.time()
    if client_id not in request_counts:
        request_counts[client_id] = {"count": 0, "window_start": current_time}
    
    # Reset window if needed
    if current_time - request_counts[client_id]["window_start"] > 60:  # 1 minute window
        request_counts[client_id] = {"count": 0, "window_start": current_time}
    
    # Check limit
    if request_counts[client_id]["count"] >= settings.api_rate_limit:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    request_counts[client_id]["count"] += 1


@router.post("/query", response_model=QueryResponse)
async def submit_query(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Submit a financial query for processing"""
    try:
        # Rate limiting
        check_rate_limit()
        
        # Process query
        response_data = await rag_service.process_query(
            query=request.query,
            company_id=request.company_id,
            company_name=request.company_name
        )
        
        # Add original query for database storage
        response_data["original_query"] = request.query
        response_data["company_id"] = request.company_id
        
        # Save to database in background
        background_tasks.add_task(rag_service.save_query_to_db, response_data, db)
        
        return QueryResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Query processing error", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/query/{query_id}", response_model=QueryResponse)
async def get_query_result(query_id: str, db: Session = Depends(get_db)):
    """Get query result by ID"""
    try:
        query = db.query(Query).filter(Query.query_id == query_id).first()
        if not query:
            raise HTTPException(status_code=404, detail="Query not found")
        
        return QueryResponse(
            query_id=query.query_id,
            response=query.response,
            cache_hit=query.cache_hit,
            response_time=query.response_time,
            tokens_used=query.tokens_used,
            cost=query.cost,
            query_type=query.query_type,
            context_sources=query.meta.get("context_sources", 0) if query.meta else 0
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Get query error", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/companies", response_model=List[CompanyResponse])
async def list_companies(db: Session = Depends(get_db)):
    """List all companies"""
    try:
        companies = db.query(Company).all()
        return [
            CompanyResponse(
                id=company.id,
                name=company.name,
                ticker=company.ticker,
                sector=company.sector,
                industry=company.industry,
                market_cap=company.market_cap,
                revenue=company.revenue,
                created_at=company.created_at
            )
            for company in companies
        ]
        
    except Exception as e:
        logger.error("List companies error", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/companies", response_model=CompanyResponse)
async def create_company(company: CompanyCreate, db: Session = Depends(get_db)):
    """Create a new company"""
    try:
        db_company = Company(
            name=company.name,
            ticker=company.ticker,
            sector=company.sector,
            industry=company.industry,
            market_cap=company.market_cap,
            revenue=company.revenue
        )
        
        db.add(db_company)
        db.commit()
        db.refresh(db_company)
        
        return CompanyResponse(
            id=db_company.id,
            name=db_company.name,
            ticker=db_company.ticker,
            sector=db_company.sector,
            industry=db_company.industry,
            market_cap=db_company.market_cap,
            revenue=db_company.revenue,
            created_at=db_company.created_at
        )
        
    except Exception as e:
        logger.error("Create company error", error=str(e))
        db.rollback()
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/documents", response_model=List[DocumentResponse])
async def list_documents(
    company_id: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """List documents with optional company filter"""
    try:
        query = db.query(Document)
        if company_id:
            query = query.filter(Document.company_id == company_id)
        
        documents = query.all()
        
        return [
            DocumentResponse(
                id=doc.id,
                company_id=doc.company_id,
                title=doc.title,
                document_type=doc.document_type,
                file_path=doc.file_path,
                file_size=doc.file_size,
                processed=doc.processed,
                created_at=doc.created_at
            )
            for doc in documents
        ]
        
    except Exception as e:
        logger.error("List documents error", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/documents/upload")
async def upload_document(
    company_id: int,
    title: str,
    document_type: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload a financial document"""
    try:
        # Validate file type
        allowed_types = [".pdf", ".docx", ".txt"]
        if not any(file.filename.lower().endswith(ext) for ext in allowed_types):
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        # Save file (simplified - in production, use proper file storage)
        file_path = f"uploads/{file.filename}"
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Create document record
        document = Document(
            company_id=company_id,
            title=title,
            document_type=document_type,
            file_path=file_path,
            file_size=len(content)
        )
        
        db.add(document)
        db.commit()
        db.refresh(document)
        
        # TODO: Add background task for document processing
        # background_tasks.add_task(process_document, document.id)
        
        return {"message": "Document uploaded successfully", "document_id": document.id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Upload document error", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/documents/{document_id}")
async def delete_document(document_id: int, db: Session = Depends(get_db)):
    """Delete a document"""
    try:
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        db.delete(document)
        db.commit()
        
        return {"message": "Document deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Delete document error", error=str(e))
        db.rollback()
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/query/history", response_model=QueryHistoryResponse)
async def get_query_history(
    company_id: Optional[int] = None,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """Get query history"""
    try:
        queries = await rag_service.get_query_history(company_id, limit, db)
        
        return QueryHistoryResponse(
            queries=queries,
            total_count=len(queries)
        )
        
    except Exception as e:
        logger.error("Get query history error", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Check Redis connection
        redis_healthy = await redis_client.get("health_check")
        
        # Check database connection
        db = next(get_db())
        db_healthy = db.query(Query).limit(1).first() is not None
        
        status = "healthy" if redis_healthy is not None and db_healthy else "unhealthy"
        
        return HealthResponse(
            status=status,
            timestamp=time.time(),
            version="1.0.0",
            uptime=time.time()  # Simplified - in production, track actual uptime
        )
        
    except Exception as e:
        logger.error("Health check error", error=str(e))
        return HealthResponse(
            status="unhealthy",
            timestamp=time.time(),
            version="1.0.0",
            uptime=0
        )


@router.get("/metrics", response_model=SystemStatsResponse)
async def get_system_metrics():
    """Get system performance metrics"""
    try:
        stats = await rag_service.get_system_stats()
        return SystemStatsResponse(**stats)
        
    except Exception as e:
        logger.error("Get metrics error", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/cache/stats")
async def get_cache_stats():
    """Get cache statistics"""
    try:
        stats = await redis_client.get_cache_stats()
        return stats
        
    except Exception as e:
        logger.error("Get cache stats error", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/cache/popular")
async def get_popular_queries(limit: int = 10):
    """Get most popular cached queries"""
    try:
        popular = await redis_client.get_popular_queries(limit)
        return {"popular_queries": popular}
        
    except Exception as e:
        logger.error("Get popular queries error", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error") 