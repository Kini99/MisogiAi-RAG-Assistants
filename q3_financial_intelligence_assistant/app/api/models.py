from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class QueryRequest(BaseModel):
    """Request model for financial queries"""
    query: str = Field(..., description="Financial query to process")
    company_id: Optional[int] = Field(None, description="Company ID for company-specific queries")
    company_name: Optional[str] = Field("the company", description="Company name for context")


class QueryResponse(BaseModel):
    """Response model for financial queries"""
    query_id: str = Field(..., description="Unique query identifier")
    response: str = Field(..., description="Generated response")
    cache_hit: bool = Field(..., description="Whether response was served from cache")
    response_time: float = Field(..., description="Response time in seconds")
    tokens_used: int = Field(..., description="Number of tokens used")
    cost: float = Field(..., description="Estimated cost in USD")
    query_type: str = Field(..., description="Type of query processed")
    context_sources: int = Field(..., description="Number of source documents used")
    error: Optional[str] = Field(None, description="Error message if any")


class CompanyCreate(BaseModel):
    """Request model for creating companies"""
    name: str = Field(..., description="Company name")
    ticker: Optional[str] = Field(None, description="Stock ticker symbol")
    sector: Optional[str] = Field(None, description="Company sector")
    industry: Optional[str] = Field(None, description="Company industry")
    market_cap: Optional[float] = Field(None, description="Market capitalization")
    revenue: Optional[float] = Field(None, description="Annual revenue")


class CompanyResponse(BaseModel):
    """Response model for company data"""
    id: int = Field(..., description="Company ID")
    name: str = Field(..., description="Company name")
    ticker: Optional[str] = Field(None, description="Stock ticker symbol")
    sector: Optional[str] = Field(None, description="Company sector")
    industry: Optional[str] = Field(None, description="Company industry")
    market_cap: Optional[float] = Field(None, description="Market capitalization")
    revenue: Optional[float] = Field(None, description="Annual revenue")
    created_at: datetime = Field(..., description="Creation timestamp")


class DocumentUpload(BaseModel):
    """Request model for document upload"""
    company_id: int = Field(..., description="Company ID")
    title: str = Field(..., description="Document title")
    document_type: str = Field(..., description="Type of document (annual_report, quarterly_earnings, etc.)")


class DocumentResponse(BaseModel):
    """Response model for document data"""
    id: int = Field(..., description="Document ID")
    company_id: int = Field(..., description="Company ID")
    title: str = Field(..., description="Document title")
    document_type: str = Field(..., description="Document type")
    file_path: str = Field(..., description="File path")
    file_size: Optional[int] = Field(None, description="File size in bytes")
    processed: bool = Field(..., description="Whether document has been processed")
    created_at: datetime = Field(..., description="Creation timestamp")


class QueryHistoryResponse(BaseModel):
    """Response model for query history"""
    queries: List[Dict[str, Any]] = Field(..., description="List of historical queries")
    total_count: int = Field(..., description="Total number of queries")


class SystemStatsResponse(BaseModel):
    """Response model for system statistics"""
    cache: Dict[str, Any] = Field(..., description="Cache statistics")
    vector_store: Dict[str, Any] = Field(..., description="Vector store statistics")
    database: Dict[str, Any] = Field(..., description="Database statistics")
    performance: Dict[str, Any] = Field(..., description="Performance targets")


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    version: str = Field(..., description="API version")
    uptime: float = Field(..., description="Service uptime in seconds")


class ErrorResponse(BaseModel):
    """Response model for errors"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(..., description="Error timestamp") 