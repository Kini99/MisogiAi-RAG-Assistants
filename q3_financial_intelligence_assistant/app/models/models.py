from sqlalchemy import Column, Integer, String, Text, DateTime, Float, Boolean, JSON, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.models.database import Base


class Company(Base):
    """Company information model"""
    __tablename__ = "companies"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), unique=True, index=True, nullable=False)
    ticker = Column(String(10), unique=True, index=True)
    sector = Column(String(100))
    industry = Column(String(100))
    market_cap = Column(Float)
    revenue = Column(Float)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    documents = relationship("Document", back_populates="company")
    queries = relationship("Query", back_populates="company")


class Document(Base):
    """Financial document model"""
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=False)
    title = Column(String(500), nullable=False)
    document_type = Column(String(50), nullable=False)  # annual_report, quarterly_earnings, etc.
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer)
    content_hash = Column(String(64), unique=True, index=True)
    vector_id = Column(String(255), index=True)  # Vector DB document ID
    meta = Column(JSON)  # Additional document metadata
    processed = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    company = relationship("Company", back_populates="documents")
    chunks = relationship("DocumentChunk", back_populates="document")


class DocumentChunk(Base):
    """Document chunk for vector storage"""
    __tablename__ = "document_chunks"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    vector_id = Column(String(255), unique=True, index=True)  # Vector DB chunk ID
    meta = Column(JSON)  # Chunk metadata (page, section, etc.)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    document = relationship("Document", back_populates="chunks")


class Query(Base):
    """User query model"""
    __tablename__ = "queries"
    
    id = Column(Integer, primary_key=True, index=True)
    query_id = Column(String(64), unique=True, index=True, nullable=False)
    company_id = Column(Integer, ForeignKey("companies.id"))
    user_query = Column(Text, nullable=False)
    query_type = Column(String(50))  # financial_metrics, comparison, trend_analysis
    response = Column(Text)
    cache_hit = Column(Boolean, default=False)
    response_time = Column(Float)  # Response time in seconds
    tokens_used = Column(Integer)
    cost = Column(Float)
    meta = Column(JSON)  # Query metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    company = relationship("Company", back_populates="queries")


class CacheEntry(Base):
    """Cache entry model for tracking cache performance"""
    __tablename__ = "cache_entries"
    
    id = Column(Integer, primary_key=True, index=True)
    cache_key = Column(String(255), unique=True, index=True, nullable=False)
    query_hash = Column(String(64), index=True, nullable=False)
    hit_count = Column(Integer, default=0)
    last_accessed = Column(DateTime(timezone=True), onupdate=func.now())
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=False)


class SystemMetrics(Base):
    """System performance metrics"""
    __tablename__ = "system_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Float, nullable=False)
    metric_unit = Column(String(20))
    meta = Column(JSON) 