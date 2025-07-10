"""
Unit tests for RAG service
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from app.rag.rag_service import RAGService


@pytest.fixture
def rag_service():
    """Create RAG service instance for testing"""
    return RAGService()


@pytest.mark.asyncio
async def test_process_query_cache_hit(rag_service):
    """Test query processing with cache hit"""
    # Mock cache response
    rag_service.cache.get = AsyncMock(return_value={
        "response": "Cached response",
        "tokens_used": 100,
        "cost": 0.01,
        "query_type": "financial_metrics"
    })
    
    # Mock other dependencies
    rag_service.llm_service.classify_query = AsyncMock(return_value="financial_metrics")
    rag_service.vector_store.similarity_search = AsyncMock(return_value=[])
    rag_service.llm_service.generate_response = AsyncMock()
    rag_service.cache.set = AsyncMock()
    
    result = await rag_service.process_query("What is the P/E ratio?")
    
    assert result["cache_hit"] is True
    assert result["response"] == "Cached response"
    assert result["tokens_used"] == 100
    assert result["cost"] == 0.01
    assert result["query_type"] == "financial_metrics"


@pytest.mark.asyncio
async def test_process_query_cache_miss(rag_service):
    """Test query processing with cache miss"""
    # Mock cache miss
    rag_service.cache.get = AsyncMock(return_value=None)
    
    # Mock other dependencies
    rag_service.llm_service.classify_query = AsyncMock(return_value="financial_metrics")
    rag_service.vector_store.similarity_search = AsyncMock(return_value=[
        {"content": "Sample content", "meta": {}, "score": 0.8}
    ])
    rag_service.llm_service.generate_response = AsyncMock(return_value={
        "response": "Generated response",
        "response_time": 1.5,
        "tokens_used": 200,
        "cost": 0.02,
        "query_type": "financial_metrics",
        "context_sources": 1
    })
    rag_service.cache.set = AsyncMock()
    
    result = await rag_service.process_query("What is the P/E ratio?")
    
    assert result["cache_hit"] is False
    assert result["response"] == "Generated response"
    assert result["tokens_used"] == 200
    assert result["cost"] == 0.02
    assert result["query_type"] == "financial_metrics"


@pytest.mark.asyncio
async def test_process_query_no_documents(rag_service):
    """Test query processing when no relevant documents found"""
    # Mock cache miss
    rag_service.cache.get = AsyncMock(return_value=None)
    
    # Mock other dependencies
    rag_service.llm_service.classify_query = AsyncMock(return_value="financial_metrics")
    rag_service.vector_store.similarity_search = AsyncMock(return_value=[])
    rag_service.cache.set = AsyncMock()
    
    result = await rag_service.process_query("What is the P/E ratio?")
    
    assert result["cache_hit"] is False
    assert "couldn't find any relevant financial documents" in result["response"]
    assert result["context_sources"] == 0


@pytest.mark.asyncio
async def test_process_query_error_handling(rag_service):
    """Test error handling in query processing"""
    # Mock cache to raise exception
    rag_service.cache.get = AsyncMock(side_effect=Exception("Cache error"))
    
    result = await rag_service.process_query("What is the P/E ratio?")
    
    assert result["cache_hit"] is False
    assert "encountered an error" in result["response"]
    assert "error" in result


@pytest.mark.asyncio
async def test_get_system_stats(rag_service):
    """Test system statistics retrieval"""
    # Mock dependencies
    rag_service.cache.get_cache_stats = AsyncMock(return_value={
        "total_keys": 100,
        "hit_ratio": 75.5
    })
    rag_service.vector_store.get_stats = AsyncMock(return_value={
        "total_vectors": 1000,
        "dimension": 1536
    })
    
    # Mock database session
    mock_db = Mock()
    mock_query = Mock()
    mock_company = Mock()
    
    mock_db.query.return_value.count.return_value = 50
    mock_db.query.return_value.count.return_value = 10
    mock_db.query.return_value.count.return_value = 200
    
    with patch('app.rag.rag_service.get_db') as mock_get_db:
        mock_get_db.return_value = iter([mock_db])
        
        stats = await rag_service.get_system_stats()
        
        assert "cache" in stats
        assert "vector_store" in stats
        assert "database" in stats
        assert "performance" in stats
        assert stats["cache"]["hit_ratio"] == 75.5
        assert stats["vector_store"]["total_vectors"] == 1000


if __name__ == "__main__":
    pytest.main([__file__]) 