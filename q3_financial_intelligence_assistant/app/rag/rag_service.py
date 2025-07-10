import time
import uuid
from typing import Dict, Any, Optional, List
from app.rag.vector_store import vector_store
from app.rag.llm_service import llm_service
from app.cache.redis_client import redis_client
from app.models.database import get_db
from app.models.models import Query, Company
from sqlalchemy.orm import Session
import structlog

logger = structlog.get_logger()


class RAGService:
    """Main RAG service for financial intelligence queries"""
    
    def __init__(self):
        self.vector_store = vector_store
        self.llm_service = llm_service
        self.cache = redis_client
    
    async def process_query(self, query: str, company_id: Optional[int] = None,
                           company_name: str = "the company") -> Dict[str, Any]:
        """Process a financial query through the RAG pipeline"""
        start_time = time.time()
        query_id = str(uuid.uuid4())
        
        try:
            # Check cache first
            cached_response = await self.cache.get(query, company_id)
            if cached_response:
                logger.info("Cache hit for query", query_id=query_id, query=query)
                return {
                    "query_id": query_id,
                    "response": cached_response["response"],
                    "cache_hit": True,
                    "response_time": time.time() - start_time,
                    "tokens_used": cached_response.get("tokens_used", 0),
                    "cost": cached_response.get("cost", 0),
                    "query_type": cached_response.get("query_type", "general")
                }
            
            # Classify query type
            query_type = await self.llm_service.classify_query(query)
            
            # Retrieve relevant documents
            filter_dict = {"company_id": company_id} if company_id else None
            retrieved_docs = await self.vector_store.similarity_search(
                query=query,
                k=5,
                filter_dict=filter_dict
            )
            
            if not retrieved_docs:
                logger.warning("No relevant documents found", query=query)
                return {
                    "query_id": query_id,
                    "response": "I couldn't find any relevant financial documents to answer your query. Please ensure the company has uploaded financial reports or try a different query.",
                    "cache_hit": False,
                    "response_time": time.time() - start_time,
                    "tokens_used": 0,
                    "cost": 0,
                    "query_type": query_type,
                    "context_sources": 0
                }
            
            # Generate response using LLM
            llm_response = await self.llm_service.generate_response(
                query=query,
                context=retrieved_docs,
                company_name=company_name,
                query_type=query_type
            )
            
            # Prepare final response
            response_data = {
                "query_id": query_id,
                "response": llm_response["response"],
                "cache_hit": False,
                "response_time": time.time() - start_time,
                "tokens_used": llm_response["tokens_used"],
                "cost": llm_response["cost"],
                "query_type": query_type,
                "context_sources": llm_response["context_sources"]
            }
            
            # Cache the response
            cache_response = {
                "response": llm_response["response"],
                "tokens_used": llm_response["tokens_used"],
                "cost": llm_response["cost"],
                "query_type": query_type,
                "context_sources": llm_response["context_sources"]
            }
            
            await self.cache.set(
                query=query,
                response=cache_response,
                company_id=company_id,
                query_type=query_type
            )
            
            logger.info("Query processed successfully", 
                       query_id=query_id,
                       query=query,
                       response_time=response_data["response_time"],
                       tokens=llm_response["tokens_used"])
            
            return response_data
            
        except Exception as e:
            logger.error("Query processing failed", error=str(e), query=query)
            return {
                "query_id": query_id,
                "response": "I encountered an error while processing your query. Please try again later.",
                "cache_hit": False,
                "response_time": time.time() - start_time,
                "tokens_used": 0,
                "cost": 0,
                "query_type": "general",
                "error": str(e)
            }
    
    async def save_query_to_db(self, query_data: Dict[str, Any], db: Session) -> None:
        """Save query and response to database"""
        try:
            db_query = Query(
                query_id=query_data["query_id"],
                company_id=query_data.get("company_id"),
                user_query=query_data.get("original_query", ""),
                query_type=query_data.get("query_type", "general"),
                response=query_data["response"],
                cache_hit=query_data["cache_hit"],
                response_time=query_data["response_time"],
                tokens_used=query_data["tokens_used"],
                cost=query_data["cost"],
                metadata={
                    "context_sources": query_data.get("context_sources", 0),
                    "error": query_data.get("error")
                }
            )
            
            db.add(db_query)
            db.commit()
            
            logger.info("Query saved to database", query_id=query_data["query_id"])
            
        except Exception as e:
            logger.error("Failed to save query to database", error=str(e))
            db.rollback()
    
    async def get_query_history(self, company_id: Optional[int] = None, 
                               limit: int = 50, db: Session = None) -> List[Dict[str, Any]]:
        """Get query history from database"""
        try:
            if db is None:
                db = next(get_db())
            
            query = db.query(Query)
            if company_id:
                query = query.filter(Query.company_id == company_id)
            
            queries = query.order_by(Query.created_at.desc()).limit(limit).all()
            
            return [
                {
                    "query_id": q.query_id,
                    "user_query": q.user_query,
                    "response": q.response,
                    "query_type": q.query_type,
                    "cache_hit": q.cache_hit,
                    "response_time": q.response_time,
                    "tokens_used": q.tokens_used,
                    "cost": q.cost,
                    "created_at": q.created_at.isoformat() if q.created_at else None
                }
                for q in queries
            ]
            
        except Exception as e:
            logger.error("Failed to get query history", error=str(e))
            return []
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        try:
            # Get cache stats
            cache_stats = await self.cache.get_cache_stats()
            
            # Get vector store stats
            vector_stats = await self.vector_store.get_stats()
            
            # Get database stats (simplified)
            db = next(get_db())
            total_queries = db.query(Query).count()
            total_companies = db.query(Company).count()
            total_documents = db.query(Query).count()  # Simplified
            
            return {
                "cache": cache_stats,
                "vector_store": vector_stats,
                "database": {
                    "total_queries": total_queries,
                    "total_companies": total_companies,
                    "total_documents": total_documents
                },
                "performance": {
                    "target_response_time": "<2s",
                    "target_cache_hit_ratio": ">70%",
                    "target_concurrent_users": "100+"
                }
            }
            
        except Exception as e:
            logger.error("Failed to get system stats", error=str(e))
            return {}


# Global RAG service instance
rag_service = RAGService() 