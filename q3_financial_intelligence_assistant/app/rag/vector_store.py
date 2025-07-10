import os
from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import Pinecone as LangchainPinecone, Weaviate
from langchain_openai import OpenAIEmbeddings
from app.core.config import settings, get_vector_db_config
import structlog

logger = structlog.get_logger()


class VectorStore:
    """Vector store interface for financial documents"""
    
    def __init__(self):
        self.config = get_vector_db_config()
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=settings.openai_api_key,
            model="text-embedding-ada-002"
        )
        self.vector_store = None
        self._initialize_vector_store()
    
    def _initialize_vector_store(self):
        """Initialize vector store based on configuration"""
        try:
            if self.config["type"] == "pinecone":
                from pinecone import Pinecone
                pc = Pinecone(api_key=self.config["api_key"])
                
                # Create index if it doesn't exist
                if self.config["index_name"] not in pc.list_indexes().names():
                    pc.create_index(
                        name=self.config["index_name"],
                        dimension=1536,  # OpenAI ada-002 embedding dimension
                        metric="cosine"
                    )
                
                self.vector_store = LangchainPinecone.from_existing_index(
                    index_name=self.config["index_name"],
                    embedding=self.embeddings,
                    text_key="text"
                )
                
            elif self.config["type"] == "weaviate":
                import weaviate
                client = weaviate.Client(
                    url=self.config["url"],
                    auth_client_secret=weaviate.AuthApiKey(api_key=self.config["api_key"]) if self.config["api_key"] else None
                )
                
                self.vector_store = Weaviate(
                    client=client,
                    index_name="FinancialDocuments",
                    text_key="text",
                    embedding=self.embeddings
                )
            
            logger.info("Vector store initialized", type=self.config["type"])
            
        except Exception as e:
            logger.error("Vector store initialization failed", error=str(e))
            raise
    
    async def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Add documents to vector store"""
        try:
            texts = [doc["content"] for doc in documents]
            metadatas = [doc["meta"] for doc in documents]
            
            ids = self.vector_store.add_texts(
                texts=texts,
                metadatas=metadatas
            )
            
            logger.info("Documents added to vector store", count=len(ids))
            return ids
            
        except Exception as e:
            logger.error("Failed to add documents to vector store", error=str(e))
            raise
    
    async def similarity_search(self, query: str, k: int = 5, 
                               filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        try:
            if filter_dict:
                results = self.vector_store.similarity_search_with_score(
                    query=query,
                    k=k,
                    filter=filter_dict
                )
            else:
                results = self.vector_store.similarity_search_with_score(
                    query=query,
                    k=k
                )
            
            # Format results
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "content": doc.page_content,
                    "meta": doc.meta,
                    "score": score
                })
            
            logger.info("Similarity search completed", query=query, results_count=len(formatted_results))
            return formatted_results
            
        except Exception as e:
            logger.error("Similarity search failed", error=str(e), query=query)
            raise
    
    async def delete_documents(self, ids: List[str]) -> bool:
        """Delete documents from vector store"""
        try:
            if hasattr(self.vector_store, 'delete'):
                self.vector_store.delete(ids)
                logger.info("Documents deleted from vector store", count=len(ids))
                return True
            else:
                logger.warning("Delete operation not supported for this vector store")
                return False
                
        except Exception as e:
            logger.error("Failed to delete documents from vector store", error=str(e))
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        try:
            if self.config["type"] == "pinecone":
                from pinecone import Pinecone
                pc = Pinecone(api_key=settings.pinecone_api_key)
                index = pc.Index(self.config["index_name"])
                stats = index.describe_index_stats()
                return {
                    "total_vectors": stats.get("total_vector_count", 0),
                    "dimension": stats.get("dimension", 0),
                    "metric": stats.get("metric", ""),
                    "namespaces": stats.get("namespaces", {})
                }
            else:
                return {"type": self.config["type"], "status": "active"}
                
        except Exception as e:
            logger.error("Failed to get vector store stats", error=str(e))
            return {}


# Global vector store instance
vector_store = VectorStore() 