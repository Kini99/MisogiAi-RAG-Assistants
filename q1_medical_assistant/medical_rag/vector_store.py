"""
Vector store implementation using ChromaDB for medical document storage and retrieval.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document as LangchainDocument
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from .config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class MedicalVectorStore:
    """Vector store for medical documents using ChromaDB."""
    
    def __init__(self):
        self.db_path = Path(settings.chroma_db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize embeddings model
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize Langchain Chroma vector store
        self.vector_store = Chroma(
            client=self.chroma_client,
            collection_name="medical_documents",
            embedding_function=self.embedding_model,
            persist_directory=str(self.db_path)
        )
        
        logger.info(f"Initialized MedicalVectorStore at {self.db_path}")
    
    def add_documents(self, documents: List[LangchainDocument]) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of Langchain documents to add
        """
        try:
            if not documents:
                logger.warning("No documents provided to add")
                return
            
            # Add documents to vector store
            self.vector_store.add_documents(documents)
            
            # Persist changes
            self.vector_store.persist()
            
            logger.info(f"Added {len(documents)} documents to vector store")
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            raise
    
    def similarity_search(
        self, 
        query: str, 
        k: int = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[LangchainDocument, float]]:
        """
        Perform similarity search for medical queries.
        
        Args:
            query: Medical query string
            k: Number of results to return
            filter_dict: Optional filters for search
            
        Returns:
            List of (document, similarity_score) tuples
        """
        try:
            if k is None:
                k = settings.vector_search_top_k
            
            # Get collection directly for better control
            collection = self.chroma_client.get_collection("medical_documents")
            
            # Get query embedding
            query_embedding = self.embedding_model.embed_query(query)
            
            # Search by embedding
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=filter_dict
            )
            
            # Convert to Langchain format with proper similarity scores
            documents = []
            for i, doc_id in enumerate(results['ids'][0]):
                doc = LangchainDocument(
                    page_content=results['documents'][0][i],
                    metadata=results['metadatas'][0][i] if results['metadatas'] else {}
                )
                distance = results['distances'][0][i] if results['distances'] else 0.0
                # Convert distance to similarity score (0-1 range)
                # ChromaDB uses L2 distance, normalize it properly
                similarity_score = max(0.0, 1.0 - distance)
                documents.append((doc, similarity_score))
            
            logger.info(f"Similarity search for '{query}' returned {len(documents)} results")
            return documents
            
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            raise
    
    def similarity_search_by_vector(
        self, 
        embedding: List[float], 
        k: int = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[LangchainDocument, float]]:
        """
        Perform similarity search using pre-computed embedding.
        
        Args:
            embedding: Pre-computed embedding vector
            k: Number of results to return
            filter_dict: Optional filters for search
            
        Returns:
            List of (document, similarity_score) tuples
        """
        try:
            if k is None:
                k = settings.vector_search_top_k
            
            # Get collection
            collection = self.chroma_client.get_collection("medical_documents")
            
            # Search by embedding
            results = collection.query(
                query_embeddings=[embedding],
                n_results=k,
                where=filter_dict
            )
            
            # Convert to Langchain format
            documents = []
            for i, doc_id in enumerate(results['ids'][0]):
                doc = LangchainDocument(
                    page_content=results['documents'][0][i],
                    metadata=results['metadatas'][0][i] if results['metadatas'] else {}
                )
                distance = results['distances'][0][i] if results['distances'] else 0.0
                documents.append((doc, 1.0 - distance))  # Convert distance to similarity
            
            return documents
            
        except Exception as e:
            logger.error(f"Error in vector similarity search: {e}")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            collection = self.chroma_client.get_collection("medical_documents")
            count = collection.count()
            
            stats = {
                "total_documents": count,
                "collection_name": "medical_documents",
                "embedding_model": settings.embedding_model,
                "db_path": str(self.db_path)
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}
    
    def delete_documents(self, filter_dict: Dict[str, Any]) -> None:
        """
        Delete documents from the vector store based on filter.
        
        Args:
            filter_dict: Filter criteria for documents to delete
        """
        try:
            collection = self.chroma_client.get_collection("medical_documents")
            collection.delete(where=filter_dict)
            logger.info(f"Deleted documents matching filter: {filter_dict}")
            
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            raise
    
    def update_document(self, doc_id: str, new_content: str, new_metadata: Dict[str, Any]) -> None:
        """
        Update a specific document in the vector store.
        
        Args:
            doc_id: Document ID to update
            new_content: New document content
            new_metadata: New document metadata
        """
        try:
            # Get embedding for new content
            embedding = self.embedding_model.embed_query(new_content)
            
            # Update in collection
            collection = self.chroma_client.get_collection("medical_documents")
            collection.update(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[new_content],
                metadatas=[new_metadata]
            )
            
            logger.info(f"Updated document {doc_id}")
            
        except Exception as e:
            logger.error(f"Error updating document {doc_id}: {e}")
            raise
    
    def clear_collection(self) -> None:
        """Clear all documents from the collection."""
        try:
            collection = self.chroma_client.get_collection("medical_documents")
            collection.delete(where={})
            logger.info("Cleared all documents from collection")
            
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            raise
    
    def get_document_by_id(self, doc_id: str) -> Optional[LangchainDocument]:
        """
        Retrieve a specific document by ID.
        
        Args:
            doc_id: Document ID to retrieve
            
        Returns:
            Document if found, None otherwise
        """
        try:
            collection = self.chroma_client.get_collection("medical_documents")
            results = collection.get(ids=[doc_id])
            
            if results['ids']:
                doc = LangchainDocument(
                    page_content=results['documents'][0],
                    metadata=results['metadatas'][0] if results['metadatas'] else {}
                )
                return doc
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving document {doc_id}: {e}")
            return None
    
    def search_by_metadata(self, metadata_filter: Dict[str, Any], k: int = None) -> List[LangchainDocument]:
        """
        Search documents by metadata filters.
        
        Args:
            metadata_filter: Metadata filter criteria
            k: Number of results to return
            
        Returns:
            List of matching documents
        """
        try:
            if k is None:
                k = settings.vector_search_top_k
            
            collection = self.chroma_client.get_collection("medical_documents")
            results = collection.get(
                where=metadata_filter,
                limit=k
            )
            
            documents = []
            for i, doc_id in enumerate(results['ids']):
                doc = LangchainDocument(
                    page_content=results['documents'][i],
                    metadata=results['metadatas'][i] if results['metadatas'] else {}
                )
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error searching by metadata: {e}")
            raise 