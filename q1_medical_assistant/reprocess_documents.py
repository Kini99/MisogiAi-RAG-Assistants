#!/usr/bin/env python3
"""
Script to reprocess existing documents with better chunking
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

def reprocess_documents():
    """Reprocess existing documents with better chunking"""
    print("=== Reprocessing Documents with Better Chunking ===")
    
    try:
        from medical_rag.vector_store import MedicalVectorStore
        from medical_rag.document_processor import MedicalDocumentProcessor
        
        # Initialize components
        vs = MedicalVectorStore()
        processor = MedicalDocumentProcessor()
        
        # Clear existing collection
        print("Clearing existing vector store...")
        vs.clear_collection()
        
        # Process the PDF again
        pdf_path = Path("data/uploads/dc23sint.pdf")
        if not pdf_path.exists():
            print(f"PDF file not found: {pdf_path}")
            return False
        
        print(f"Processing PDF: {pdf_path}")
        documents = processor.process_document(str(pdf_path))
        print(f"Created {len(documents)} new chunks")
        
        # Show some sample chunks
        print("\nSample chunks:")
        for i in range(min(3, len(documents))):
            print(f"Chunk {i+1}:")
            print(f"  Content: {documents[i].page_content[:200]}...")
            print(f"  Length: {len(documents[i].page_content)} characters")
            print()
        
        # Add to vector store
        print("Adding documents to vector store...")
        vs.add_documents(documents)
        
        # Check stats
        stats = vs.get_collection_stats()
        print(f"Vector store stats: {stats}")
        
        return True
        
    except Exception as e:
        print(f"Error reprocessing documents: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_improved_retrieval():
    """Test retrieval with the improved chunks"""
    print("\n=== Testing Improved Retrieval ===")
    
    try:
        from medical_rag.vector_store import MedicalVectorStore
        from medical_rag.generation import MedicalResponseGenerator
        
        vs = MedicalVectorStore()
        generator = MedicalResponseGenerator()
        
        # Test queries
        test_queries = [
            "What are the symptoms of diabetes?",
            "How is diabetes diagnosed?",
            "What are the treatment guidelines for diabetes?",
            "What is the recommended blood glucose monitoring?",
            "What are the complications of diabetes?"
        ]
        
        for query in test_queries:
            print(f"\n--- Testing: {query} ---")
            
            # Retrieve documents
            retrieved_docs = vs.similarity_search(query, k=3)
            print(f"Retrieved {len(retrieved_docs)} documents")
            
            # Show top result
            if retrieved_docs:
                top_doc, top_score = retrieved_docs[0]
                print(f"Top result (similarity: {top_score:.3f}):")
                print(f"  Content: {top_doc.page_content[:300]}...")
                
                # Generate response
                generation_result = generator.generate_response(
                    query, 
                    retrieved_docs, 
                    include_sources=True
                )
                
                print(f"Response: {generation_result['response'][:400]}...")
                print(f"Safety score: {generation_result['safety_score']}")
            else:
                print("No documents retrieved")
        
    except Exception as e:
        print(f"Error testing improved retrieval: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function"""
    print("Starting Document Reprocessing...")
    
    # Reprocess documents
    success = reprocess_documents()
    
    if success:
        # Test improved retrieval
        test_improved_retrieval()
    
    print("\n=== Reprocessing Complete ===")

if __name__ == "__main__":
    main() 