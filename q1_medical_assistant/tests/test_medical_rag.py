"""
Tests for Medical Knowledge Assistant RAG system components.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch

from medical_rag.document_processor import MedicalDocumentProcessor
from medical_rag.vector_store import MedicalVectorStore
from medical_rag.generation import MedicalResponseGenerator
from ragas_framework.metrics import MedicalRAGASMetrics
from ragas_framework.evaluation import RAGASEvaluationPipeline


class TestMedicalDocumentProcessor:
    """Test document processor functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = MedicalDocumentProcessor()
    
    def test_validate_medical_content(self):
        """Test medical content validation."""
        medical_text = "Patient diagnosed with diabetes. Treatment includes insulin therapy."
        result = self.processor.validate_medical_content(medical_text)
        
        assert result["is_medical"] is True
        assert len(result["medical_keywords_found"]) > 0
        assert "diagnosis" in result["medical_keywords_found"]
        assert "treatment" in result["medical_keywords_found"]
    
    def test_validate_non_medical_content(self):
        """Test non-medical content validation."""
        non_medical_text = "The weather is sunny today. I went for a walk."
        result = self.processor.validate_medical_content(non_medical_text)
        
        assert result["is_medical"] is False
        assert len(result["medical_keywords_found"]) == 0


class TestMedicalVectorStore:
    """Test vector store functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        with patch('medical_rag.vector_store.settings') as mock_settings:
            mock_settings.chroma_db_path = tempfile.mkdtemp()
            mock_settings.embedding_model = "all-MiniLM-L6-v2"
            mock_settings.vector_search_top_k = 5
            self.vector_store = MedicalVectorStore()
    
    def test_get_collection_stats(self):
        """Test collection statistics retrieval."""
        stats = self.vector_store.get_collection_stats()
        
        assert "total_documents" in stats
        assert "collection_name" in stats
        assert "embedding_model" in stats


class TestMedicalResponseGenerator:
    """Test response generator functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        with patch('medical_rag.generation.settings') as mock_settings:
            mock_settings.openai_api_key = "test_key"
            mock_settings.openai_model = "gpt-4"
            mock_settings.openai_temperature = 0.1
            mock_settings.openai_max_tokens = 1000
            self.generator = MedicalResponseGenerator()
    
    def test_validate_response_safety(self):
        """Test response safety validation."""
        safe_response = "This information is for educational purposes. Consult healthcare professionals."
        context = "Medical information about diabetes treatment."
        
        safety_score = self.generator._validate_response_safety(safe_response, context)
        
        assert 0.0 <= safety_score <= 1.0
        assert safety_score > 0.5  # Should be relatively safe
    
    def test_validate_unsafe_response(self):
        """Test unsafe response validation."""
        unsafe_response = "Take this medication immediately without consulting your doctor."
        context = "Medical information about medication."
        
        safety_score = self.generator._validate_response_safety(unsafe_response, context)
        
        assert 0.0 <= safety_score <= 1.0
        assert safety_score < 0.5  # Should be flagged as unsafe


class TestMedicalRAGASMetrics:
    """Test RAGAS metrics functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.metrics = MedicalRAGASMetrics()
    
    def test_calculate_safety_score(self):
        """Test safety score calculation."""
        safe_answers = [
            "This information is for educational purposes. Consult healthcare professionals.",
            "Always follow your doctor's advice for medical decisions."
        ]
        
        safety_score = self.metrics._calculate_safety_score(safe_answers)
        
        assert 0.0 <= safety_score <= 1.0
        assert safety_score > 0.7  # Should be high for safe answers
    
    def test_calculate_medical_accuracy(self):
        """Test medical accuracy calculation."""
        answers = [
            "Diabetes treatment involves blood sugar monitoring and medication.",
            "Hypertension symptoms include headaches and high blood pressure."
        ]
        ground_truths = [
            "Diabetes management requires blood sugar monitoring and medication.",
            "Hypertension symptoms include headaches and elevated blood pressure."
        ]
        
        accuracy = self.metrics._calculate_medical_accuracy(answers, ground_truths)
        
        assert 0.0 <= accuracy <= 1.0
        assert accuracy > 0.5  # Should have some accuracy


class TestRAGASEvaluationPipeline:
    """Test RAGAS evaluation pipeline."""
    
    def setup_method(self):
        """Set up test fixtures."""
        with patch('ragas_framework.evaluation.settings') as mock_settings:
            mock_settings.evaluation_results_dir = tempfile.mkdtemp()
            mock_settings.ragas_faithfulness_threshold = 0.90
            mock_settings.ragas_context_precision_threshold = 0.85
            mock_settings.ragas_context_recall_threshold = 0.80
            mock_settings.ragas_answer_relevancy_threshold = 0.85
            self.pipeline = RAGASEvaluationPipeline()
    
    def test_check_quality_thresholds(self):
        """Test quality threshold checking."""
        good_metrics = {
            "faithfulness": 0.95,
            "context_precision": 0.90,
            "context_recall": 0.85,
            "answer_relevancy": 0.90
        }
        
        quality_check = self.pipeline._check_quality_thresholds(good_metrics)
        
        assert quality_check["overall_pass"] is True
        assert len(quality_check["failed_metrics"]) == 0
    
    def test_check_quality_thresholds_failed(self):
        """Test quality threshold checking with failed metrics."""
        bad_metrics = {
            "faithfulness": 0.80,  # Below threshold
            "context_precision": 0.90,
            "context_recall": 0.85,
            "answer_relevancy": 0.90
        }
        
        quality_check = self.pipeline._check_quality_thresholds(bad_metrics)
        
        assert quality_check["overall_pass"] is False
        assert "faithfulness" in quality_check["failed_metrics"]


# Integration tests
class TestMedicalRAGIntegration:
    """Integration tests for the complete RAG pipeline."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_questions = [
            "What are the side effects of aspirin?",
            "How is diabetes managed?"
        ]
        self.test_contexts = [
            ["Aspirin can cause stomach upset and bleeding."],
            ["Diabetes management involves diet, exercise, and medication."]
        ]
        self.test_answers = [
            "Aspirin side effects include stomach upset and bleeding. Consult your doctor.",
            "Diabetes management includes diet control, exercise, and medication under medical supervision."
        ]
    
    def test_ragas_evaluation_integration(self):
        """Test complete RAGAS evaluation integration."""
        with patch('ragas_framework.evaluation.settings') as mock_settings:
            mock_settings.evaluation_results_dir = tempfile.mkdtemp()
            mock_settings.ragas_faithfulness_threshold = 0.90
            mock_settings.ragas_context_precision_threshold = 0.85
            mock_settings.ragas_context_recall_threshold = 0.80
            mock_settings.ragas_answer_relevancy_threshold = 0.85
            
            pipeline = RAGASEvaluationPipeline()
            
            # Run evaluation
            results = pipeline.evaluate_batch(
                self.test_questions,
                self.test_contexts,
                self.test_answers
            )
            
            # Verify results structure
            assert "batch_name" in results
            assert "timestamp" in results
            assert "evaluation_time" in results
            assert "num_queries" in results
            assert "metrics" in results
            assert "quality_check" in results
            
            # Verify metrics
            metrics = results["metrics"]
            assert "faithfulness" in metrics
            assert "context_precision" in metrics
            assert "context_recall" in metrics
            assert "answer_relevancy" in metrics
            
            # Verify quality check
            quality_check = results["quality_check"]
            assert "overall_pass" in quality_check
            assert "thresholds_met" in quality_check
    
    def test_medical_metrics_integration(self):
        """Test medical-specific metrics integration."""
        metrics = MedicalRAGASMetrics()
        
        # Calculate custom medical metrics
        custom_metrics = metrics.calculate_custom_medical_metrics(
            self.test_questions,
            self.test_contexts,
            self.test_answers
        )
        
        # Verify custom metrics
        assert "safety_score" in custom_metrics
        assert "completeness" in custom_metrics
        assert "source_utilization" in custom_metrics
        
        # Verify metric ranges
        for metric, score in custom_metrics.items():
            assert 0.0 <= score <= 1.0


# Performance tests
class TestPerformance:
    """Performance tests for the RAG system."""
    
    def test_document_processing_performance(self):
        """Test document processing performance."""
        processor = MedicalDocumentProcessor()
        
        # Create a large test document
        large_text = "Medical information. " * 1000
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(large_text)
            temp_file = f.name
        
        try:
            start_time = time.time()
            documents = processor.process_document(temp_file)
            processing_time = time.time() - start_time
            
            # Should process large documents in reasonable time
            assert processing_time < 10.0  # Less than 10 seconds
            assert len(documents) > 0
            
        finally:
            os.unlink(temp_file)
    
    def test_vector_search_performance(self):
        """Test vector search performance."""
        with patch('medical_rag.vector_store.settings') as mock_settings:
            mock_settings.chroma_db_path = tempfile.mkdtemp()
            mock_settings.embedding_model = "all-MiniLM-L6-v2"
            mock_settings.vector_search_top_k = 5
            
            vector_store = MedicalVectorStore()
            
            start_time = time.time()
            results = vector_store.similarity_search("test query")
            search_time = time.time() - start_time
            
            # Should complete search in reasonable time
            assert search_time < 5.0  # Less than 5 seconds


if __name__ == "__main__":
    pytest.main([__file__]) 