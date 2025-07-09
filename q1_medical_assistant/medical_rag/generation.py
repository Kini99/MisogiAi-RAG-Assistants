"""
Medical response generation using OpenAI with safety features and medical context.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from openai import OpenAI
from langchain_core.documents import Document as LangchainDocument
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from .config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class MedicalResponseGenerator:
    """Generate medical responses using OpenAI with safety features."""
    
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model
        self.temperature = settings.openai_temperature
        self.max_tokens = settings.openai_max_tokens
        
        # Medical safety prompt
        self.safety_prompt = """You are a medical knowledge assistant. You must:
1. Only provide information based on the provided medical context
2. Always include disclaimers that you are not a doctor
3. Never provide specific medical advice or diagnoses
4. Always recommend consulting healthcare professionals
5. Be accurate and factual in your responses
6. If information is not available in the context, say so clearly

IMPORTANT: This is for informational purposes only. Always consult with qualified healthcare professionals for medical advice."""

        # Medical response template
        self.response_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""{safety_prompt}

Medical Context:
{context}

Question: {question}

Please provide a comprehensive, accurate response based on the medical context provided. If the specific information requested is not available in the context, please:

1. Acknowledge what information is available in the context
2. Provide general educational information about the topic if appropriate
3. Clearly state what specific information is missing
4. Recommend consulting healthcare professionals for specific medical advice

Include relevant information about:
- Medical conditions and symptoms
- Treatment options and medications
- Drug interactions and side effects
- Clinical guidelines and protocols
- Safety considerations and warnings

Remember: This information is for educational purposes only. Always consult healthcare professionals for medical decisions.

Response:"""
        )
        
        logger.info(f"Initialized MedicalResponseGenerator with model {self.model}")
    
    def generate_response(
        self, 
        query: str, 
        context_docs: List[Tuple[LangchainDocument, float]],
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Generate a medical response based on retrieved context.
        
        Args:
            query: Medical query
            context_docs: List of (document, similarity_score) tuples
            include_sources: Whether to include source information
            
        Returns:
            Dictionary with response and metadata
        """
        try:
            start_time = time.time()
            
            # Prepare context from retrieved documents
            context_text = self._prepare_context(context_docs)
            
            # Generate response using OpenAI
            response = self._generate_with_openai(query, context_text)
            
            # Calculate generation time
            generation_time = time.time() - start_time
            
            # Prepare sources if requested
            sources = []
            if include_sources:
                sources = self._extract_sources(context_docs)
            
            # Validate response safety
            safety_score = self._validate_response_safety(response, context_text)
            
            result = {
                "response": response,
                "sources": sources,
                "generation_time": generation_time,
                "safety_score": safety_score,
                "context_used": len(context_docs),
                "model": self.model
            }
            
            logger.info(f"Generated response in {generation_time:.2f}s with safety score {safety_score:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    def _prepare_context(self, context_docs: List[Tuple[LangchainDocument, float]]) -> str:
        """Prepare context text from retrieved documents."""
        context_parts = []
        
        for i, (doc, score) in enumerate(context_docs):
            context_part = f"Source {i+1} (Relevance: {score:.2f}):\n{doc.page_content}\n"
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def _generate_with_openai(self, query: str, context: str) -> str:
        """Generate response using OpenAI API."""
        try:
            # Prepare messages
            messages = [
                SystemMessage(content=self.safety_prompt),
                HumanMessage(content=f"""Medical Context:
{context}

Question: {query}

Please provide a comprehensive, accurate response based on the medical context provided. Include relevant information about medical conditions, treatments, medications, and safety considerations.

Remember: This information is for educational purposes only. Always consult healthcare professionals for medical decisions.""")
            ]
            
            # Generate response
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user" if msg.type == "human" else msg.type, "content": msg.content} for msg in messages],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=0.9,
                frequency_penalty=0.1,
                presence_penalty=0.1
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    def _extract_sources(self, context_docs: List[Tuple[LangchainDocument, float]]) -> List[Dict[str, Any]]:
        """Extract source information from context documents."""
        sources = []
        
        for i, (doc, score) in enumerate(context_docs):
            source_info = {
                "source_id": i + 1,
                "source_file": doc.metadata.get("source", "Unknown"),
                "relevance_score": score,
                "chunk_id": doc.metadata.get("chunk_id", "Unknown"),
                "file_type": doc.metadata.get("file_type", "Unknown")
            }
            sources.append(source_info)
        
        return sources
    
    def _validate_response_safety(self, response: str, context: str) -> float:
        """
        Validate response safety and medical accuracy.
        
        Args:
            response: Generated response
            context: Source context
            
        Returns:
            Safety score between 0 and 1
        """
        try:
            # Check for dangerous medical advice patterns
            dangerous_patterns = [
                "take this medication",
                "you should diagnose",
                "self-treat",
                "ignore your doctor",
                "stop taking prescribed",
                "alternative to prescribed"
            ]
            
            response_lower = response.lower()
            context_lower = context.lower()
            
            # Check for dangerous patterns
            danger_score = 0.0
            for pattern in dangerous_patterns:
                if pattern in response_lower:
                    danger_score += 0.2
            
            # Check if response is grounded in context
            context_words = set(context_lower.split())
            response_words = set(response_lower.split())
            common_words = context_words.intersection(response_words)
            
            if len(response_words) > 0:
                grounding_score = len(common_words) / len(response_words)
            else:
                grounding_score = 0.0
            
            # Check for medical disclaimers
            disclaimer_patterns = [
                "consult healthcare",
                "not a doctor",
                "informational purposes",
                "medical professional",
                "qualified healthcare"
            ]
            
            disclaimer_score = 0.0
            for pattern in disclaimer_patterns:
                if pattern in response_lower:
                    disclaimer_score += 0.1
            
            # Calculate overall safety score
            safety_score = max(0.0, min(1.0, 
                1.0 - danger_score + 
                0.3 * grounding_score + 
                0.2 * disclaimer_score
            ))
            
            return safety_score
            
        except Exception as e:
            logger.error(f"Error validating response safety: {e}")
            return 0.5  # Default to medium safety if validation fails
    
    def generate_batch_responses(
        self, 
        queries: List[str], 
        contexts: List[List[Tuple[LangchainDocument, float]]]
    ) -> List[Dict[str, Any]]:
        """
        Generate responses for multiple queries in batch.
        
        Args:
            queries: List of medical queries
            contexts: List of context documents for each query
            
        Returns:
            List of response dictionaries
        """
        results = []
        
        for i, (query, context_docs) in enumerate(zip(queries, contexts)):
            try:
                result = self.generate_response(query, context_docs)
                results.append(result)
                logger.info(f"Generated batch response {i+1}/{len(queries)}")
            except Exception as e:
                logger.error(f"Error generating batch response {i+1}: {e}")
                results.append({
                    "response": "Error generating response",
                    "sources": [],
                    "generation_time": 0.0,
                    "safety_score": 0.0,
                    "error": str(e)
                })
        
        return results
    
    def generate_medical_summary(self, context_docs: List[Tuple[LangchainDocument, float]]) -> str:
        """
        Generate a medical summary from context documents.
        
        Args:
            context_docs: List of (document, similarity_score) tuples
            
        Returns:
            Medical summary text
        """
        try:
            context_text = self._prepare_context(context_docs)
            
            summary_prompt = f"""Based on the following medical context, provide a concise summary of the key medical information:

{context_text}

Please summarize the main medical points, including:
- Key medical conditions or topics
- Important treatments or medications mentioned
- Safety considerations or warnings
- Clinical guidelines or protocols

Summary:"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": summary_prompt}],
                temperature=0.3,
                max_tokens=500
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating medical summary: {e}")
            raise
    
    def validate_medical_accuracy(self, response: str, context: str) -> Dict[str, Any]:
        """
        Validate medical accuracy of response against context.
        
        Args:
            response: Generated response
            context: Source context
            
        Returns:
            Accuracy validation results
        """
        try:
            # Check for factual consistency
            context_entities = self._extract_medical_entities(context)
            response_entities = self._extract_medical_entities(response)
            
            # Calculate entity overlap
            common_entities = set(context_entities).intersection(set(response_entities))
            entity_accuracy = len(common_entities) / max(len(response_entities), 1)
            
            # Check for contradictions
            contradictions = self._check_contradictions(response, context)
            
            return {
                "entity_accuracy": entity_accuracy,
                "contradictions_found": len(contradictions),
                "contradiction_details": contradictions,
                "overall_accuracy": max(0.0, entity_accuracy - 0.1 * len(contradictions))
            }
            
        except Exception as e:
            logger.error(f"Error validating medical accuracy: {e}")
            return {"error": str(e)}
    
    def _extract_medical_entities(self, text: str) -> List[str]:
        """Extract medical entities from text."""
        # Simple medical entity extraction (in production, use NER models)
        medical_terms = [
            "diagnosis", "treatment", "medication", "symptom", "disease",
            "condition", "therapy", "drug", "dosage", "side effect",
            "contraindication", "interaction", "prescription"
        ]
        
        entities = []
        text_lower = text.lower()
        
        for term in medical_terms:
            if term in text_lower:
                entities.append(term)
        
        return entities
    
    def _check_contradictions(self, response: str, context: str) -> List[str]:
        """Check for contradictions between response and context."""
        contradictions = []
        
        # Simple contradiction checking (in production, use more sophisticated methods)
        response_lower = response.lower()
        context_lower = context.lower()
        
        # Check for conflicting dosage information
        if "dosage" in response_lower and "dosage" in context_lower:
            # This is a simplified check - in production, use more detailed analysis
            pass
        
        return contradictions 