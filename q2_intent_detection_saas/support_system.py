"""
Main Customer Support System
Integrates LLM wrapper, intent detection, and specialized processors
"""

import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from llm_wrapper import LLMWrapper, LLMResponse
from intent_detector import IntentDetector, IntentResult
from processors import TechnicalProcessor, BillingProcessor, FeatureRequestProcessor

logger = logging.getLogger(__name__)

@dataclass
class SupportResponse:
    """Complete support response with all metadata"""
    query: str
    intent: IntentResult
    response: str
    processor_used: str
    model_used: str
    response_time: float
    tokens_used: int
    confidence: float
    metadata: Dict[str, Any]

class CustomerSupportSystem:
    """Main customer support system with intent detection and specialized processing"""
    
    def __init__(self):
        # Initialize LLM wrapper
        self.llm_wrapper = LLMWrapper()
        
        # Initialize intent detector
        self.intent_detector = IntentDetector(self.llm_wrapper)
        
        # Initialize specialized processors
        self.technical_processor = TechnicalProcessor(self.llm_wrapper)
        self.billing_processor = BillingProcessor(self.llm_wrapper)
        self.feature_processor = FeatureRequestProcessor(self.llm_wrapper)
        
        # Processor mapping
        self.processors = {
            "technical": self.technical_processor,
            "billing": self.billing_processor,
            "feature": self.feature_processor
        }
        
        # Performance tracking
        self.stats = {
            "total_queries": 0,
            "intent_distribution": {"technical": 0, "billing": 0, "feature": 0},
            "avg_response_time": 0.0,
            "total_tokens": 0,
            "success_rate": 0.0
        }
        
        logger.info("Customer Support System initialized successfully")
    
    def process_query(self, query: str, context: Optional[Dict] = None) -> SupportResponse:
        """Process a customer query end-to-end"""
        start_time = time.time()
        
        try:
            # Step 1: Intent detection
            intent_result = self.intent_detector.classify_intent(query)
            
            # Step 2: Route to appropriate processor
            processor = self.processors.get(intent_result.intent, self.technical_processor)
            
            # Step 3: Process with specialized processor
            if intent_result.intent == "technical":
                processor_response = processor.process_query(query, context)
                response_text = processor.format_response(processor_response, include_extras=False)
            elif intent_result.intent == "billing":
                processor_response = processor.process_query(query, context)
                response_text = processor.format_response(processor_response)
            elif intent_result.intent == "feature":
                processor_response = processor.process_query(query, context)
                response_text = processor.format_response(processor_response)
            else:
                # Fallback to technical processor
                processor_response = self.technical_processor.process_query(query, context)
                response_text = processor.format_response(processor_response, include_extras=False)
            
            # Step 4: Calculate metrics
            response_time = time.time() - start_time
            
            # Update statistics
            self._update_stats(intent_result.intent, response_time, 0)  # Token count not available from processors
            
            # Create response object
            support_response = SupportResponse(
                query=query,
                intent=intent_result,
                response=response_text,
                processor_used=intent_result.intent,
                model_used="processor",  # Processors use their own LLM calls
                response_time=response_time,
                tokens_used=0,  # Not tracked by processors
                confidence=intent_result.confidence,
                metadata={
                    "keywords": intent_result.keywords,
                    "reasoning": intent_result.reasoning,
                    "processor_response": processor_response
                }
            )
            
            logger.info(f"Query processed successfully: {intent_result.intent} (confidence: {intent_result.confidence:.2f})")
            return support_response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            
            # Return error response
            error_response = SupportResponse(
                query=query,
                intent=IntentResult(
                    intent="technical",
                    confidence=0.0,
                    keywords=[],
                    reasoning="Error occurred during processing"
                ),
                response="I apologize, but I encountered an error while processing your request. Please try again or contact our support team for assistance.",
                processor_used="error",
                model_used="none",
                response_time=time.time() - start_time,
                tokens_used=0,
                confidence=0.0,
                metadata={"error": str(e)}
            )
            
            return error_response
    
    def process_query_with_llm(self, query: str, context: Optional[Dict] = None) -> SupportResponse:
        """Process query using direct LLM with intent context"""
        start_time = time.time()
        
        try:
            # Step 1: Intent detection
            intent_result = self.intent_detector.classify_intent(query)
            
            # Step 2: Generate LLM response with intent context
            strategy = self.intent_detector.get_processing_strategy(intent_result.intent)
            
            prompt = self._build_llm_prompt(query, intent_result, strategy)
            llm_response = self.llm_wrapper.generate(prompt)
            
            if not llm_response or not llm_response.success:
                # Fallback to processor-based approach
                return self.process_query(query, context)
            
            # Step 3: Update statistics
            response_time = time.time() - start_time
            self._update_stats(intent_result.intent, response_time, llm_response.tokens_used)
            
            # Step 4: Create response object
            support_response = SupportResponse(
                query=query,
                intent=intent_result,
                response=llm_response.content,
                processor_used=f"llm_{intent_result.intent}",
                model_used=llm_response.model_used,
                response_time=response_time,
                tokens_used=llm_response.tokens_used,
                confidence=intent_result.confidence,
                metadata={
                    "keywords": intent_result.keywords,
                    "reasoning": intent_result.reasoning,
                    "strategy": strategy
                }
            )
            
            logger.info(f"LLM query processed: {intent_result.intent} using {llm_response.model_used}")
            return support_response
            
        except Exception as e:
            logger.error(f"Error in LLM processing: {e}")
            return self.process_query(query, context)  # Fallback to processor approach
    
    def _build_llm_prompt(self, query: str, intent_result: IntentResult, strategy: Dict[str, str]) -> str:
        """Build LLM prompt based on intent and strategy with knowledge base context"""
        
        # Get knowledge base context for the intent
        kb_context = self._get_knowledge_base_context(intent_result.intent)
        
        base_prompts = {
            "technical": f"""You are a technical support specialist for a SaaS API platform.

Customer Query: "{query}"

Knowledge Base Context:
{kb_context}

Provide a helpful, step-by-step solution that includes:
1. Clear explanation of the issue
2. Step-by-step resolution
3. Code examples if applicable
4. Best practices to avoid this issue

Keep the response concise but comprehensive. Focus on practical solutions.""",
            
            "billing": f"""You are a billing support specialist for a SaaS platform.

Customer Query: "{query}"

Knowledge Base Context:
{kb_context}

Provide a helpful, clear response that:
1. Directly addresses the customer's question
2. Includes relevant pricing information if applicable
3. Explains any policies or procedures
4. Provides clear next steps
5. Is professional and customer-friendly

Keep the response concise but informative.""",
            
            "feature": f"""You are a product manager for a SaaS platform handling feature requests.

Customer Query: "{query}"

Knowledge Base Context:
{kb_context}

Provide a helpful, encouraging response that:
1. Acknowledges the feature request
2. Explains current status and timeline if applicable
3. Mentions alternatives or workarounds
4. Encourages voting on the feature request portal
5. Is positive and shows we value customer input

Keep the response friendly and informative."""
        }
        
        base_prompt = base_prompts.get(intent_result.intent, base_prompts["technical"])
        return base_prompt.format(query=query)
    
    def _get_knowledge_base_context(self, intent: str) -> str:
        """Get relevant knowledge base context for the intent"""
        context = ""
        
        try:
            if intent == "technical":
                # Get technical knowledge base context
                processor = self.processors.get("technical")
                if processor and hasattr(processor, 'knowledge_base'):
                    kb = processor.knowledge_base
                    if "api_documentation" in kb:
                        context += "API Documentation Available: Authentication, Endpoints, Rate Limiting\n"
                    if "troubleshooting" in kb:
                        context += "Troubleshooting Guides: Common Errors, Integration Guides\n"
                    if "best_practices" in kb:
                        context += "Best Practices: Security, Performance, Error Handling\n"
            
            elif intent == "billing":
                # Get billing knowledge base context
                processor = self.processors.get("billing")
                if processor and hasattr(processor, 'knowledge_base'):
                    kb = processor.knowledge_base
                    if "pricing_plans" in kb:
                        plans = list(kb["pricing_plans"].keys())
                        context += f"Available Plans: {', '.join(plans)}\n"
                    if "billing_policies" in kb:
                        policies = list(kb["billing_policies"].keys())
                        context += f"Policies: {', '.join(policies)}\n"
            
            elif intent == "feature":
                # Get feature roadmap context
                processor = self.processors.get("feature")
                if processor and hasattr(processor, 'knowledge_base'):
                    kb = processor.knowledge_base
                    if "roadmap" in kb:
                        context += "Roadmap Available: Current Quarter, Next Quarter, Future Plans\n"
                    if "feature_voting" in kb:
                        context += "Feature Voting: Top Requests, Voting Process\n"
                    if "competitor_analysis" in kb:
                        context += "Competitor Analysis: Feature Comparison, Market Positioning\n"
        
        except Exception as e:
            logger.warning(f"Failed to get knowledge base context: {e}")
            context = "Knowledge base context unavailable"
        
        return context if context else "No specific knowledge base context available"
    
    def _update_stats(self, intent: str, response_time: float, tokens: int):
        """Update system statistics"""
        self.stats["total_queries"] += 1
        self.stats["intent_distribution"][intent] += 1
        self.stats["total_tokens"] += tokens
        
        # Update average response time
        current_avg = self.stats["avg_response_time"]
        total_queries = self.stats["total_queries"]
        self.stats["avg_response_time"] = (current_avg * (total_queries - 1) + response_time) / total_queries
        
        # Update success rate (simplified)
        if response_time < 30:  # Consider under 30 seconds as success
            self.stats["success_rate"] = (self.stats["success_rate"] * (total_queries - 1) + 1) / total_queries
        else:
            self.stats["success_rate"] = (self.stats["success_rate"] * (total_queries - 1)) / total_queries
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        llm_stats = self.llm_wrapper.get_stats()
        health = self.llm_wrapper.health_check()
        
        return {
            "support_system": self.stats,
            "llm_wrapper": llm_stats,
            "health": health,
            "intent_accuracy": self._calculate_intent_accuracy(),
            "processor_usage": self._get_processor_usage()
        }
    
    def _calculate_intent_accuracy(self) -> float:
        """Calculate intent classification accuracy (simplified)"""
        total = self.stats["total_queries"]
        if total == 0:
            return 0.0
        
        # Assume 80% accuracy for now (in real system, this would be based on feedback)
        return 0.8
    
    def _get_processor_usage(self) -> Dict[str, int]:
        """Get processor usage statistics"""
        return self.stats["intent_distribution"]
    
    def get_processor(self, intent: str):
        """Get processor for specific intent"""
        return self.processors.get(intent)
    
    def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check"""
        llm_health = self.llm_wrapper.health_check()
        
        return {
            "system_status": "healthy" if llm_health["local_available"] or llm_health["openai_available"] else "degraded",
            "llm_services": llm_health,
            "processors_available": len(self.processors),
            "total_queries_processed": self.stats["total_queries"],
            "avg_response_time": self.stats["avg_response_time"]
        }
    
    def reset_stats(self):
        """Reset system statistics"""
        self.stats = {
            "total_queries": 0,
            "intent_distribution": {"technical": 0, "billing": 0, "feature": 0},
            "avg_response_time": 0.0,
            "total_tokens": 0,
            "success_rate": 0.0
        }
        logger.info("System statistics reset")
    
    def get_intent_examples(self) -> Dict[str, List[str]]:
        """Get example queries for each intent"""
        return {
            "technical": [
                "How do I integrate the API?",
                "Getting 404 error when calling endpoint",
                "Authentication not working",
                "How to handle rate limiting?",
                "API documentation examples"
            ],
            "billing": [
                "How much does the premium plan cost?",
                "I want to cancel my subscription",
                "Update my billing information",
                "What's included in the Pro plan?",
                "Refund policy for annual plans"
            ],
            "feature": [
                "Can you add dark mode?",
                "Need export to PDF functionality",
                "Request for mobile app",
                "When will you support webhooks?",
                "Add support for Python SDK"
            ]
        } 