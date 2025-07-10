"""
Intent Detection System for Customer Support
Classifies queries into Technical Support, Billing/Account, or Feature Requests
"""

import re
import json
import logging
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from llm_wrapper import LLMWrapper

logger = logging.getLogger(__name__)

@dataclass
class IntentResult:
    """Result of intent classification"""
    intent: str
    confidence: float
    keywords: List[str]
    reasoning: str

class IntentDetector:
    """Intent classification system using LLM and keyword matching"""
    
    def __init__(self, llm_wrapper: LLMWrapper):
        self.llm = llm_wrapper
        
        # Intent categories
        self.intents = {
            "technical": "Technical Support",
            "billing": "Billing/Account", 
            "feature": "Feature Request"
        }
        
        # Load knowledge bases for enhanced classification
        self.knowledge_bases = self._load_knowledge_bases()
        
        # Enhanced keyword patterns using knowledge base content
        self.keyword_patterns = self._build_enhanced_keyword_patterns()
        
        # Specialized prompt templates for each intent
        self.classification_prompts = {
            "technical": """You are a technical support classifier for a SaaS API platform. Determine if this query is about technical support, billing/account, or feature request.

Query: "{query}"

Technical Support includes: API issues, integration problems, error messages, setup/configuration, code examples, troubleshooting, authentication issues, performance problems, bugs, crashes, documentation questions, rate limiting, webhooks, SDK usage, deployment issues.

Billing/Account includes: Payment issues, subscription management, pricing questions, account settings, billing information, refunds, plan changes, trial questions.

Feature Request includes: Requests for new functionality, suggestions for improvements, asking if features exist, enhancement requests, new tool requests.

Respond with exactly one word: "technical", "billing", or "feature". Then provide a brief reason (max 20 words).""",
            
            "billing": """You are a billing support classifier for a SaaS platform. Determine if this query is about technical support, billing/account, or feature request.

Query: "{query}"

Technical Support includes: API issues, integration problems, error messages, setup/configuration, code examples, troubleshooting, authentication issues, performance problems, bugs, crashes, documentation questions.

Billing/Account includes: Payment issues, subscription management, pricing questions, account settings, billing information, refunds, plan changes, trial questions, plan upgrades/downgrades, payment methods, billing cycles, account management.

Feature Request includes: Requests for new functionality, suggestions for improvements, asking if features exist, enhancement requests, new tool requests.

Respond with exactly one word: "technical", "billing", or "feature". Then provide a brief reason (max 20 words).""",
            
            "feature": """You are a product manager classifier for a SaaS platform. Determine if this query is about technical support, billing/account, or feature request.

Query: "{query}"

Technical Support includes: API issues, integration problems, error messages, setup/configuration, code examples, troubleshooting, authentication issues, performance problems, bugs, crashes, documentation questions.

Billing/Account includes: Payment issues, subscription management, pricing questions, account settings, billing information, refunds, plan changes, trial questions.

Feature Request includes: Requests for new functionality, suggestions for improvements, asking if features exist, enhancement requests, new tool requests, roadmap questions, feature availability, competitor comparisons, voting on features.

Respond with exactly one word: "technical", "billing", or "feature". Then provide a brief reason (max 20 words)."""
        }
        
        # Confidence thresholds
        self.confidence_thresholds = {
            "keyword": 0.3,
            "llm": 0.7
        }
    
    def _load_knowledge_bases(self) -> Dict[str, Dict]:
        """Load knowledge base files for enhanced classification"""
        knowledge_bases = {}
        data_dir = "data"
        
        try:
            # Load technical knowledge base
            tech_path = os.path.join(data_dir, "technical_kb.json")
            if os.path.exists(tech_path):
                with open(tech_path, 'r') as f:
                    knowledge_bases["technical"] = json.load(f)
            
            # Load billing knowledge base
            billing_path = os.path.join(data_dir, "billing_kb.json")
            if os.path.exists(billing_path):
                with open(billing_path, 'r') as f:
                    knowledge_bases["billing"] = json.load(f)
            
            # Load feature roadmap
            feature_path = os.path.join(data_dir, "feature_roadmap.json")
            if os.path.exists(feature_path):
                with open(feature_path, 'r') as f:
                    knowledge_bases["feature"] = json.load(f)
                    
            logger.info(f"Loaded {len(knowledge_bases)} knowledge bases")
            
        except Exception as e:
            logger.warning(f"Failed to load knowledge bases: {e}")
            knowledge_bases = {}
        
        return knowledge_bases
    
    def _build_enhanced_keyword_patterns(self) -> Dict[str, List[str]]:
        """Build enhanced keyword patterns using knowledge base content"""
        patterns = {
            "technical": [
                r"error|bug|issue|problem|not working|broken|failed|crash|exception",
                r"how to|how do i|tutorial|guide|documentation|api|integration|setup",
                r"404|500|timeout|connection|authentication|authorization|permission",
                r"code|script|function|method|class|library|framework|sdk",
                r"install|configure|deploy|deployment|server|database|backend",
                r"rate limit|webhook|endpoint|request|response|status code"
            ],
            "billing": [
                r"billing|payment|invoice|subscription|plan|pricing|cost|price|charge",
                r"cancel|cancellation|refund|money|credit|debit|card|paypal|stripe",
                r"upgrade|downgrade|plan|tier|premium|basic|pro|enterprise",
                r"account|profile|settings|billing info|payment method|cc|credit card",
                r"trial|free|paid|monthly|yearly|annual|recurring|auto-renew"
            ],
            "feature": [
                r"feature|functionality|capability|option|setting|tool|utility",
                r"add|implement|include|support|enable|provide|offer|have",
                r"request|suggest|propose|idea|enhancement|improvement|new",
                r"missing|need|want|would like|could you|can you|please add",
                r"mobile|app|ios|android|dark mode|export|import|sync|integration",
                r"roadmap|timeline|when|eta|planned|coming soon|future"
            ]
        }
        
        # Enhance patterns with knowledge base content
        if "technical" in self.knowledge_bases:
            tech_kb = self.knowledge_bases["technical"]
            # Add API-specific terms
            if "api_documentation" in tech_kb:
                patterns["technical"].extend([
                    r"oauth|jwt|bearer|token|key|endpoint|method|get|post|put|delete"
                ])
        
        if "billing" in self.knowledge_bases:
            billing_kb = self.knowledge_bases["billing"]
            # Add billing-specific terms
            if "pricing_plans" in billing_kb:
                patterns["billing"].extend([
                    r"basic|pro|enterprise|free|tier|limit|usage|quota"
                ])
        
        if "feature" in self.knowledge_bases:
            feature_kb = self.knowledge_bases["feature"]
            # Add feature-specific terms
            if "roadmap" in feature_kb:
                patterns["feature"].extend([
                    r"q1|q2|q3|q4|quarter|2024|roadmap|timeline|progress"
                ])
        
        return patterns
    
    def classify_intent(self, query: str) -> IntentResult:
        """Classify the intent of a customer query"""
        query_lower = query.lower().strip()
        
        # Step 1: Keyword-based classification
        keyword_result = self._classify_by_keywords(query_lower)
        
        # Step 2: LLM-based classification
        llm_result = self._classify_by_llm(query)
        
        # Step 3: Combine results
        final_result = self._combine_classifications(keyword_result, llm_result, query_lower)
        
        logger.info(f"Intent classification for '{query}': {final_result.intent} (confidence: {final_result.confidence:.2f})")
        
        return final_result
    
    def _classify_by_keywords(self, query: str) -> IntentResult:
        """Classify intent using keyword patterns"""
        scores = {}
        matched_keywords = {}
        
        for intent, patterns in self.keyword_patterns.items():
            score = 0
            keywords = []
            
            for pattern in patterns:
                matches = re.findall(pattern, query, re.IGNORECASE)
                if matches:
                    score += len(matches)
                    keywords.extend(matches)
            
            scores[intent] = score
            matched_keywords[intent] = keywords
        
        # Find the intent with highest score
        if scores:
            best_intent = max(scores, key=scores.get)
            max_score = scores[best_intent]
            total_keywords = sum(len(kw) for kw in matched_keywords.values())
            
            confidence = min(max_score / max(total_keywords, 1), 1.0)
            
            return IntentResult(
                intent=best_intent,
                confidence=confidence,
                keywords=matched_keywords[best_intent],
                reasoning=f"Keyword match: {', '.join(matched_keywords[best_intent])}"
            )
        
        return IntentResult(
            intent="technical",  # Default fallback
            confidence=0.1,
            keywords=[],
            reasoning="No keyword matches found"
        )
    
    def _classify_by_llm(self, query: str) -> IntentResult:
        """Classify intent using LLM"""
        try:
            # Use a generic prompt for classification
            prompt = f"""You are a customer support classifier. Determine if this query is about technical support, billing/account, or feature request.

Query: "{query}"

Technical Support includes: API issues, integration problems, error messages, setup/configuration, code examples, troubleshooting, authentication issues, performance problems, bugs, crashes, documentation questions.

Billing/Account includes: Payment issues, subscription management, pricing questions, account settings, billing information, refunds, plan changes, trial questions.

Feature Request includes: Requests for new functionality, suggestions for improvements, asking if features exist, enhancement requests, new tool requests.

Respond with exactly one word: "technical", "billing", or "feature". Then provide a brief reason (max 20 words)."""

            response = self.llm.generate(prompt)
            
            if response and response.success:
                content = response.content.strip().lower()
                
                # Extract intent from response
                intent = None
                if "technical" in content:
                    intent = "technical"
                elif "billing" in content:
                    intent = "billing"
                elif "feature" in content:
                    intent = "feature"
                
                if intent:
                    # Extract reasoning (everything after the intent word)
                    reasoning_start = content.find(intent) + len(intent)
                    reasoning = content[reasoning_start:].strip()
                    
                    return IntentResult(
                        intent=intent,
                        confidence=0.8,  # High confidence for LLM
                        keywords=[],
                        reasoning=f"LLM classification: {reasoning}"
                    )
            
        except Exception as e:
            logger.warning(f"LLM classification failed: {e}")
        
        return IntentResult(
            intent="technical",
            confidence=0.1,
            keywords=[],
            reasoning="LLM classification failed"
        )
    
    def _combine_classifications(self, keyword_result: IntentResult, llm_result: IntentResult, query: str) -> IntentResult:
        """Combine keyword and LLM classification results"""
        
        # If both agree, use higher confidence
        if keyword_result.intent == llm_result.intent:
            final_intent = keyword_result.intent
            confidence = max(keyword_result.confidence, llm_result.confidence)
            reasoning = f"Both methods agree: {keyword_result.reasoning}; {llm_result.reasoning}"
            keywords = keyword_result.keywords
        else:
            # If they disagree, prefer LLM if it has high confidence
            if llm_result.confidence > self.confidence_thresholds["llm"]:
                final_intent = llm_result.intent
                confidence = llm_result.confidence
                reasoning = f"LLM preferred: {llm_result.reasoning}"
                keywords = []
            else:
                # Otherwise, use keyword result if it has reasonable confidence
                if keyword_result.confidence > self.confidence_thresholds["keyword"]:
                    final_intent = keyword_result.intent
                    confidence = keyword_result.confidence
                    reasoning = f"Keyword preferred: {keyword_result.reasoning}"
                    keywords = keyword_result.keywords
                else:
                    # Fallback to technical support
                    final_intent = "technical"
                    confidence = 0.2
                    reasoning = "Fallback to technical support"
                    keywords = []
        
        return IntentResult(
            intent=final_intent,
            confidence=confidence,
            keywords=keywords,
            reasoning=reasoning
        )
    
    def get_intent_description(self, intent: str) -> str:
        """Get human-readable description of intent"""
        return self.intents.get(intent, "Unknown")
    
    def get_processing_strategy(self, intent: str) -> Dict[str, str]:
        """Get processing strategy for each intent type"""
        strategies = {
            "technical": {
                "approach": "code_examples_and_documentation",
                "knowledge_base": "technical_docs",
                "response_style": "step_by_step",
                "priority": "high"
            },
            "billing": {
                "approach": "pricing_and_policies",
                "knowledge_base": "billing_info",
                "response_style": "clear_and_concise",
                "priority": "medium"
            },
            "feature": {
                "approach": "roadmap_and_comparison",
                "knowledge_base": "feature_requests",
                "response_style": "informative_and_encouraging",
                "priority": "low"
            }
        }
        
        return strategies.get(intent, strategies["technical"])
    
    def validate_intent(self, intent: str) -> bool:
        """Validate if intent is supported"""
        return intent in self.intents
    
    def get_all_intents(self) -> List[str]:
        """Get list of all supported intents"""
        return list(self.intents.keys())
    
    def get_intent_statistics(self, queries: List[str]) -> Dict[str, int]:
        """Get statistics of intent distribution for a list of queries"""
        stats = {intent: 0 for intent in self.intents}
        
        for query in queries:
            result = self.classify_intent(query)
            stats[result.intent] += 1
        
        return stats 