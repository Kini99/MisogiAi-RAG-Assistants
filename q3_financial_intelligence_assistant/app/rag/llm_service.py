import time
from typing import List, Dict, Any, Optional
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
from app.core.config import settings
import structlog

logger = structlog.get_logger()


class LLMService:
    """LLM service for financial query processing"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            openai_api_key=settings.openai_api_key,
            model_name=settings.openai_model,
            temperature=0.1,
            max_tokens=2000
        )
        
        # Financial analysis system prompt
        self.system_prompt = """You are a financial intelligence assistant specialized in analyzing corporate financial reports, earnings data, and financial statements. 

Your expertise includes:
- Financial metrics analysis (P/E ratios, ROE, debt-to-equity, etc.)
- Company performance comparisons
- Trend analysis and forecasting
- Risk assessment
- Industry benchmarking

When analyzing financial data:
1. Provide accurate, data-driven insights
2. Include relevant financial metrics and ratios
3. Compare with industry standards when possible
4. Highlight key trends and patterns
5. Identify potential risks and opportunities
6. Use clear, professional language suitable for financial professionals

Always base your analysis on the provided financial documents and data. If information is not available in the documents, clearly state that."""
        
        # Query-specific prompts
        self.prompts = {
            "financial_metrics": """Analyze the financial metrics for {company_name} based on the following documents:

Context: {context}

Query: {query}

Please provide:
1. Key financial metrics and their values
2. Analysis of the company's financial health
3. Comparison with industry benchmarks if available
4. Key insights and recommendations""",
            
            "comparison": """Compare {company_name} with other companies in the same industry based on the following documents:

Context: {context}

Query: {query}

Please provide:
1. Comparative analysis of key metrics
2. Relative performance assessment
3. Competitive advantages/disadvantages
4. Market positioning analysis""",
            
            "trend_analysis": """Analyze the financial trends for {company_name} over time based on the following documents:

Context: {context}

Query: {query}

Please provide:
1. Historical trend analysis
2. Key drivers of performance changes
3. Future outlook and projections
4. Risk factors and opportunities""",
            
            "general": """Analyze the financial information for {company_name} based on the following documents:

Context: {context}

Query: {query}

Please provide a comprehensive analysis addressing the specific query."""
        }
    
    async def generate_response(self, query: str, context: List[Dict[str, Any]], 
                               company_name: str = "the company",
                               query_type: str = "general") -> Dict[str, Any]:
        """Generate response using LLM"""
        start_time = time.time()
        
        try:
            # Prepare context
            context_text = self._prepare_context(context)
            
            # Get appropriate prompt template
            prompt_template = self.prompts.get(query_type, self.prompts["general"])
            
            # Create messages
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=prompt_template.format(
                    company_name=company_name,
                    context=context_text,
                    query=query
                ))
            ]
            
            # Generate response
            response = await self.llm.agenerate([messages])
            
            # Extract response content
            response_content = response.generations[0][0].text
            
            # Calculate metrics
            end_time = time.time()
            response_time = end_time - start_time
            
            # Estimate tokens (rough calculation)
            input_tokens = len(context_text + query) // 4  # Rough estimate
            output_tokens = len(response_content) // 4
            total_tokens = input_tokens + output_tokens
            
            # Estimate cost (GPT-4 pricing)
            cost = self._estimate_cost(total_tokens)
            
            logger.info("LLM response generated", 
                       query_type=query_type,
                       response_time=response_time,
                       tokens=total_tokens,
                       cost=cost)
            
            return {
                "response": response_content,
                "response_time": response_time,
                "tokens_used": total_tokens,
                "cost": cost,
                "query_type": query_type,
                "context_sources": len(context)
            }
            
        except Exception as e:
            logger.error("LLM response generation failed", error=str(e), query=query)
            raise
    
    def _prepare_context(self, context: List[Dict[str, Any]]) -> str:
        """Prepare context from retrieved documents"""
        context_parts = []
        
        for i, doc in enumerate(context, 1):
            content = doc.get("content", "")
            metadata = doc.get("meta", {})
            score = doc.get("score", 0)
            
            # Add source information
            source_info = f"Source {i}: "
            if metadata.get("document_title"):
                source_info += f"{metadata['document_title']} "
            if metadata.get("page"):
                source_info += f"(Page {metadata['page']}) "
            source_info += f"[Relevance: {score:.3f}]"
            
            context_parts.append(f"{source_info}\n{content}\n")
        
        return "\n".join(context_parts)
    
    def _estimate_cost(self, total_tokens: int) -> float:
        """Estimate cost based on token usage"""
        # GPT-4 pricing (approximate)
        input_cost_per_1k = 0.03
        output_cost_per_1k = 0.06
        
        # Rough estimate: 70% input, 30% output
        input_tokens = int(total_tokens * 0.7)
        output_tokens = total_tokens - input_tokens
        
        input_cost = (input_tokens / 1000) * input_cost_per_1k
        output_cost = (output_tokens / 1000) * output_cost_per_1k
        
        return round(input_cost + output_cost, 4)
    
    async def classify_query(self, query: str) -> str:
        """Classify query type for better prompt selection"""
        try:
            classification_prompt = f"""Classify the following financial query into one of these categories:
- financial_metrics: Queries about specific financial ratios, metrics, or performance indicators
- comparison: Queries comparing companies, periods, or benchmarks
- trend_analysis: Queries about trends, growth, or historical patterns
- general: General financial analysis queries

Query: {query}

Respond with only the category name."""

            messages = [HumanMessage(content=classification_prompt)]
            response = await self.llm.agenerate([messages])
            
            classification = response.generations[0][0].text.strip().lower()
            
            # Validate classification
            valid_types = ["financial_metrics", "comparison", "trend_analysis", "general"]
            if classification not in valid_types:
                classification = "general"
            
            logger.info("Query classified", query=query, classification=classification)
            return classification
            
        except Exception as e:
            logger.error("Query classification failed", error=str(e))
            return "general"


# Global LLM service instance
llm_service = LLMService() 