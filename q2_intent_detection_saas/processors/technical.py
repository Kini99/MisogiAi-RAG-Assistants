"""
Technical Support Processor
Handles technical queries with code examples and documentation
"""

import json
import logging
import os
from typing import Dict, List, Optional
from dataclasses import dataclass
from llm_wrapper import LLMWrapper

logger = logging.getLogger(__name__)

@dataclass
class TechnicalResponse:
    """Technical support response with structured information"""
    solution: str
    code_examples: List[str]
    documentation_links: List[str]
    troubleshooting_steps: List[str]
    related_topics: List[str]

class TechnicalProcessor:
    """Processor for technical support queries"""
    
    def __init__(self, llm_wrapper: LLMWrapper):
        self.llm = llm_wrapper
        
        # Load technical knowledge base
        self.knowledge_base = self._load_technical_kb()
        
        # Common technical issues and solutions
        self.common_issues = {
            "404": "Endpoint not found - check URL and API version",
            "401": "Unauthorized - verify API key and permissions",
            "403": "Forbidden - check account status and plan limits",
            "429": "Rate limited - implement exponential backoff",
            "500": "Server error - retry request or contact support",
            "timeout": "Request timeout - check network and increase timeout",
            "connection": "Connection failed - verify network and firewall settings"
        }
    
    def _load_technical_kb(self) -> Dict:
        """Load technical knowledge base from JSON file"""
        try:
            kb_path = os.path.join("data", "technical_kb.json")
            if os.path.exists(kb_path):
                with open(kb_path, 'r') as f:
                    kb_data = json.load(f)
                logger.info("Technical knowledge base loaded successfully")
                return kb_data
            else:
                logger.warning("Technical knowledge base file not found, using fallback data")
                return self._get_fallback_kb()
        except Exception as e:
            logger.error(f"Failed to load technical knowledge base: {e}")
            return self._get_fallback_kb()
    
    def _get_fallback_kb(self) -> Dict:
        """Fallback knowledge base when JSON file is not available"""
        return {
            "api_documentation": {
                "authentication": {
                    "methods": ["API Key", "OAuth 2.0", "JWT"],
                    "examples": {
                        "api_key": {
                            "curl": "curl -H 'Authorization: Bearer YOUR_API_KEY' https://api.example.com/v1/data",
                            "python": "import requests\n\nheaders = {'Authorization': 'Bearer YOUR_API_KEY'}\nresponse = requests.get('https://api.example.com/v1/data', headers=headers)",
                            "javascript": "const response = await fetch('https://api.example.com/v1/data', {\n  headers: {\n    'Authorization': 'Bearer YOUR_API_KEY'\n  }\n});"
                        }
                    }
                }
            },
            "troubleshooting": {
                "common_errors": {
                    "401_unauthorized": {
                        "description": "Authentication failed",
                        "causes": ["Invalid API key", "Expired token", "Missing authorization header"],
                        "solutions": [
                            "Check your API key is correct",
                            "Refresh your authentication token",
                            "Ensure Authorization header is present"
                        ]
                    }
                }
            }
        }
    
    def process_query(self, query: str, context: Optional[Dict] = None) -> TechnicalResponse:
        """Process technical support query"""
        query_lower = query.lower()
        
        # Identify the type of technical issue
        issue_type = self._identify_issue_type(query_lower)
        
        # Generate response using LLM
        llm_response = self._generate_llm_response(query, issue_type)
        
        # Get relevant examples and documentation
        examples = self._get_relevant_examples(issue_type)
        docs = self._get_relevant_docs(issue_type)
        troubleshooting = self._get_troubleshooting_steps(issue_type)
        related = self._get_related_topics(issue_type)
        
        return TechnicalResponse(
            solution=llm_response,
            code_examples=examples,
            documentation_links=docs,
            troubleshooting_steps=troubleshooting,
            related_topics=related
        )
    
    def _identify_issue_type(self, query: str) -> str:
        """Identify the type of technical issue"""
        if any(word in query for word in ["api", "endpoint", "request", "call"]):
            return "api_integration"
        elif any(word in query for word in ["auth", "login", "key", "token", "permission"]):
            return "authentication"
        elif any(word in query for word in ["error", "exception", "fail", "crash", "bug"]):
            return "error_handling"
        elif any(word in query for word in ["install", "setup", "configure", "deploy"]):
            return "setup_installation"
        else:
            return "api_integration"  # Default
    
    def _generate_llm_response(self, query: str, issue_type: str) -> str:
        """Generate LLM response for technical query"""
        prompt = f"""You are a technical support specialist for a SaaS API platform. 

Customer Query: "{query}"

Issue Type: {issue_type}

Provide a helpful, step-by-step solution that includes:
1. Clear explanation of the issue
2. Step-by-step resolution
3. Code examples if applicable
4. Best practices to avoid this issue

Keep the response concise but comprehensive. Focus on practical solutions."""

        try:
            response = self.llm.generate(prompt)
            if response and response.success:
                return response.content
            else:
                return self._get_fallback_response(issue_type)
        except Exception as e:
            logger.error(f"LLM response generation failed: {e}")
            return self._get_fallback_response(issue_type)
    
    def _get_fallback_response(self, issue_type: str) -> str:
        """Get fallback response when LLM fails"""
        fallback_responses = {
            "api_integration": "I can help you with API integration. Please check our documentation for step-by-step guides and code examples. If you're still having issues, please provide more details about your specific use case.",
            "authentication": "For authentication issues, please verify your API key is valid and has the correct permissions. Check our authentication documentation for detailed setup instructions.",
            "error_handling": "To resolve this error, please check the error code and message. Our error handling documentation provides specific guidance for common issues.",
            "setup_installation": "For setup and installation help, please follow our installation guide. Make sure you have the required dependencies and correct versions installed."
        }
        
        return fallback_responses.get(issue_type, "I can help you with this technical issue. Please check our documentation or provide more specific details.")
    
    def _get_relevant_examples(self, issue_type: str) -> List[str]:
        """Get relevant code examples from knowledge base"""
        examples = []
        
        if "api_documentation" in self.knowledge_base:
            api_docs = self.knowledge_base["api_documentation"]
            
            if issue_type == "authentication" and "authentication" in api_docs:
                auth_examples = api_docs["authentication"].get("examples", {})
                for lang, code in auth_examples.items():
                    if lang == "python":
                        examples.append(f"```python\n{code}\n```")
                    elif lang == "javascript":
                        examples.append(f"```javascript\n{code}\n```")
                    elif lang == "curl":
                        examples.append(f"```bash\n{code}\n```")
            
            elif issue_type == "api_integration" and "endpoints" in api_docs:
                for endpoint_name, endpoint_data in api_docs["endpoints"].items():
                    if "examples" in endpoint_data:
                        for example_name, example_code in endpoint_data["examples"].items():
                            examples.append(f"```\n{example_code}\n```")
        
        return examples
    
    def _get_relevant_docs(self, issue_type: str) -> List[str]:
        """Get relevant documentation links from knowledge base"""
        docs = []
        
        if "api_documentation" in self.knowledge_base:
            api_docs = self.knowledge_base["api_documentation"]
            
            if issue_type == "authentication" and "authentication" in api_docs:
                # Add authentication docs
                docs.extend([
                    "https://docs.example.com/auth/overview",
                    "https://docs.example.com/auth/tokens"
                ])
            
            elif issue_type == "api_integration":
                # Add general API docs
                docs.extend([
                    "https://docs.example.com/api/getting-started",
                    "https://docs.example.com/api/endpoints"
                ])
        
        return docs
    
    def _get_troubleshooting_steps(self, issue_type: str) -> List[str]:
        """Get troubleshooting steps from knowledge base"""
        steps = []
        
        if "troubleshooting" in self.knowledge_base:
            troubleshooting = self.knowledge_base["troubleshooting"]
            
            if "common_errors" in troubleshooting:
                for error_code, error_info in troubleshooting["common_errors"].items():
                    if issue_type in error_code or issue_type in error_info.get("description", "").lower():
                        steps.extend(error_info.get("solutions", []))
        
        return steps
    
    def _get_related_topics(self, issue_type: str) -> List[str]:
        """Get related topics"""
        related_topics = {
            "api_integration": ["Rate Limiting", "Webhooks", "SDK Usage"],
            "authentication": ["API Keys", "OAuth", "Permissions"],
            "error_handling": ["Status Codes", "Logging", "Monitoring"],
            "setup_installation": ["Environment Setup", "Configuration", "Testing"]
        }
        
        return related_topics.get(issue_type, [])
    
    def get_common_solutions(self, error_code: str) -> Optional[str]:
        """Get common solution for error codes"""
        return self.common_issues.get(error_code.lower())
    
    def format_response(self, response: TechnicalResponse, include_extras: bool = True) -> str:
        """Format technical response for display"""
        formatted = f"{response.solution}\n\n"
        if include_extras:
            if response.code_examples:
                formatted += "**Code Examples:**\n"
                for example in response.code_examples:
                    formatted += f"{example}\n\n"
            if response.troubleshooting_steps:
                formatted += "**Troubleshooting Steps:**\n"
                for i, step in enumerate(response.troubleshooting_steps, 1):
                    formatted += f"{i}. {step}\n"
                formatted += "\n"
            if response.documentation_links:
                formatted += "**Documentation:**\n"
                for link in response.documentation_links:
                    formatted += f"- {link}\n"
                formatted += "\n"
            if response.related_topics:
                formatted += "**Related Topics:**\n"
                for topic in response.related_topics:
                    formatted += f"- {topic}\n"
        return formatted 