"""
Billing/Account Processor
Handles billing queries with pricing tables and policies
"""

import json
import logging
import os
from typing import Dict, List, Optional
from dataclasses import dataclass
from llm_wrapper import LLMWrapper

logger = logging.getLogger(__name__)

@dataclass
class BillingResponse:
    """Billing response with structured information"""
    answer: str
    pricing_info: Dict[str, any]
    policy_links: List[str]
    next_steps: List[str]
    contact_info: Dict[str, str]

class BillingProcessor:
    """Processor for billing and account queries"""
    
    def __init__(self, llm_wrapper: LLMWrapper):
        self.llm = llm_wrapper
        
        # Load billing knowledge base
        self.knowledge_base = self._load_billing_kb()
        
        # Extract pricing and policy information from knowledge base
        self.pricing_plans = self._extract_pricing_plans()
        self.policies = self._extract_policies()
        self.contact_info = self._extract_contact_info()
    
    def _load_billing_kb(self) -> Dict:
        """Load billing knowledge base from JSON file"""
        try:
            kb_path = os.path.join("data", "billing_kb.json")
            if os.path.exists(kb_path):
                with open(kb_path, 'r') as f:
                    kb_data = json.load(f)
                logger.info("Billing knowledge base loaded successfully")
                return kb_data
            else:
                logger.warning("Billing knowledge base file not found, using fallback data")
                return self._get_fallback_kb()
        except Exception as e:
            logger.error(f"Failed to load billing knowledge base: {e}")
            return self._get_fallback_kb()
    
    def _get_fallback_kb(self) -> Dict:
        """Fallback knowledge base when JSON file is not available"""
        return {
            "pricing_plans": {
                "basic": {
                    "price": "$29/month",
                    "features": ["Up to 1,000 API calls/month", "Basic support"],
                    "limits": {"api_calls": 1000, "users": 5}
                }
            },
            "billing_policies": {
                "cancellation": {
                    "process": "Cancel anytime from account settings",
                    "effective_date": "End of current billing period"
                }
            },
            "support_contacts": {
                "billing_support": {
                    "email": "billing@example.com",
                    "phone": "+1-555-0123"
                }
            }
        }
    
    def _extract_pricing_plans(self) -> Dict:
        """Extract pricing plans from knowledge base"""
        if "pricing_plans" in self.knowledge_base:
            return self.knowledge_base["pricing_plans"]
        return {}
    
    def _extract_policies(self) -> Dict:
        """Extract policies from knowledge base"""
        policies = {}
        if "billing_policies" in self.knowledge_base:
            billing_policies = self.knowledge_base["billing_policies"]
            
            # Convert billing policies to the expected format
            if "cancellation" in billing_policies:
                policies["cancellation"] = {
                    "title": "Cancellation Policy",
                    "description": billing_policies["cancellation"].get("process", ""),
                    "link": "https://example.com/policies/cancellation",
                    "steps": ["Log into your account dashboard", "Go to Billing & Subscription", "Click 'Cancel Subscription'"]
                }
            
            if "refund_policy" in billing_policies:
                refund_info = billing_policies["refund_policy"]
                policies["refund"] = {
                    "title": "Refund Policy",
                    "description": f"{refund_info.get('paid_plans', '30-day money-back guarantee')}",
                    "link": "https://example.com/policies/refund",
                    "steps": ["Contact support within 30 days", "Provide reason for refund"]
                }
        
        return policies
    
    def _extract_contact_info(self) -> Dict:
        """Extract contact information from knowledge base"""
        if "support_contacts" in self.knowledge_base:
            contacts = self.knowledge_base["support_contacts"]
            return {
                "billing_support": contacts.get("billing_support", {}).get("email", "billing@example.com"),
                "general_support": contacts.get("general_support", {}).get("email", "support@example.com"),
                "sales": "sales@example.com",
                "phone": contacts.get("billing_support", {}).get("phone", "+1-555-0123"),
                "hours": contacts.get("billing_support", {}).get("hours", "Monday-Friday, 9AM-6PM EST")
            }
        return {
            "billing_support": "billing@example.com",
            "general_support": "support@example.com",
            "sales": "sales@example.com",
            "phone": "+1-555-0123",
            "hours": "Monday-Friday, 9AM-6PM EST"
        }
    
    def process_query(self, query: str, context: Optional[Dict] = None) -> BillingResponse:
        """Process billing/account query"""
        query_lower = query.lower()
        
        # Identify the type of billing query
        query_type = self._identify_query_type(query_lower)
        
        # Generate response using LLM
        llm_response = self._generate_llm_response(query, query_type)
        
        # Get relevant pricing and policy information
        pricing_info = self._get_pricing_info(query_type)
        policy_links = self._get_policy_links(query_type)
        next_steps = self._get_next_steps(query_type)
        
        return BillingResponse(
            answer=llm_response,
            pricing_info=pricing_info,
            policy_links=policy_links,
            next_steps=next_steps,
            contact_info=self.contact_info
        )
    
    def _identify_query_type(self, query: str) -> str:
        """Identify the type of billing query"""
        if any(word in query for word in ["price", "cost", "plan", "tier", "subscription"]):
            return "pricing"
        elif any(word in query for word in ["cancel", "cancellation", "stop", "end"]):
            return "cancellation"
        elif any(word in query for word in ["refund", "money back", "return"]):
            return "refund"
        elif any(word in query for word in ["upgrade", "downgrade", "change plan"]):
            return "plan_change"
        elif any(word in query for word in ["billing", "invoice", "payment", "charge"]):
            return "billing"
        else:
            return "general"
    
    def _generate_llm_response(self, query: str, query_type: str) -> str:
        """Generate LLM response for billing query"""
        pricing_context = self._get_pricing_context()
        
        prompt = f"""You are a billing support specialist for a SaaS platform.

Customer Query: "{query}"

Query Type: {query_type}

Pricing Information:
{pricing_context}

Provide a helpful, clear response that:
1. Directly addresses the customer's question
2. Includes relevant pricing information if applicable
3. Explains any policies or procedures
4. Provides clear next steps
5. Is professional and customer-friendly

Keep the response concise but informative."""

        try:
            response = self.llm.generate(prompt)
            if response and response.success:
                return response.content
            else:
                return self._get_fallback_response(query_type)
        except Exception as e:
            logger.error(f"LLM response generation failed: {e}")
            return self._get_fallback_response(query_type)
    
    def _get_pricing_context(self) -> str:
        """Get pricing context for LLM"""
        context = "Available Plans:\n"
        for plan_id, plan in self.pricing_plans.items():
            context += f"- {plan['name']}: {plan['price']} ({plan['annual_price']})\n"
            context += f"  Features: {', '.join(plan['features'][:3])}\n"
        return context
    
    def _get_fallback_response(self, query_type: str) -> str:
        """Get fallback response when LLM fails"""
        fallback_responses = {
            "pricing": "I can help you with pricing information. We offer Basic ($29/month), Pro ($99/month), and Enterprise (custom pricing) plans. Each plan includes different features and usage limits. Would you like me to provide more details about a specific plan?",
            "cancellation": "You can cancel your subscription at any time through your account dashboard. No refunds are provided for partial months. The cancellation will take effect at the end of your current billing period.",
            "refund": "We offer a 30-day money-back guarantee for new subscriptions. If you're within this period, please contact our billing support team with your request.",
            "plan_change": "You can upgrade your plan anytime and only pay the prorated difference. Downgrades take effect at the end of your current billing cycle. Both can be done through your account dashboard.",
            "billing": "For billing questions, please check your account dashboard for current invoices and payment history. If you need further assistance, our billing support team is available to help.",
            "general": "I can help you with billing and account questions. Please let me know what specific information you need about pricing, plans, or account management."
        }
        
        return fallback_responses.get(query_type, "I can help you with billing and account questions. Please provide more specific details about what you need assistance with.")
    
    def _get_pricing_info(self, query_type: str) -> Dict[str, any]:
        """Get relevant pricing information"""
        if query_type == "pricing":
            return self.pricing_plans
        elif query_type in ["upgrade", "downgrade", "plan_change"]:
            return {k: v for k, v in self.pricing_plans.items() if k in ["pro", "enterprise"]}
        else:
            return {}
    
    def _get_policy_links(self, query_type: str) -> List[str]:
        """Get relevant policy links"""
        if query_type == "cancellation":
            return [self.policies["cancellation"]["link"]]
        elif query_type == "refund":
            return [self.policies["refund"]["link"]]
        elif query_type in ["upgrade", "downgrade", "plan_change"]:
            return [self.policies["upgrade"]["link"], self.policies["downgrade"]["link"]]
        else:
            return []
    
    def _get_next_steps(self, query_type: str) -> List[str]:
        """Get next steps for the customer"""
        if query_type == "cancellation":
            return self.policies["cancellation"]["steps"]
        elif query_type == "refund":
            return self.policies["refund"]["steps"]
        elif query_type == "upgrade":
            return self.policies["upgrade"]["steps"]
        elif query_type == "downgrade":
            return self.policies["downgrade"]["steps"]
        else:
            return ["Log into your account dashboard", "Contact support if you need further assistance"]
    
    def get_plan_comparison(self) -> Dict[str, any]:
        """Get plan comparison table"""
        comparison = {
            "plans": list(self.pricing_plans.keys()),
            "prices": {plan: data["price"] for plan, data in self.pricing_plans.items()},
            "annual_prices": {plan: data["annual_price"] for plan, data in self.pricing_plans.items()},
            "features": {plan: data["features"] for plan, data in self.pricing_plans.items()},
            "limits": {plan: data["limits"] for plan, data in self.pricing_plans.items()}
        }
        return comparison
    
    def calculate_cost(self, plan: str, months: int = 1) -> Dict[str, any]:
        """Calculate cost for a plan over specified months"""
        if plan not in self.pricing_plans:
            return {"error": "Plan not found"}
        
        plan_data = self.pricing_plans[plan]
        monthly_price = float(plan_data["price"].replace("$", "").replace("/month", ""))
        
        if months == 12:
            annual_price = plan_data["annual_price"]
            if "Contact sales" in annual_price:
                total = "Contact sales for annual pricing"
            else:
                total = annual_price
        else:
            total = f"${monthly_price * months:.2f}"
        
        return {
            "plan": plan_data["name"],
            "monthly_cost": plan_data["price"],
            "total_cost": total,
            "months": months
        }
    
    def format_response(self, response: BillingResponse) -> str:
        """Format billing response for display"""
        formatted = f"{response.answer}\n\n"
        
        if response.pricing_info:
            formatted += "**Pricing Information:**\n"
            for plan_id, plan_data in response.pricing_info.items():
                formatted += f"- {plan_data['name']}: {plan_data['price']}\n"
            formatted += "\n"
        
        if response.next_steps:
            formatted += "**Next Steps:**\n"
            for i, step in enumerate(response.next_steps, 1):
                formatted += f"{i}. {step}\n"
            formatted += "\n"
        
        if response.policy_links:
            formatted += "**Related Policies:**\n"
            for link in response.policy_links:
                formatted += f"- {link}\n"
            formatted += "\n"
        
        formatted += "**Contact Information:**\n"
        formatted += f"- Billing Support: {response.contact_info['billing_support']}\n"
        formatted += f"- Phone: {response.contact_info['phone']}\n"
        formatted += f"- Hours: {response.contact_info['hours']}\n"
        
        return formatted 