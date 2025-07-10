"""
Feature Request Processor
Handles feature requests with roadmap and comparison data
"""

import json
import logging
import os
from typing import Dict, List, Optional
from dataclasses import dataclass
from llm_wrapper import LLMWrapper

logger = logging.getLogger(__name__)

@dataclass
class FeatureResponse:
    """Feature request response with structured information"""
    response: str
    roadmap_info: Dict[str, any]
    alternatives: List[str]
    voting_info: Dict[str, str]
    contact_info: Dict[str, str]

class FeatureRequestProcessor:
    """Processor for feature request queries"""
    
    def __init__(self, llm_wrapper: LLMWrapper):
        self.llm = llm_wrapper
        
        # Load feature roadmap knowledge base
        self.knowledge_base = self._load_feature_kb()
        
        # Extract roadmap and other information from knowledge base
        self.roadmap = self._extract_roadmap()
        self.competitor_features = self._extract_competitor_features()
        self.voting_info = self._extract_voting_info()
        self.contact_info = self._extract_contact_info()
        
        # Common feature categories
        self.feature_categories = {
            "mobile": ["mobile app", "ios", "android", "sdk", "native"],
            "analytics": ["analytics", "dashboard", "reporting", "metrics", "insights"],
            "integration": ["webhook", "api", "third-party", "connector", "plugin"],
            "security": ["authentication", "encryption", "sso", "security", "compliance"],
            "ui_ux": ["interface", "design", "branding", "customization", "theme"]
        }
    
    def _load_feature_kb(self) -> Dict:
        """Load feature roadmap knowledge base from JSON file"""
        try:
            kb_path = os.path.join("data", "feature_roadmap.json")
            if os.path.exists(kb_path):
                with open(kb_path, 'r') as f:
                    kb_data = json.load(f)
                logger.info("Feature roadmap knowledge base loaded successfully")
                return kb_data
            else:
                logger.warning("Feature roadmap knowledge base file not found, using fallback data")
                return self._get_fallback_kb()
        except Exception as e:
            logger.error(f"Failed to load feature roadmap knowledge base: {e}")
            return self._get_fallback_kb()
    
    def _get_fallback_kb(self) -> Dict:
        """Fallback knowledge base when JSON file is not available"""
        return {
            "roadmap": {
                "current_quarter": {
                    "in_progress": [
                        {
                            "feature": "Dark Mode",
                            "description": "Complete dark theme for the web interface",
                            "progress": "75%",
                            "eta": "Q1 2024"
                        }
                    ]
                }
            },
            "feature_voting": {
                "current_top_votes": [
                    {
                        "feature": "Dark Mode",
                        "votes": 1247,
                        "status": "In Progress"
                    }
                ]
            }
        }
    
    def _extract_roadmap(self) -> Dict:
        """Extract roadmap information from knowledge base"""
        if "roadmap" in self.knowledge_base:
            return self.knowledge_base["roadmap"]
        return {}
    
    def _extract_competitor_features(self) -> Dict:
        """Extract competitor feature comparison from knowledge base"""
        if "competitor_analysis" in self.knowledge_base:
            competitor_analysis = self.knowledge_base["competitor_analysis"]
            if "competitors" in competitor_analysis:
                # Convert competitor analysis to the expected format
                features = {}
                for competitor_name, competitor_data in competitor_analysis["competitors"].items():
                    if "features_we_lack" in competitor_data:
                        for feature in competitor_data["features_we_lack"]:
                            features[feature.lower().replace(" ", "_")] = {
                                "our_platform": "Not available yet",
                                competitor_name: "Available",
                                "status": "planned"
                            }
                return features
        return {}
    
    def _extract_voting_info(self) -> Dict:
        """Extract voting information from knowledge base"""
        if "feature_voting" in self.knowledge_base:
            voting_data = self.knowledge_base["feature_voting"]
            return {
                "how_to_vote": voting_data.get("voting_process", {}).get("how_to_vote", "Visit our feature request portal"),
                "current_top_requests": [item["feature"] for item in voting_data.get("current_top_votes", [])],
                "voting_deadline": voting_data.get("voting_process", {}).get("review_frequency", "Votes are counted quarterly")
            }
        return {
            "how_to_vote": "Visit our feature request portal at https://example.com/features",
            "current_top_requests": ["Dark Mode", "Mobile SDK", "Advanced Analytics"],
            "voting_deadline": "Votes are counted quarterly for roadmap planning"
        }
    
    def _extract_contact_info(self) -> Dict:
        """Extract contact information from knowledge base"""
        if "competitor_analysis" in self.knowledge_base:
            # Use competitor analysis contact info if available
            return {
                "product_team": "product@example.com",
                "feature_portal": "https://example.com/features",
                "community_forum": "https://community.example.com",
                "sales_contact": "sales@example.com"
            }
        return {
            "product_team": "product@example.com",
            "feature_portal": "https://example.com/features",
            "community_forum": "https://community.example.com",
            "sales_contact": "sales@example.com"
        }
    
    def process_query(self, query: str, context: Optional[Dict] = None) -> FeatureResponse:
        """Process feature request query"""
        query_lower = query.lower()
        
        # Identify the type of feature request
        feature_type = self._identify_feature_type(query_lower)
        
        # Generate response using LLM
        llm_response = self._generate_llm_response(query, feature_type)
        
        # Get relevant roadmap and comparison information
        roadmap_info = self._get_roadmap_info(feature_type)
        alternatives = self._get_alternatives(feature_type)
        
        return FeatureResponse(
            response=llm_response,
            roadmap_info=roadmap_info,
            alternatives=alternatives,
            voting_info=self.voting_info,
            contact_info=self.contact_info
        )
    
    def _identify_feature_type(self, query: str) -> str:
        """Identify the type of feature being requested"""
        for category, keywords in self.feature_categories.items():
            if any(keyword in query for keyword in keywords):
                return category
        
        # Check for specific features
        if any(word in query for word in ["dark mode", "theme", "color"]):
            return "ui_ux"
        elif any(word in query for word in ["export", "import", "backup"]):
            return "integration"
        elif any(word in query for word in ["notification", "alert", "email"]):
            return "integration"
        else:
            return "general"
    
    def _generate_llm_response(self, query: str, feature_type: str) -> str:
        """Generate LLM response for feature request"""
        roadmap_context = self._get_roadmap_context(feature_type)
        competitor_context = self._get_competitor_context(feature_type)
        
        prompt = f"""You are a product manager for a SaaS platform handling feature requests.

Customer Query: "{query}"

Feature Type: {feature_type}

Roadmap Information:
{roadmap_context}

Competitor Comparison:
{competitor_context}

Provide a helpful, encouraging response that:
1. Acknowledges the feature request
2. Explains current status and timeline if applicable
3. Mentions alternatives or workarounds
4. Encourages voting on the feature request portal
5. Is positive and shows we value customer input

Keep the response friendly and informative."""

        try:
            response = self.llm.generate(prompt)
            if response and response.success:
                return response.content
            else:
                return self._get_fallback_response(feature_type)
        except Exception as e:
            logger.error(f"LLM response generation failed: {e}")
            return self._get_fallback_response(feature_type)
    
    def _get_roadmap_context(self, feature_type: str) -> str:
        """Get roadmap context for LLM from knowledge base"""
        context = "Current Roadmap:\n"
        
        if "roadmap" in self.knowledge_base:
            roadmap = self.knowledge_base["roadmap"]
            
            # Check current quarter
            if "current_quarter" in roadmap:
                current = roadmap["current_quarter"]
                
                # In-progress features
                if "in_progress" in current:
                    relevant_features = []
                    for feature in current["in_progress"]:
                        if (feature_type in feature["feature"].lower() or 
                            feature_type in feature.get("description", "").lower()):
                            relevant_features.append(f"{feature['feature']} ({feature.get('progress', 'Unknown')}%)")
                    
                    if relevant_features:
                        context += f"- Current Quarter (In Progress): {', '.join(relevant_features)}\n"
                
                # Planned features
                if "planned" in current:
                    relevant_features = []
                    for feature in current["planned"]:
                        if (feature_type in feature["feature"].lower() or 
                            feature_type in feature.get("description", "").lower()):
                            relevant_features.append(f"{feature['feature']} (ETA: {feature.get('eta', 'Unknown')})")
                    
                    if relevant_features:
                        context += f"- Current Quarter (Planned): {', '.join(relevant_features)}\n"
            
            # Check next quarter
            if "next_quarter" in roadmap:
                next_q = roadmap["next_quarter"]
                if "planned" in next_q:
                    relevant_features = []
                    for feature in next_q["planned"]:
                        if (feature_type in feature["feature"].lower() or 
                            feature_type in feature.get("description", "").lower()):
                            relevant_features.append(f"{feature['feature']} (ETA: {feature.get('eta', 'Unknown')})")
                    
                    if relevant_features:
                        context += f"- Next Quarter (Planned): {', '.join(relevant_features)}\n"
        
        return context
    
    def _get_competitor_context(self, feature_type: str) -> str:
        """Get competitor context for LLM from knowledge base"""
        context = "Feature Comparison:\n"
        
        if "competitor_analysis" in self.knowledge_base:
            competitor_analysis = self.knowledge_base["competitor_analysis"]
            
            if "competitors" in competitor_analysis:
                for competitor_name, competitor_data in competitor_analysis["competitors"].items():
                    # Check features we lack
                    if "features_we_lack" in competitor_data:
                        relevant_features = []
                        for feature in competitor_data["features_we_lack"]:
                            if feature_type in feature.lower():
                                relevant_features.append(feature)
                        
                        if relevant_features:
                            context += f"- {competitor_name}: Has {', '.join(relevant_features)}\n"
            
            # Add our advantages
            if "market_positioning" in competitor_analysis and "our_advantages" in competitor_analysis["market_positioning"]:
                advantages = competitor_analysis["market_positioning"]["our_advantages"]
                relevant_advantages = [adv for adv in advantages if feature_type in adv.lower()]
                if relevant_advantages:
                    context += f"- Our advantages: {', '.join(relevant_advantages[:2])}\n"
        
        return context
    
    def _get_fallback_response(self, feature_type: str) -> str:
        """Get fallback response when LLM fails"""
        fallback_responses = {
            "mobile": "Thank you for requesting mobile SDK features! We're actively working on mobile support and it's planned for Q4 2024. In the meantime, our web API works great on mobile browsers. Please vote for mobile features on our portal to help prioritize development.",
            "analytics": "We appreciate your interest in advanced analytics! We're planning to enhance our analytics capabilities in Q2 2024. Currently, we offer basic analytics in our Pro plan. Please vote for analytics features to help us understand your specific needs.",
            "integration": "Thanks for the integration request! We're continuously adding new integrations and webhook capabilities. Check our roadmap for upcoming features, and please vote for specific integrations you need.",
            "security": "Security is a top priority for us. We're planning several security enhancements in Q3 2024. Please vote for security features and let us know your specific requirements.",
            "ui_ux": "We're always working to improve our user interface and experience. Custom branding options are planned for Q3 2024. Please vote for UI/UX features to help us prioritize improvements.",
            "general": "Thank you for your feature request! We value all customer feedback and use it to prioritize our development roadmap. Please visit our feature request portal to vote for this feature and see our current roadmap."
        }
        
        return fallback_responses.get(feature_type, fallback_responses["general"])
    
    def _get_roadmap_info(self, feature_type: str) -> Dict[str, any]:
        """Get relevant roadmap information from knowledge base"""
        relevant_roadmap = {}
        
        if "roadmap" in self.knowledge_base:
            roadmap = self.knowledge_base["roadmap"]
            
            # Check current quarter
            if "current_quarter" in roadmap:
                current = roadmap["current_quarter"]
                relevant_features = []
                
                # In-progress features
                if "in_progress" in current:
                    for feature in current["in_progress"]:
                        if (feature_type in feature["feature"].lower() or 
                            feature_type in feature.get("description", "").lower()):
                            relevant_features.append({
                                "name": feature["feature"],
                                "status": "In Progress",
                                "progress": feature.get("progress", "Unknown"),
                                "eta": feature.get("eta", "Unknown")
                            })
                
                # Planned features
                if "planned" in current:
                    for feature in current["planned"]:
                        if (feature_type in feature["feature"].lower() or 
                            feature_type in feature.get("description", "").lower()):
                            relevant_features.append({
                                "name": feature["feature"],
                                "status": "Planned",
                                "eta": feature.get("eta", "Unknown")
                            })
                
                if relevant_features:
                    relevant_roadmap["current_quarter"] = {
                        "title": "Current Quarter",
                        "features": relevant_features
                    }
            
            # Check next quarter
            if "next_quarter" in roadmap:
                next_q = roadmap["next_quarter"]
                relevant_features = []
                
                if "planned" in next_q:
                    for feature in next_q["planned"]:
                        if (feature_type in feature["feature"].lower() or 
                            feature_type in feature.get("description", "").lower()):
                            relevant_features.append({
                                "name": feature["feature"],
                                "status": "Planned",
                                "eta": feature.get("eta", "Unknown")
                            })
                
                if relevant_features:
                    relevant_roadmap["next_quarter"] = {
                        "title": "Next Quarter",
                        "features": relevant_features
                    }
        
        return relevant_roadmap
    
    def _get_alternatives(self, feature_type: str) -> List[str]:
        """Get alternative solutions or workarounds"""
        alternatives = {
            "mobile": [
                "Use our responsive web API in mobile browsers",
                "Integrate with our REST API in native apps",
                "Use third-party mobile SDK wrappers"
            ],
            "analytics": [
                "Export data and analyze in external tools",
                "Use our basic analytics in Pro plan",
                "Set up custom tracking with webhooks"
            ],
            "integration": [
                "Use our webhook system for real-time updates",
                "Set up scheduled API calls for data sync",
                "Use our REST API for custom integrations"
            ],
            "security": [
                "Use API key authentication",
                "Implement IP whitelisting",
                "Set up audit logging"
            ],
            "ui_ux": [
                "Use our current interface with custom CSS",
                "Integrate our API into your own UI",
                "Use our webhook system for custom notifications"
            ]
        }
        
        return alternatives.get(feature_type, ["Check our documentation for current capabilities", "Contact our sales team for custom solutions"])
    
    def get_feature_status(self, feature_name: str) -> Dict[str, str]:
        """Get status of a specific feature"""
        feature_lower = feature_name.lower()
        
        # Check roadmap
        for quarter, data in self.roadmap.items():
            for feature in data["features"]:
                if feature_lower in feature.lower():
                    return {
                        "status": data["status"],
                        "timeline": data["completion"],
                        "quarter": quarter
                    }
        
        # Check competitor comparison
        for feature, comparison in self.competitor_features.items():
            if feature_lower in feature.lower():
                return {
                    "status": comparison["status"],
                    "our_platform": comparison["our_platform"],
                    "competitors": {k: v for k, v in comparison.items() if k.startswith("competitor")}
                }
        
        return {"status": "not_planned", "message": "Feature not currently planned"}
    
    def get_roadmap_summary(self) -> Dict[str, any]:
        """Get summary of current roadmap"""
        summary = {
            "quarters": {},
            "total_features": 0,
            "in_progress": 0,
            "planned": 0
        }
        
        for quarter, data in self.roadmap.items():
            summary["quarters"][quarter] = {
                "title": data["title"],
                "feature_count": len(data["features"]),
                "status": data["status"],
                "completion": data["completion"]
            }
            summary["total_features"] += len(data["features"])
            
            if data["status"] == "in_progress":
                summary["in_progress"] += len(data["features"])
            elif data["status"] == "planned":
                summary["planned"] += len(data["features"])
        
        return summary
    
    def format_response(self, response: FeatureResponse) -> str:
        """Format feature request response for display"""
        formatted = f"{response.response}\n\n"
        
        if response.roadmap_info:
            formatted += "**Roadmap Information:**\n"
            for quarter, data in response.roadmap_info.items():
                formatted += f"- {data['title']} ({data['status']}): {', '.join(data['features'])}\n"
            formatted += "\n"
        
        if response.alternatives:
            formatted += "**Current Alternatives:**\n"
            for i, alternative in enumerate(response.alternatives, 1):
                formatted += f"{i}. {alternative}\n"
            formatted += "\n"
        
        formatted += "**Feature Voting:**\n"
        formatted += f"- Portal: {response.voting_info['how_to_vote']}\n"
        formatted += f"- Current top requests: {', '.join(response.voting_info['current_top_requests'][:3])}\n"
        formatted += f"- Voting deadline: {response.voting_info['voting_deadline']}\n\n"
        
        formatted += "**Contact Information:**\n"
        formatted += f"- Product Team: {response.contact_info['product_team']}\n"
        formatted += f"- Feature Portal: {response.contact_info['feature_portal']}\n"
        formatted += f"- Community Forum: {response.contact_info['community_forum']}\n"
        
        return formatted 