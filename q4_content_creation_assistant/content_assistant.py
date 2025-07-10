import os
import json
import datetime
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
import re

class ContentAssistant:
    def __init__(self):
        # Initialize OpenAI client
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Setup workspace paths
        self.workspace_path = Path("content-workspace")
        self.ideas_path = self.workspace_path / "ideas"
        self.generated_path = self.workspace_path / "generated"
        self.published_path = self.workspace_path / "published"
        self.templates_path = self.workspace_path / "templates"
    
    def capture_idea(self, message):
        """Capture and structure an idea from user input"""
        try:
            # Extract topic from message
            topic = self.extract_topic_from_message(message)
            
            # Generate structured idea using AI
            idea_content = self.generate_idea_structure(topic)
            
            # Create filename with date and sanitized title
            date_str = datetime.datetime.now().strftime("%Y-%m-%d")
            safe_title = self.sanitize_filename(topic)
            filename = f"{date_str}-{safe_title}.md"
            filepath = self.ideas_path / filename
            
            # Save idea to file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(idea_content)
            
            return f"Idea captured successfully! Saved as '{filename}' in the ideas folder. You can now generate an article from this idea."
            
        except Exception as e:
            return f"Error capturing idea: {str(e)}"
    
    def generate_article(self, message):
        """Generate an article from an existing idea"""
        try:
            # Find the most recent idea file
            idea_file = self.find_latest_idea_file()
            if not idea_file:
                return "No idea files found. Please capture an idea first."
            
            # Read the idea content
            with open(idea_file, 'r', encoding='utf-8') as f:
                idea_content = f.read()
            
            # Generate article using AI
            article_content = self.generate_article_content(idea_content)
            
            # Create filename for generated article
            date_str = datetime.datetime.now().strftime("%Y-%m-%d")
            safe_title = self.sanitize_filename(idea_file.stem)
            filename = f"{date_str}-{safe_title}-article.md"
            filepath = self.generated_path / filename
            
            # Save generated article
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(article_content)
            
            return f"Article generated successfully! Saved as '{filename}' in the generated folder. You can review and edit it before publishing."
            
        except Exception as e:
            return f"Error generating article: {str(e)}"
    
    def chat_response(self, message):
        """Generate a general chat response"""
        try:
            system_prompt = """You are a helpful content creation assistant. You help users with:
            - Capturing and structuring content ideas
            - Generating blog articles
            - Providing writing tips and suggestions
            - Answering questions about content creation
            
            Be concise, helpful, and encouraging."""
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=message)
            ]
            
            response = self.llm.invoke(messages)
            return response.content
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def extract_topic_from_message(self, message):
        """Extract the main topic from a user message"""
        # Remove common phrases
        topic = message.lower()
        topic = re.sub(r'i want to (write about|capture an idea about|create content about)', '', topic)
        topic = re.sub(r'generate article from', '', topic)
        topic = topic.strip()
        
        # If topic is still too long, use AI to extract key topic
        if len(topic) > 50:
            prompt = f"Extract the main topic (2-5 words) from this message: '{topic}'"
            response = self.llm.invoke([HumanMessage(content=prompt)])
            topic = response.content.strip()
        
        return topic
    
    def generate_idea_structure(self, topic):
        """Generate a structured idea template using AI"""
        prompt = f"""Create a structured content idea for the topic: "{topic}"

        Format it as a markdown file with the following structure:
        
        # {topic}
        
        ## Overview
        [Brief description of the content idea]
        
        ## Key Points
        - [Point 1]
        - [Point 2]
        - [Point 3]
        
        ## Target Audience
        [Description of who this content is for]
        
        ## Content Type
        [Blog post, tutorial, guide, etc.]
        
        ## Estimated Word Count
        [Approximate length]
        
        ## SEO Keywords
        - [keyword 1]
        - [keyword 2]
        - [keyword 3]
        
        ## Notes
        [Any additional notes or ideas]
        
        ---
        Created: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content
    
    def generate_article_content(self, idea_content):
        """Generate a full article based on the idea content"""
        prompt = f"""Based on this content idea, generate a complete blog article:

        {idea_content}
        
        Please create a well-structured, engaging blog post that:
        - Has a compelling introduction
        - Includes all the key points mentioned
        - Uses proper markdown formatting
        - Is optimized for the target audience
        - Has a clear conclusion
        - Is ready for Medium publishing
        
        Make it informative, engaging, and professional."""
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content
    
    def find_latest_idea_file(self):
        """Find the most recent idea file"""
        idea_files = list(self.ideas_path.glob("*.md"))
        if not idea_files:
            return None
        
        # Sort by modification time (newest first)
        idea_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return idea_files[0]
    
    def sanitize_filename(self, title):
        """Convert title to a safe filename"""
        # Remove special characters and replace spaces with hyphens
        safe_title = re.sub(r'[^\w\s-]', '', title)
        safe_title = re.sub(r'[-\s]+', '-', safe_title)
        safe_title = safe_title.strip('-')
        
        # Limit length
        if len(safe_title) > 50:
            safe_title = safe_title[:50]
        
        return safe_title.lower()
    
    def list_ideas(self):
        """List all available ideas"""
        idea_files = list(self.ideas_path.glob("*.md"))
        if not idea_files:
            return "No ideas found."
        
        ideas_list = "Available ideas:\n"
        for idea_file in idea_files:
            ideas_list += f"- {idea_file.name}\n"
        
        return ideas_list
    
    def read_idea(self, filename):
        """Read a specific idea file"""
        filepath = self.ideas_path / filename
        if not filepath.exists():
            return f"Idea file '{filename}' not found."
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            return content
        except Exception as e:
            return f"Error reading file: {str(e)}" 