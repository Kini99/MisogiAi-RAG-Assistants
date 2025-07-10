import os
import json
import datetime
import requests
from pathlib import Path
from typing import Dict, Optional, List
import re

class MediumPublisher:
    def __init__(self):
        """Initialize the Medium publisher with API credentials"""
        self.access_token = os.getenv("MEDIUM_ACCESS_TOKEN")
        self.user_id = os.getenv("MEDIUM_USER_ID")
        self.api_base_url = "https://api.medium.com/v1"
        
        # Setup workspace paths
        self.workspace_path = Path("content-workspace")
        self.published_path = self.workspace_path / "published"
        self.generated_path = self.workspace_path / "generated"
    
    def publish_content(self, message: str) -> str:
        """
        Publish content to Medium based on user message
        
        Args:
            message: User message indicating what to publish
            
        Returns:
            str: Response message
        """
        try:
            # Find the most recent generated article
            article_file = self.find_latest_generated_article()
            if not article_file:
                return "No generated articles found. Please generate an article first."
            
            # Read the article content
            with open(article_file, 'r', encoding='utf-8') as f:
                article_content = f.read()
            
            # Extract title and content
            title, content = self.extract_title_and_content(article_content)
            
            # Publish to Medium
            publication_result = self.publish_to_medium(title, content)
            
            if publication_result.get('success'):
                # Move file to published folder
                self.move_to_published(article_file)
                
                # Update file with Medium URL
                self.update_file_with_medium_url(article_file, publication_result['url'])
                
                return f"Successfully published to Medium! Article URL: {publication_result['url']}"
            else:
                return f"Failed to publish to Medium: {publication_result.get('error', 'Unknown error')}"
                
        except Exception as e:
            return f"Error publishing content: {str(e)}"
    
    def publish_to_medium(self, title: str, content: str) -> Dict:
        """
        Publish content to Medium using the API
        
        Args:
            title: Article title
            content: Article content in markdown
            
        Returns:
            Dict: Publication result with success status and URL
        """
        try:
            if not self.access_token or not self.user_id:
                return {
                    'success': False,
                    'error': 'Medium API credentials not configured. Please set MEDIUM_ACCESS_TOKEN and MEDIUM_USER_ID environment variables.'
                }
            
            # Convert markdown to HTML for Medium
            html_content = self.markdown_to_html(content)
            
            # Prepare the post data
            post_data = {
                'title': title,
                'contentFormat': 'html',
                'content': html_content,
                'publishStatus': 'public'  # or 'draft' for drafts
            }
            
            # Make the API request
            headers = {
                'Authorization': f'Bearer {self.access_token}',
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
            
            url = f"{self.api_base_url}/users/{self.user_id}/posts"
            
            response = requests.post(url, headers=headers, json=post_data)
            
            if response.status_code == 201:
                result = response.json()
                return {
                    'success': True,
                    'url': result['data']['url'],
                    'post_id': result['data']['id']
                }
            else:
                return {
                    'success': False,
                    'error': f"API request failed with status {response.status_code}: {response.text}"
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def markdown_to_html(self, markdown_content: str) -> str:
        """
        Convert markdown content to HTML for Medium
        
        Args:
            markdown_content: Markdown formatted content
            
        Returns:
            str: HTML formatted content
        """
        # Basic markdown to HTML conversion
        html = markdown_content
        
        # Headers
        html = re.sub(r'^### (.*$)', r'<h3>\1</h3>', html, flags=re.MULTILINE)
        html = re.sub(r'^## (.*$)', r'<h2>\1</h2>', html, flags=re.MULTILINE)
        html = re.sub(r'^# (.*$)', r'<h1>\1</h1>', html, flags=re.MULTILINE)
        
        # Bold and italic
        html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html)
        html = re.sub(r'\*(.*?)\*', r'<em>\1</em>', html)
        
        # Code blocks
        html = re.sub(r'```(.*?)```', r'<pre><code>\1</code></pre>', html, flags=re.DOTALL)
        html = re.sub(r'`(.*?)`', r'<code>\1</code>', html)
        
        # Links
        html = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', html)
        
        # Lists
        html = re.sub(r'^- (.*$)', r'<li>\1</li>', html, flags=re.MULTILINE)
        html = re.sub(r'(\n<li>.*</li>\n)+', r'<ul>\g<0></ul>', html, flags=re.DOTALL)
        
        # Paragraphs
        html = re.sub(r'\n\n([^<].*?)\n\n', r'<p>\1</p>', html, flags=re.DOTALL)
        
        # Line breaks
        html = html.replace('\n', '<br>')
        
        return html
    
    def extract_title_and_content(self, article_content: str) -> tuple:
        """
        Extract title and content from article
        
        Args:
            article_content: Full article content
            
        Returns:
            tuple: (title, content)
        """
        lines = article_content.split('\n')
        
        # Extract title from first line (should start with #)
        title = ""
        content_start = 0
        
        for i, line in enumerate(lines):
            if line.startswith('# '):
                title = line[2:].strip()
                content_start = i + 1
                break
        
        # If no title found, use first line
        if not title:
            title = lines[0].strip()
            content_start = 1
        
        # Get content (everything after title)
        content = '\n'.join(lines[content_start:]).strip()
        
        return title, content
    
    def find_latest_generated_article(self) -> Optional[Path]:
        """
        Find the most recent generated article
        
        Returns:
            Path: Path to the latest article file
        """
        article_files = list(self.generated_path.glob("*-article.md"))
        if not article_files:
            return None
        
        # Sort by modification time (newest first)
        article_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return article_files[0]
    
    def move_to_published(self, article_file: Path) -> bool:
        """
        Move an article from generated to published folder
        
        Args:
            article_file: Path to the article file
            
        Returns:
            bool: True if successful
        """
        try:
            # Create destination path
            dest_path = self.published_path / article_file.name
            
            # Move the file
            article_file.rename(dest_path)
            
            return True
            
        except Exception as e:
            print(f"Error moving file to published: {str(e)}")
            return False
    
    def update_file_with_medium_url(self, article_file: Path, medium_url: str) -> bool:
        """
        Update the article file with Medium URL and publication date
        
        Args:
            article_file: Path to the article file
            medium_url: URL of the published Medium article
            
        Returns:
            bool: True if successful
        """
        try:
            # Read current content
            with open(article_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Add publication metadata
            publication_info = f"""
---
Published on Medium: {medium_url}
Publication Date: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
---
"""
            
            # Add to end of file
            updated_content = content + publication_info
            
            # Write back to file
            with open(article_file, 'w', encoding='utf-8') as f:
                f.write(updated_content)
            
            return True
            
        except Exception as e:
            print(f"Error updating file with Medium URL: {str(e)}")
            return False
    
    def get_medium_user_info(self) -> Dict:
        """
        Get Medium user information
        
        Returns:
            Dict: User information
        """
        try:
            if not self.access_token:
                return {'error': 'Access token not configured'}
            
            headers = {
                'Authorization': f'Bearer {self.access_token}',
                'Accept': 'application/json'
            }
            
            response = requests.get(f"{self.api_base_url}/me", headers=headers)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {'error': f'Failed to get user info: {response.status_code}'}
                
        except Exception as e:
            return {'error': str(e)}
    
    def list_published_articles(self) -> List[Dict]:
        """
        List articles in the published folder
        
        Returns:
            List[Dict]: List of published articles
        """
        try:
            articles = []
            
            for article_file in self.published_path.glob("*.md"):
                # Read publication info
                with open(article_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extract Medium URL if present
                medium_url = None
                if 'Published on Medium:' in content:
                    url_match = re.search(r'Published on Medium: (https://.*?)\n', content)
                    if url_match:
                        medium_url = url_match.group(1)
                
                articles.append({
                    'filename': article_file.name,
                    'path': str(article_file),
                    'medium_url': medium_url,
                    'modified': datetime.datetime.fromtimestamp(article_file.stat().st_mtime).isoformat()
                })
            
            return articles
            
        except Exception as e:
            print(f"Error listing published articles: {str(e)}")
            return []
    
    def create_draft(self, title: str, content: str) -> Dict:
        """
        Create a draft post on Medium
        
        Args:
            title: Article title
            content: Article content
            
        Returns:
            Dict: Draft creation result
        """
        try:
            if not self.access_token or not self.user_id:
                return {
                    'success': False,
                    'error': 'Medium API credentials not configured'
                }
            
            # Convert markdown to HTML
            html_content = self.markdown_to_html(content)
            
            # Prepare draft data
            draft_data = {
                'title': title,
                'contentFormat': 'html',
                'content': html_content,
                'publishStatus': 'draft'
            }
            
            headers = {
                'Authorization': f'Bearer {self.access_token}',
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
            
            url = f"{self.api_base_url}/users/{self.user_id}/posts"
            
            response = requests.post(url, headers=headers, json=draft_data)
            
            if response.status_code == 201:
                result = response.json()
                return {
                    'success': True,
                    'draft_id': result['data']['id'],
                    'url': result['data']['url']
                }
            else:
                return {
                    'success': False,
                    'error': f"Failed to create draft: {response.status_code}"
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            } 