"""
LLM Wrapper for Customer Support System
Handles Ollama (local) and OpenAI (fallback) with request queuing and streaming support.
"""

import asyncio
import json
import time
import logging
from typing import Dict, List, Optional, AsyncGenerator, Any
from dataclasses import dataclass
from queue import Queue, Empty
from threading import Thread, Lock
import requests
import openai
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LLMResponse:
    """Response from LLM with metadata"""
    content: str
    model_used: str
    tokens_used: int
    response_time: float
    success: bool
    error_message: Optional[str] = None

class RequestQueue:
    """Thread-safe request queue for concurrent processing"""
    
    def __init__(self, max_size: int = 100):
        self.queue = Queue(maxsize=max_size)
        self.processing_lock = Lock()
        self.active_requests = 0
        
    def add_request(self, request_id: str, prompt: str, callback) -> bool:
        """Add request to queue"""
        try:
            self.queue.put_nowait((request_id, prompt, callback))
            return True
        except:
            return False
    
    def get_request(self) -> Optional[tuple]:
        """Get next request from queue"""
        try:
            return self.queue.get_nowait()
        except Empty:
            return None
    
    def mark_complete(self):
        """Mark request as complete"""
        with self.processing_lock:
            self.active_requests -= 1
            self.queue.task_done()

class LLMWrapper:
    """Wrapper for LLM APIs with local/fallback switching and queuing"""
    
    def __init__(self):
        self.ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.local_model = os.getenv("LOCAL_MODEL_NAME", "tinyllama:1.1b")
        self.openai_model = os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo")
        
        # Initialize OpenAI client
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            self.openai_client = AsyncOpenAI(api_key=openai_api_key)
            self.openai_available = True
        else:
            self.openai_client = None
            self.openai_available = False
            logger.warning("OpenAI API key not found. Fallback will not be available.")
        
        # Request queue for concurrent processing
        self.request_queue = RequestQueue()
        self.processing_thread = Thread(target=self._process_queue, daemon=True)
        self.processing_thread.start()
        
        # Performance tracking
        self.local_success_rate = 0.0
        self.local_response_times = []
        self.fallback_usage_count = 0
        
    def _process_queue(self):
        """Background thread to process queued requests"""
        while True:
            request_data = self.request_queue.get_request()
            if request_data:
                request_id, prompt, callback = request_data
                try:
                    response = self._process_single_request(prompt)
                    callback(request_id, response)
                except Exception as e:
                    logger.error(f"Error processing request {request_id}: {e}")
                    error_response = LLMResponse(
                        content="",
                        model_used="error",
                        tokens_used=0,
                        response_time=0,
                        success=False,
                        error_message=str(e)
                    )
                    callback(request_id, error_response)
                finally:
                    self.request_queue.mark_complete()
            else:
                time.sleep(0.1)  # Small delay to prevent busy waiting
    
    def _process_single_request(self, prompt: str) -> LLMResponse:
        """Process a single request with local/fallback logic"""
        start_time = time.time()
        
        # Try local model first
        try:
            response = self._call_ollama(prompt)
            if response.success:
                self._update_local_metrics(response.response_time)
                return response
        except Exception as e:
            logger.warning(f"Local model failed: {e}")
        
        # Fallback to OpenAI if available
        if self.openai_available:
            try:
                response = self._call_openai(prompt)
                self.fallback_usage_count += 1
                return response
            except Exception as e:
                logger.error(f"OpenAI fallback failed: {e}")
        
        # If both fail, return error response
        return LLMResponse(
            content="",
            model_used="none",
            tokens_used=0,
            response_time=time.time() - start_time,
            success=False,
            error_message="Both local and OpenAI models failed"
        )
    
    def _call_ollama(self, prompt: str) -> LLMResponse:
        """Call Ollama API"""
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.local_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "max_tokens": 1000
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                content = data.get("response", "")
                tokens_used = len(content.split())  # Approximate token count
                
                return LLMResponse(
                    content=content,
                    model_used=f"ollama:{self.local_model}",
                    tokens_used=tokens_used,
                    response_time=time.time() - start_time,
                    success=True
                )
            else:
                raise Exception(f"Ollama API error: {response.status_code}")
                
        except Exception as e:
            raise Exception(f"Ollama request failed: {e}")
    
    async def _call_openai_async(self, prompt: str) -> LLMResponse:
        """Call OpenAI API asynchronously"""
        start_time = time.time()
        
        try:
            response = await self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.7
            )
            
            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if response.usage else 0
            
            return LLMResponse(
                content=content,
                model_used=f"openai:{self.openai_model}",
                tokens_used=tokens_used,
                response_time=time.time() - start_time,
                success=True
            )
            
        except Exception as e:
            raise Exception(f"OpenAI request failed: {e}")
    
    def _call_openai(self, prompt: str) -> LLMResponse:
        """Synchronous wrapper for OpenAI call"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._call_openai_async(prompt))
        finally:
            loop.close()
    
    def _update_local_metrics(self, response_time: float):
        """Update local model performance metrics"""
        self.local_response_times.append(response_time)
        if len(self.local_response_times) > 100:
            self.local_response_times.pop(0)
        
        # Calculate success rate (simplified)
        recent_times = self.local_response_times[-10:]
        self.local_success_rate = sum(1 for t in recent_times if t < 10.0) / len(recent_times)
    
    async def generate_stream(self, prompt: str) -> AsyncGenerator[str, None]:
        """Generate streaming response"""
        start_time = time.time()
        
        # Try local streaming first
        try:
            async for chunk in self._stream_ollama(prompt):
                yield chunk
            return
        except Exception as e:
            logger.warning(f"Local streaming failed: {e}")
        
        # Fallback to OpenAI streaming
        if self.openai_available:
            try:
                async for chunk in self._stream_openai(prompt):
                    yield chunk
                return
            except Exception as e:
                logger.error(f"OpenAI streaming failed: {e}")
        
        # If both fail, yield error message
        yield "Error: Unable to generate response"
    
    async def _stream_ollama(self, prompt: str) -> AsyncGenerator[str, None]:
        """Stream response from Ollama"""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.local_model,
                    "prompt": prompt,
                    "stream": True,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "max_tokens": 1000
                    }
                },
                stream=True,
                timeout=30
            )
            
            for line in response.iter_lines():
                if line:
                    data = json.loads(line.decode('utf-8'))
                    if 'response' in data:
                        yield data['response']
                    if data.get('done', False):
                        break
                        
        except Exception as e:
            raise Exception(f"Ollama streaming failed: {e}")
    
    async def _stream_openai(self, prompt: str) -> AsyncGenerator[str, None]:
        """Stream response from OpenAI"""
        try:
            stream = await self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.7,
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            raise Exception(f"OpenAI streaming failed: {e}")
    
    def generate(self, prompt: str, callback=None) -> Optional[LLMResponse]:
        """Generate response with optional callback for async processing"""
        if callback:
            # Add to queue for async processing
            request_id = f"req_{int(time.time() * 1000)}"
            self.request_queue.add_request(request_id, prompt, callback)
            return None
        else:
            # Synchronous processing
            return self._process_single_request(prompt)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        avg_local_time = sum(self.local_response_times) / len(self.local_response_times) if self.local_response_times else 0
        
        return {
            "local_success_rate": self.local_success_rate,
            "avg_local_response_time": avg_local_time,
            "fallback_usage_count": self.fallback_usage_count,
            "queue_size": self.request_queue.queue.qsize(),
            "active_requests": self.request_queue.active_requests
        }
    
    def health_check(self) -> Dict[str, bool]:
        """Check health of both LLM services"""
        health = {
            "local_available": False,
            "openai_available": self.openai_available
        }
        
        # Check local model
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                health["local_available"] = any(
                    model.get("name", "").startswith(self.local_model) 
                    for model in models
                )
        except:
            pass
        
        return health 