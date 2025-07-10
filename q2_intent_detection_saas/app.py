"""
Flask Web Application for Customer Support System
Provides web interface for testing and evaluation
"""

import os
import json
import asyncio
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response, stream_template
from flask_cors import CORS
import logging

from support_system import CustomerSupportSystem
from evaluation.evaluator import Evaluator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Initialize support system
support_system = CustomerSupportSystem()

@app.route('/')
def index():
    """Main page with chat interface"""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat requests"""
    try:
        data = request.get_json()
        query = data.get('message', '').strip()
        
        if not query:
            return jsonify({'error': 'No message provided'}), 400
        
        # Process query
        result = support_system.process_query(query)
        
        response_data = {
            'response': result.response,
            'intent': result.intent.intent,
            'confidence': result.intent.confidence,
            'model_used': result.model_used,
            'response_time': result.response_time,
            'tokens_used': result.tokens_used,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/chat/stream')
def chat_stream():
    """Streaming chat endpoint"""
    query = request.args.get('message', '').strip()
    
    if not query:
        return jsonify({'error': 'No message provided'}), 400
    
    def generate():
        try:
            # Process query with streaming
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def stream_response():
                async for chunk in support_system.llm_wrapper.generate_stream(query):
                    yield f"data: {json.dumps({'chunk': chunk})}\n\n"

                # Send final metadata
                result = support_system.process_query(query)
                metadata = {
                    'intent': result.intent.intent,
                    'confidence': result.intent.confidence,
                    'model_used': result.model_used,
                    'response_time': result.response_time,
                    'tokens_used': result.tokens_used,
                    'done': True
                }
                yield f"data: {json.dumps(metadata)}\n\n"

            # Correctly consume the async generator and yield all chunks
            gen = stream_response()
            while True:
                try:
                    chunk = loop.run_until_complete(gen.__anext__())
                    yield chunk
                except StopAsyncIteration:
                    break

        except Exception as e:
            logger.error(f"Error in streaming: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return Response(generate(), mimetype='text/plain')

@app.route('/api/stats')
def get_stats():
    """Get system statistics"""
    try:
        stats = support_system.get_system_stats()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({'error': 'Failed to get statistics'}), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    try:
        health = support_system.health_check()
        return jsonify(health)
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return jsonify({'error': 'Health check failed'}), 500

@app.route('/api/examples')
def get_examples():
    """Get example queries for each intent"""
    try:
        examples = support_system.get_intent_examples()
        return jsonify(examples)
    except Exception as e:
        logger.error(f"Error getting examples: {e}")
        return jsonify({'error': 'Failed to get examples'}), 500

@app.route('/api/evaluate', methods=['POST'])
def run_evaluation():
    """Run evaluation"""
    try:
        data = request.get_json()
        mode = data.get('mode', 'balanced')
        samples = data.get('samples', 3)
        
        evaluator = Evaluator()
        
        if mode == 'full':
            results = evaluator.run_full_evaluation()
        elif mode == 'balanced':
            results = evaluator.run_balanced_evaluation(samples)
        else:
            return jsonify({'error': 'Invalid evaluation mode'}), 400
        
        return jsonify({
            'success': True,
            'results': results,
            'summary': results.get('summary', {})
        })
        
    except Exception as e:
        logger.error(f"Error in evaluation: {e}")
        return jsonify({'error': 'Evaluation failed'}), 500

@app.route('/api/reset-stats', methods=['POST'])
def reset_stats():
    """Reset system statistics"""
    try:
        support_system.reset_stats()
        return jsonify({'success': True, 'message': 'Statistics reset successfully'})
    except Exception as e:
        logger.error(f"Error resetting stats: {e}")
        return jsonify({'error': 'Failed to reset statistics'}), 500

@app.route('/dashboard')
def dashboard():
    """Dashboard page with metrics and visualizations"""
    return render_template('dashboard.html')

@app.route('/evaluation')
def evaluation_page():
    """Evaluation page"""
    return render_template('evaluation.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    print("="*60)
    print("CUSTOMER SUPPORT SYSTEM WEB APPLICATION")
    print("="*60)
    print(f"Starting server on port {port}")
    print(f"Debug mode: {debug}")
    print(f"Access the application at: http://localhost:{port}")
    print("="*60)
    
    app.run(host='0.0.0.0', port=port, debug=debug) 