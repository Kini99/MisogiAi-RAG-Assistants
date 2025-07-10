# SaaS Customer Support System

A comprehensive customer support system for SaaS companies that handles different query types (technical issues, billing questions, feature requests) with tailored processing strategies using local LLMs and OpenAI fallback.

## Features

- **Local LLM Integration**: Ollama with TinyLlama 1.1B model support
- **Intent Detection**: Automatic classification of queries into three categories
- **Tailored Processing**: Different strategies for each query type
- **Request Queuing**: Concurrent request handling
- **Evaluation Framework**: Comprehensive metrics and A/B testing
- **Web UI**: Simple interface for testing and evaluation
- **Streaming Support**: Real-time response streaming
- **Dashboard**: Real-time metrics and visualizations
- **Health Monitoring**: System status and performance tracking

## Query Types

1. **Technical Support**: Routes to code examples and documentation
2. **Billing/Account**: Routes to pricing tables and policies  
3. **Feature Requests**: Routes to roadmap and comparison data

## Requirements

- Python 3.11+
- Ollama (for local LLM)
- OpenAI API key (for fallback)

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd w5d1q2
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Ollama**
   ```bash
   # macOS/Linux
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Windows
   # Download from https://ollama.ai/download
   ```

5. **Pull TinyLlama model**
   ```bash
   ollama pull tinyllama:1.1b
   ```

6. **Set up environment variables**
   ```bash
   cp env.example .env
   # Edit .env with your OpenAI API key
   ```

## Usage

### Starting the System

1. **Start Ollama service**
   ```bash
   ollama serve
   ```

2. **Run the web application**
   ```bash
   python app.py
   ```

3. **Access the web UI**
   - Main chat interface: http://localhost:5000
   - Dashboard: http://localhost:5000/dashboard
   - Evaluation: http://localhost:5000/evaluation

### API Usage

```python
from support_system import CustomerSupportSystem

# Initialize the system
support = CustomerSupportSystem()

# Process a query
response = support.process_query("How do I integrate the API?")
print(response.response)
print(f"Intent: {response.intent.intent}")
print(f"Confidence: {response.intent.confidence}")
```

### Evaluation

Run the evaluation framework:

```bash
# Full evaluation with all test queries
python evaluate.py

# Quick evaluation with balanced samples
python evaluate.py --mode balanced --samples 5

# Health check only
python evaluate.py --mode health

# Specific intent evaluation
python evaluate.py --mode intent --intent technical
```

The evaluation will:
- Test 20 queries per intent (60 total)
- Generate metrics report
- Create visualizations
- Perform A/B testing between local and OpenAI models

## Project Structure

```
w5d1q2/
├── app.py                 # Main web application
├── support_system.py      # Core support system
├── llm_wrapper.py         # LLM API wrapper
├── intent_detector.py     # Intent classification
├── processors/            # Query processors
│   ├── __init__.py
│   ├── technical.py
│   ├── billing.py
│   └── feature_request.py
├── evaluation/            # Evaluation framework
│   ├── __init__.py
│   ├── evaluator.py
│   ├── test_queries.py
│   └── metrics.py
├── static/                # Web UI assets
│   ├── css/
│   │   └── style.css
│   └── js/
│       ├── chat.js
│       ├── dashboard.js
│       └── evaluation.js
├── templates/             # HTML templates
│   ├── index.html
│   ├── dashboard.html
│   └── evaluation.html
├── data/                  # Sample data and knowledge base
│   ├── technical_kb.json
│   ├── billing_kb.json
│   └── feature_roadmap.json
├── requirements.txt       # Python dependencies
├── env.example            # Environment variables template
├── evaluate.py            # Evaluation script
└── README.md              # This file
```

## Configuration

### Environment Variables

Create a `.env` file with:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
LOCAL_MODEL_NAME=tinyllama:1.1b

# OpenAI Model Configuration
OPENAI_MODEL_NAME=gpt-3.5-turbo

# Application Configuration
FLASK_ENV=development
FLASK_DEBUG=True
PORT=5000

# Evaluation Configuration
EVAL_OUTPUT_DIR=./evaluation_results
LOG_LEVEL=INFO
```

### Model Configuration

- **Local Model**: TinyLlama 1.1B (optimized for speed and resource usage)
- **Fallback Model**: OpenAI GPT-3.5-turbo
- **Intent Detection**: Uses local model with specialized prompts

## API Endpoints

### Chat Interface
- `POST /api/chat` - Process a chat message
- `GET /api/chat/stream` - Stream chat responses
- `GET /api/examples` - Get example queries
- `POST /api/reset-stats` - Reset system statistics

### System Monitoring
- `GET /api/stats` - Get system statistics
- `GET /api/health` - Health check

### Evaluation
- `POST /api/evaluate` - Run evaluation tests

## Evaluation Metrics

The system evaluates:

- **Intent Classification Accuracy**: Percentage of correctly classified queries
- **Response Relevance**: Cosine similarity between response and expected output
- **Context Utilization Score**: How well the system uses available context
- **Response Time**: Average processing time per query
- **Token Usage**: Efficiency metrics for both models
- **System Health**: Overall system performance and availability

## Sample Test Queries

### Technical Support
- "How do I integrate the API?"
- "Getting 404 error when calling endpoint"
- "Authentication not working"

### Billing/Account
- "How much does the premium plan cost?"
- "I want to cancel my subscription"
- "Update my billing information"

### Feature Requests
- "Can you add dark mode?"
- "Need export to PDF functionality"
- "Request for mobile app"

## A/B Testing

The evaluation framework automatically compares:
- Local TinyLlama vs OpenAI GPT-3.5-turbo
- Response quality and relevance
- Processing speed and cost efficiency

## System Requirements

- **Memory Usage**: ~2GB RAM for TinyLlama model
- **Storage**: ~1.5GB for TinyLlama model
- **Network**: Internet connection for OpenAI API fallback
- **Browser**: Modern web browser for UI access

