// Chat functionality for Customer Support System

let isStreaming = false;
let currentStreamResponse = null;

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    loadExamples();
    checkSystemHealth();
    updateStats();
    
    // Set up periodic updates
    setInterval(updateStats, 30000); // Update stats every 30 seconds
    setInterval(checkSystemHealth, 60000); // Check health every minute
});

// Handle Enter key press
function handleKeyPress(event) {
    if (event.key === 'Enter') {
        sendMessage();
    }
}

// Send message to the server
async function sendMessage() {
    const input = document.getElementById('message-input');
    const message = input.value.trim();
    
    if (!message) return;
    
    // Add user message to chat
    addMessage(message, 'user');
    input.value = '';
    
    // Check if streaming is enabled
    const streamToggle = document.getElementById('stream-toggle');
    if (streamToggle.checked) {
        // Do NOT show typing indicator for streaming
        await sendStreamingMessage(message);
    } else {
        // Show typing indicator for non-streaming
        showTypingIndicator();
        await sendRegularMessage(message);
    }
}

// Send regular (non-streaming) message
async function sendRegularMessage(message) {
    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            hideTypingIndicator();
            addAssistantMessage(data.response, data.intent, data.confidence, data.model_used, data.response_time);
            updateStats(); // Update quick stats in real time
        } else {
            hideTypingIndicator();
            addErrorMessage(data.error || 'Failed to get response');
            updateStats(); // Update quick stats in real time
        }
    } catch (error) {
        hideTypingIndicator();
        addErrorMessage('Network error: ' + error.message);
        updateStats(); // Update quick stats in real time
    }
}

// Send streaming message
async function sendStreamingMessage(message) {
    try {
        const response = await fetch(`/api/chat/stream?message=${encodeURIComponent(message)}`);
        
        if (!response.ok) {
            throw new Error('Streaming request failed');
        }
        
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let responseText = '';
        let metadata = null;
        
        // Create assistant message container
        const messageElement = createAssistantMessageContainer();
        const contentElement = messageElement.querySelector('.message-content');
        
        while (true) {
            const { done, value } = await reader.read();
            
            if (done) break;
            
            const chunk = decoder.decode(value);
            const lines = chunk.split('\n');
            
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.slice(6));
                        
                        if (data.chunk) {
                            // Add chunk to response
                            responseText += data.chunk;
                            contentElement.textContent = responseText;
                            contentElement.classList.add('streaming-text');
                        } else if (data.done) {
                            // Final metadata
                            metadata = data;
                            contentElement.classList.remove('streaming-text');
                            
                            // Add metadata to message
                            addMessageMetadata(messageElement, metadata.intent, metadata.confidence, metadata.model_used, metadata.response_time);
                        } else if (data.error) {
                            throw new Error(data.error);
                        }
                    } catch (e) {
                        console.error('Error parsing stream data:', e);
                    }
                }
            }
        }
        
        hideTypingIndicator(); // In case it was shown for any reason
        updateStats(); // Update quick stats in real time
        
    } catch (error) {
        hideTypingIndicator();
        addErrorMessage('Streaming error: ' + error.message);
        updateStats(); // Update quick stats in real time
    }
}

// Add message to chat
function addMessage(content, type) {
    const chatContainer = document.getElementById('chat-messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = content;
    
    messageDiv.appendChild(contentDiv);
    chatContainer.appendChild(messageDiv);
    
    // Scroll to bottom
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// Add assistant message with metadata
function addAssistantMessage(content, intent, confidence, modelUsed, responseTime) {
    const chatContainer = document.getElementById('chat-messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = content;
    
    messageDiv.appendChild(contentDiv);
    
    // Add metadata
    addMessageMetadata(messageDiv, intent, confidence, modelUsed, responseTime);
    
    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// Create assistant message container for streaming
function createAssistantMessageContainer() {
    const chatContainer = document.getElementById('chat-messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    messageDiv.appendChild(contentDiv);
    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    
    return messageDiv;
}

// Add metadata to message
function addMessageMetadata(messageElement, intent, confidence, modelUsed, responseTime) {
    const metaDiv = document.createElement('div');
    metaDiv.className = 'message-meta';
    
    // Intent badge
    const intentBadge = document.createElement('span');
    intentBadge.className = `intent-badge ${intent}`;
    intentBadge.textContent = intent.charAt(0).toUpperCase() + intent.slice(1);
    metaDiv.appendChild(intentBadge);
    
    // Confidence
    const confidenceText = document.createElement('span');
    confidenceText.textContent = `Confidence: ${(confidence * 100).toFixed(1)}%`;
    metaDiv.appendChild(confidenceText);
    
    // Model used
    if (modelUsed) {
        const modelText = document.createElement('span');
        modelText.textContent = ` | Model: ${modelUsed}`;
        metaDiv.appendChild(modelText);
    }
    
    // Response time
    if (responseTime) {
        const timeText = document.createElement('span');
        timeText.textContent = ` | Time: ${responseTime.toFixed(2)}s`;
        metaDiv.appendChild(timeText);
    }
    
    // Confidence bar
    const confidenceBar = document.createElement('div');
    confidenceBar.className = 'confidence-bar';
    const confidenceFill = document.createElement('div');
    confidenceFill.className = 'confidence-fill';
    
    if (confidence < 0.5) {
        confidenceFill.classList.add('low');
    } else if (confidence < 0.8) {
        confidenceFill.classList.add('medium');
    }
    
    confidenceFill.style.width = `${confidence * 100}%`;
    confidenceBar.appendChild(confidenceFill);
    metaDiv.appendChild(confidenceBar);
    
    messageElement.appendChild(metaDiv);
}

// Add error message
function addErrorMessage(error) {
    const chatContainer = document.getElementById('chat-messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.style.backgroundColor = '#f8d7da';
    contentDiv.style.color = '#721c24';
    contentDiv.innerHTML = `<i class="fas fa-exclamation-triangle me-2"></i>${error}`;
    
    messageDiv.appendChild(contentDiv);
    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// Show typing indicator
function showTypingIndicator() {
    const chatContainer = document.getElementById('chat-messages');
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message assistant typing-indicator';
    typingDiv.id = 'typing-indicator';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = `
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
    `;
    
    typingDiv.appendChild(contentDiv);
    chatContainer.appendChild(typingDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// Hide typing indicator
function hideTypingIndicator() {
    const typingIndicator = document.getElementById('typing-indicator');
    if (typingIndicator) {
        typingIndicator.remove();
    }
}

// Load example queries
async function loadExamples() {
    try {
        const response = await fetch('/api/examples');
        const examples = await response.json();
        
        if (response.ok) {
            populateExamples('technical-examples', examples.technical);
            populateExamples('billing-examples', examples.billing);
            populateExamples('feature-examples', examples.feature);
        }
    } catch (error) {
        console.error('Error loading examples:', error);
    }
}

// Populate example queries
function populateExamples(containerId, examples) {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    container.innerHTML = '';
    
    examples.forEach(example => {
        const div = document.createElement('div');
        div.className = 'example-query';
        div.textContent = example;
        div.onclick = () => {
            document.getElementById('message-input').value = example;
        };
        container.appendChild(div);
    });
}

// Check system health
async function checkSystemHealth() {
    try {
        const response = await fetch('/api/health');
        const health = await response.json();
        
        if (response.ok) {
            updateHealthStatus(health);
        } else {
            updateHealthStatus({ system_status: 'error' });
        }
    } catch (error) {
        updateHealthStatus({ system_status: 'error' });
    }
}

// Update health status display
function updateHealthStatus(health) {
    const indicator = document.getElementById('status-indicator');
    const statusText = document.getElementById('status-text');
    const statusDetails = document.getElementById('status-details');
    
    if (health.system_status === 'healthy') {
        indicator.className = 'status-indicator healthy';
        statusText.textContent = 'Healthy';
        statusDetails.textContent = 'All systems operational';
    } else if (health.system_status === 'degraded') {
        indicator.className = 'status-indicator degraded';
        statusText.textContent = 'Degraded';
        statusDetails.textContent = 'Some services unavailable';
    } else {
        indicator.className = 'status-indicator error';
        statusText.textContent = 'Error';
        statusDetails.textContent = 'System unavailable';
    }
}

// Update statistics
async function updateStats() {
    try {
        const response = await fetch('/api/stats');
        const stats = await response.json();
        
        if (response.ok) {
            const supportStats = stats.support_system;
            document.getElementById('total-queries').textContent = supportStats.total_queries;
            document.getElementById('avg-time').textContent = `${supportStats.avg_response_time.toFixed(2)}s`;
        }
    } catch (error) {
        console.error('Error updating stats:', error);
    }
} 