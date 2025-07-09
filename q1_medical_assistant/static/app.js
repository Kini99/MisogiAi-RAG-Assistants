// Enhanced JavaScript functionality for Medical Knowledge Assistant

class MedicalRAGApp {
    constructor() {
        this.isProcessing = false;
        this.currentQuery = null;
        this.queryHistory = [];
        this.init();
    }

    init() {
        this.initializeUploadArea();
        this.initializeQueryForm();
        this.initializeEventListeners();
        this.checkSystemHealth();
        this.startHealthMonitoring();
    }

    initializeUploadArea() {
        const uploadArea = document.getElementById('upload-area');
        const fileInput = document.getElementById('file-input');

        if (!uploadArea || !fileInput) return;

        // Drag and drop functionality
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            this.handleFileUpload(files);
        });

        // File input change
        fileInput.addEventListener('change', (e) => {
            this.handleFileUpload(e.target.files);
        });
    }

    initializeQueryForm() {
        const form = document.getElementById('query-form');
        if (!form) return;

        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            if (this.isProcessing) return;
            
            const query = document.getElementById('query-input')?.value.trim();
            const includeSources = document.getElementById('include-sources')?.checked;
            const evaluateResponse = document.getElementById('evaluate-response')?.checked;
            
            if (!query) {
                this.showAlert('Please enter a medical query.', 'warning');
                return;
            }
            
            await this.submitQuery(query, includeSources, evaluateResponse);
        });
    }

    initializeEventListeners() {
        // Add keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 'Enter') {
                const submitBtn = document.getElementById('submit-btn');
                if (submitBtn && !submitBtn.disabled) {
                    submitBtn.click();
                }
            }
        });

        // Add query history functionality
        this.initializeQueryHistory();
    }

    initializeQueryHistory() {
        const queryInput = document.getElementById('query-input');
        if (!queryInput) return;

        // Load query history from localStorage
        const history = JSON.parse(localStorage.getItem('medicalQueryHistory') || '[]');
        this.queryHistory = history;

        // Add autocomplete functionality
        queryInput.addEventListener('input', (e) => {
            const value = e.target.value.toLowerCase();
            if (value.length > 2) {
                this.showQuerySuggestions(value);
            } else {
                this.hideQuerySuggestions();
            }
        });
    }

    showQuerySuggestions(query) {
        const suggestions = this.queryHistory
            .filter(item => item.toLowerCase().includes(query))
            .slice(0, 5);

        if (suggestions.length === 0) return;

        let suggestionsDiv = document.getElementById('query-suggestions');
        if (!suggestionsDiv) {
            suggestionsDiv = document.createElement('div');
            suggestionsDiv.id = 'query-suggestions';
            suggestionsDiv.className = 'query-suggestions';
            document.getElementById('query-input').parentNode.appendChild(suggestionsDiv);
        }

        suggestionsDiv.innerHTML = suggestions
            .map(suggestion => `<div class="suggestion-item" onclick="app.selectSuggestion('${suggestion}')">${suggestion}</div>`)
            .join('');
        suggestionsDiv.style.display = 'block';
    }

    hideQuerySuggestions() {
        const suggestionsDiv = document.getElementById('query-suggestions');
        if (suggestionsDiv) {
            suggestionsDiv.style.display = 'none';
        }
    }

    selectSuggestion(suggestion) {
        document.getElementById('query-input').value = suggestion;
        this.hideQuerySuggestions();
    }

    async handleFileUpload(files) {
        if (files.length === 0) return;
        
        const formData = new FormData();
        for (let file of files) {
            formData.append('files', file);
        }
        
        this.showAlert('Uploading documents...', 'info');
        
        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (response.ok) {
                this.showAlert(`Successfully uploaded ${result.documents_processed} documents with ${result.chunks_created} chunks created.`, 'success');
                this.updateVectorStoreStats();
            } else {
                this.showAlert(`Upload failed: ${result.detail}`, 'error');
            }
        } catch (error) {
            this.showAlert(`Upload error: ${error.message}`, 'error');
        }
    }

    async submitQuery(query, includeSources, evaluateResponse) {
        this.isProcessing = true;
        this.currentQuery = query;
        
        const submitBtn = document.getElementById('submit-btn');
        const responseSection = document.getElementById('response-section');
        const loading = document.getElementById('loading');
        const responseContent = document.getElementById('response-content');
        
        // Show loading state
        submitBtn.disabled = true;
        submitBtn.textContent = '🔄 Processing...';
        responseSection.style.display = 'block';
        loading.style.display = 'block';
        responseContent.innerHTML = '';
        
        // Add query to history
        this.addToQueryHistory(query);
        
        try {
            const response = await fetch('/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    query: query,
                    include_sources: includeSources,
                    evaluate_response: evaluateResponse
                })
            });
            
            const result = await response.json();
            
            if (response.ok) {
                this.displayResponse(result);
                this.animateResponse();
            } else {
                this.showAlert(`Query failed: ${result.detail}`, 'error');
            }
        } catch (error) {
            this.showAlert(`Query error: ${error.message}`, 'error');
        } finally {
            // Reset loading state
            this.isProcessing = false;
            submitBtn.disabled = false;
            submitBtn.textContent = '🔍 Submit Query';
            loading.style.display = 'none';
        }
    }

    displayResponse(result) {
        const responseContent = document.getElementById('response-content');
        
        let html = `
            <div class="response-content fade-in">
                <h3>📝 Medical Response</h3>
                <p>${result.response}</p>
            </div>
        `;
        
        // Display RAGAS metrics if available
        if (result.ragas_metrics) {
            html += `
                <h3>📊 RAGAS Evaluation Metrics</h3>
                <div class="metrics-grid">
                    ${this.createMetricCard('Faithfulness', result.ragas_metrics.faithfulness || 0, 0.90)}
                    ${this.createMetricCard('Context Precision/Utilization', result.ragas_metrics.context_precision || result.ragas_metrics.context_utilization || 0, 0.85)}
                    ${this.createMetricCard('Context Recall', result.ragas_metrics.context_recall || 0, 0.80)}
                    ${this.createMetricCard('Answer Relevancy', result.ragas_metrics.answer_relevancy || 0, 0.85)}
                </div>
            `;
            
            // Quality check results
            if (result.quality_check) {
                const qualityStatus = result.quality_check.overall_pass ? 'PASS' : 'FAIL';
                const qualityClass = result.quality_check.overall_pass ? 'success' : 'error';
                
                html += `
                    <div class="alert alert-${qualityClass} fade-in">
                        <strong>Quality Check:</strong> ${qualityStatus}
                        ${result.quality_check.warnings ? '<br>Warnings: ' + result.quality_check.warnings.join(', ') : ''}
                    </div>
                `;
            }
        }
        
        // Display sources if available
        if (result.sources && result.sources.length > 0) {
            html += `
                <div class="sources-section fade-in">
                    <h3>📚 Sources</h3>
                    ${result.sources.map(source => `
                        <div class="source-item">
                            <strong>${source.title || 'Medical Document'}</strong><br>
                            <small>Page: ${source.page || 'N/A'} | Score: ${(source.score || 0).toFixed(3)}</small>
                        </div>
                    `).join('')}
                </div>
            `;
        }
        
        // Performance metrics
        html += `
            <div class="metrics-grid">
                ${this.createMetricCard('Generation Time', (result.generation_time || 0).toFixed(2) + 's', null)}
                ${this.createMetricCard('Safety Score', result.safety_score || 0, 0.95)}
            </div>
        `;
        
        responseContent.innerHTML = html;
    }

    createMetricCard(label, value, threshold) {
        const numericValue = typeof value === 'number' ? value : parseFloat(value);
        let qualityClass = '';
        
        if (threshold !== null && typeof numericValue === 'number') {
            if (numericValue >= threshold) {
                qualityClass = 'quality-excellent';
            } else if (numericValue >= threshold * 0.8) {
                qualityClass = 'quality-good';
            } else {
                qualityClass = 'quality-poor';
            }
        }
        
        return `
            <div class="metric-card fade-in">
                <div class="metric-value">
                    ${typeof value === 'number' ? value.toFixed(3) : value}
                    ${qualityClass ? `<span class="quality-indicator ${qualityClass}"></span>` : ''}
                </div>
                <div class="metric-label">${label}</div>
                ${threshold ? `<div class="progress-bar"><div class="progress-fill" style="width: ${Math.min((numericValue / threshold) * 100, 100)}%"></div></div>` : ''}
            </div>
        `;
    }

    animateResponse() {
        const responseSection = document.getElementById('response-section');
        responseSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    addToQueryHistory(query) {
        if (!this.queryHistory.includes(query)) {
            this.queryHistory.unshift(query);
            this.queryHistory = this.queryHistory.slice(0, 10); // Keep only last 10 queries
            localStorage.setItem('medicalQueryHistory', JSON.stringify(this.queryHistory));
        }
    }

    async updateVectorStoreStats() {
        try {
            const response = await fetch('/vector-store/stats');
            const stats = await response.json();
            
            // Update stats display if available
            const statsElement = document.getElementById('vector-stats');
            if (statsElement) {
                statsElement.innerHTML = `
                    <div class="metric-card">
                        <div class="metric-value">${stats.total_documents || 0}</div>
                        <div class="metric-label">Total Documents</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">${stats.total_chunks || 0}</div>
                        <div class="metric-label">Total Chunks</div>
                    </div>
                `;
            }
        } catch (error) {
            console.error('Error updating vector store stats:', error);
        }
    }

    showAlert(message, type) {
        const statusDiv = document.getElementById('upload-status');
        const alertClass = `alert alert-${type}`;
        
        statusDiv.innerHTML = `<div class="${alertClass} fade-in">${message}</div>`;
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            statusDiv.innerHTML = '';
        }, 5000);
    }

    async checkSystemHealth() {
        try {
            const response = await fetch('/health');
            const health = await response.json();
            
            const statusIndicator = document.getElementById('system-status');
            if (health.status === 'healthy') {
                statusIndicator.textContent = 'System Online';
                statusIndicator.style.background = '#27ae60';
            } else {
                statusIndicator.textContent = 'System Issues';
                statusIndicator.style.background = '#e74c3c';
            }
        } catch (error) {
            const statusIndicator = document.getElementById('system-status');
            statusIndicator.textContent = 'System Offline';
            statusIndicator.style.background = '#e74c3c';
        }
    }

    startHealthMonitoring() {
        // Check system health every 30 seconds
        setInterval(() => {
            this.checkSystemHealth();
        }, 30000);
    }
}

// Initialize the application when DOM is loaded
let app;
document.addEventListener('DOMContentLoaded', function() {
    app = new MedicalRAGApp();
});

// Export for global access
window.app = app; 