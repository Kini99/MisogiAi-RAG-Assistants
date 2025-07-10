// Evaluation functionality for Customer Support System

let evaluationInProgress = false;

// Initialize evaluation page
document.addEventListener('DOMContentLoaded', function() {
    setupEventListeners();
});

// Setup event listeners
function setupEventListeners() {
    const modeSelect = document.getElementById('evaluation-mode');
    const intentSelect = document.getElementById('specific-intent');
    
    modeSelect.addEventListener('change', function() {
        if (this.value === 'intent') {
            intentSelect.disabled = false;
        } else {
            intentSelect.disabled = true;
        }
    });
}

// Start evaluation
async function startEvaluation() {
    if (evaluationInProgress) {
        alert('Evaluation already in progress. Please wait for it to complete.');
        return;
    }
    
    const mode = document.getElementById('evaluation-mode').value;
    const samples = parseInt(document.getElementById('samples-per-intent').value);
    const intent = document.getElementById('specific-intent').value;
    
    // Validate inputs
    if (mode === 'intent' && !intent) {
        alert('Please select a specific intent for intent-specific evaluation.');
        return;
    }
    
    if (samples < 1 || samples > 20) {
        alert('Samples per intent must be between 1 and 20.');
        return;
    }
    
    // Show progress
    showProgress();
    evaluationInProgress = true;
    
    try {
        const requestBody = {
            mode: mode,
            samples: samples
        };
        
        if (mode === 'intent') {
            requestBody.intent = intent;
        }
        
        const response = await fetch('/api/evaluate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestBody)
        });
        
        const result = await response.json();
        
        if (response.ok) {
            hideProgress();
            displayResults(result);
        } else {
            throw new Error(result.error || 'Evaluation failed');
        }
        
    } catch (error) {
        hideProgress();
        alert('Evaluation failed: ' + error.message);
    } finally {
        evaluationInProgress = false;
    }
}

// Show evaluation progress
function showProgress() {
    document.getElementById('evaluation-progress').style.display = 'block';
    document.getElementById('evaluation-results').style.display = 'none';
    document.getElementById('ab-test-results').style.display = 'none';
    document.getElementById('recommendations').style.display = 'none';
    
    // Simulate progress updates
    let progress = 0;
    const progressBar = document.getElementById('progress-bar');
    const progressText = document.getElementById('progress-text');
    
    const progressInterval = setInterval(() => {
        progress += Math.random() * 15;
        if (progress > 90) progress = 90;
        
        progressBar.style.width = progress + '%';
        
        if (progress < 30) {
            progressText.textContent = 'Initializing evaluation...';
        } else if (progress < 60) {
            progressText.textContent = 'Running tests with local model...';
        } else if (progress < 90) {
            progressText.textContent = 'Running tests with OpenAI model...';
        } else {
            progressText.textContent = 'Calculating metrics...';
        }
    }, 500);
    
    // Store interval for cleanup
    window.progressInterval = progressInterval;
}

// Hide evaluation progress
function hideProgress() {
    document.getElementById('evaluation-progress').style.display = 'none';
    
    if (window.progressInterval) {
        clearInterval(window.progressInterval);
    }
    
    // Complete progress bar
    document.getElementById('progress-bar').style.width = '100%';
    document.getElementById('progress-text').textContent = 'Evaluation completed!';
}

// Display evaluation results
function displayResults(result) {
    const resultsContent = document.getElementById('results-content');
    const summary = result.summary;
    
    let html = `
        <div class="row">
            <div class="col-md-6">
                <h6>Evaluation Summary</h6>
                <ul class="list-unstyled">
                    <li><strong>Total Queries:</strong> ${summary.total_queries_tested}</li>
                    <li><strong>Evaluation Mode:</strong> ${summary.evaluation_mode || 'Balanced'}</li>
                    <li><strong>Timestamp:</strong> ${new Date().toLocaleString()}</li>
                </ul>
            </div>
            <div class="col-md-6">
                <h6>Intent Distribution</h6>
                <ul class="list-unstyled">
    `;
    
    for (const [intent, count] of Object.entries(summary.intent_distribution || {})) {
        html += `<li><strong>${intent.charAt(0).toUpperCase() + intent.slice(1)}:</strong> ${count}</li>`;
    }
    
    html += `
                </ul>
            </div>
        </div>
    `;
    
    resultsContent.innerHTML = html;
    document.getElementById('evaluation-results').style.display = 'block';
    
    // Display A/B test results if available
    if (result.results && result.results.ab_test_results) {
        displayABTestResults(result.results);
    }
    
    // Display recommendations if available
    if (summary.recommendations) {
        displayRecommendations(summary.recommendations);
    }
}

// Display A/B test results
function displayABTestResults(results) {
    const localResults = document.getElementById('local-results');
    const openaiResults = document.getElementById('openai-results');
    const comparisonResults = document.getElementById('comparison-results');
    
    const local = results.local_results;
    const openai = results.openai_results;
    const abTest = results.ab_test_results;
    
    // Local model results
    localResults.innerHTML = `
        <div class="card">
            <div class="card-body">
                <p><strong>Accuracy:</strong> ${(local.intent_accuracy.overall_accuracy * 100).toFixed(1)}%</p>
                <p><strong>Relevance:</strong> ${(local.response_relevance.overall_mean_relevance * 100).toFixed(1)}%</p>
                <p><strong>Avg Response Time:</strong> ${local.performance_metrics.avg_response_time.toFixed(2)}s</p>
                <p><strong>Total Tokens:</strong> ${local.performance_metrics.total_tokens}</p>
            </div>
        </div>
    `;
    
    // OpenAI model results
    openaiResults.innerHTML = `
        <div class="card">
            <div class="card-body">
                <p><strong>Accuracy:</strong> ${(openai.intent_accuracy.overall_accuracy * 100).toFixed(1)}%</p>
                <p><strong>Relevance:</strong> ${(openai.response_relevance.overall_mean_relevance * 100).toFixed(1)}%</p>
                <p><strong>Avg Response Time:</strong> ${openai.performance_metrics.avg_response_time.toFixed(2)}s</p>
                <p><strong>Total Tokens:</strong> ${openai.performance_metrics.total_tokens}</p>
            </div>
        </div>
    `;
    
    // Comparison results
    const winner = abTest.winner;
    const winnerClass = winner === 'local' ? 'text-success' : winner === 'openai' ? 'text-primary' : 'text-warning';
    
    comparisonResults.innerHTML = `
        <div class="alert ${winnerClass}">
            <h6><i class="fas fa-trophy me-2"></i>A/B Test Winner: ${winner.toUpperCase()}</h6>
            <p><strong>Accuracy Difference:</strong> ${(abTest.accuracy_comparison.difference * 100).toFixed(1)}%</p>
            <p><strong>Relevance Difference:</strong> ${(abTest.relevance_comparison.difference * 100).toFixed(1)}%</p>
            <p><strong>Speed Improvement:</strong> ${abTest.performance_comparison.speed_improvement.toFixed(2)}x</p>
            <p><strong>Token Efficiency:</strong> ${abTest.cost_comparison.token_efficiency.toFixed(2)}x</p>
        </div>
    `;
    
    document.getElementById('ab-test-results').style.display = 'block';
}

// Display recommendations
function displayRecommendations(recommendations) {
    const recommendationsContent = document.getElementById('recommendations-content');
    
    let html = '<ul class="list-group">';
    recommendations.forEach(rec => {
        html += `<li class="list-group-item"><i class="fas fa-lightbulb me-2 text-warning"></i>${rec}</li>`;
    });
    html += '</ul>';
    
    recommendationsContent.innerHTML = html;
    document.getElementById('recommendations').style.display = 'block';
}

// Load previous results (placeholder)
function loadPreviousResults() {
    alert('This feature would load previous evaluation results from saved files. Not implemented in this demo.');
}

// Export results (placeholder)
function exportResults() {
    alert('This feature would export evaluation results to various formats. Not implemented in this demo.');
} 