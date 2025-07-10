// Dashboard functionality for Customer Support System

let intentChart = null;
let responseTimeChart = null;
let updateInterval = null;

// Initialize dashboard
document.addEventListener('DOMContentLoaded', function() {
    loadDashboard();
    
    // Set up periodic updates
    updateInterval = setInterval(loadDashboard, 10000); // Update every 10 seconds
});

// Load dashboard data
async function loadDashboard() {
    try {
        const response = await fetch('/api/stats');
        const stats = await response.json();
        
        if (response.ok) {
            updateMetrics(stats);
            updateCharts(stats);
            updateHealthStatus(stats);
        }
    } catch (error) {
        console.error('Error loading dashboard:', error);
    }
}

// Update key metrics
function updateMetrics(stats) {
    const supportStats = stats.support_system;
    
    document.getElementById('total-queries').textContent = supportStats.total_queries;
    document.getElementById('avg-response-time').textContent = `${supportStats.avg_response_time.toFixed(2)}s`;
    document.getElementById('success-rate').textContent = `${(supportStats.success_rate * 100).toFixed(1)}%`;
    document.getElementById('total-tokens').textContent = supportStats.total_tokens;
}

// Update charts
function updateCharts(stats) {
    updateIntentChart(stats.support_system.intent_distribution);
    updateResponseTimeChart(stats.support_system.avg_response_time);
}

// Update intent distribution chart
function updateIntentChart(intentDistribution) {
    const ctx = document.getElementById('intentChart').getContext('2d');
    
    if (intentChart) {
        intentChart.destroy();
    }
    
    const labels = Object.keys(intentDistribution);
    const data = Object.values(intentDistribution);
    const colors = ['#007bff', '#28a745', '#ffc107'];
    
    intentChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels.map(label => label.charAt(0).toUpperCase() + label.slice(1)),
            datasets: [{
                data: data,
                backgroundColor: colors,
                borderWidth: 2,
                borderColor: '#fff'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom'
                }
            }
        }
    });
}

// Update response time chart
function updateResponseTimeChart(avgResponseTime) {
    const ctx = document.getElementById('responseTimeChart').getContext('2d');
    
    if (responseTimeChart) {
        // Add new data point
        responseTimeChart.data.labels.push(new Date().toLocaleTimeString());
        responseTimeChart.data.datasets[0].data.push(avgResponseTime);
        
        // Keep only last 10 data points
        if (responseTimeChart.data.labels.length > 10) {
            responseTimeChart.data.labels.shift();
            responseTimeChart.data.datasets[0].data.shift();
        }
        
        responseTimeChart.update();
    } else {
        responseTimeChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [new Date().toLocaleTimeString()],
                datasets: [{
                    label: 'Average Response Time (s)',
                    data: [avgResponseTime],
                    borderColor: '#007bff',
                    backgroundColor: 'rgba(0, 123, 255, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Response Time (seconds)'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Time'
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
    }
}

// Update health status
function updateHealthStatus(stats) {
    const health = stats.health;
    const llmStats = stats.llm_wrapper;
    
    // Update health indicator
    const indicator = document.getElementById('health-indicator');
    const healthText = document.getElementById('health-text');
    const healthDetails = document.getElementById('health-details');
    
    if (health.system_status === 'healthy') {
        indicator.className = 'status-indicator healthy';
        healthText.textContent = 'Healthy';
        healthDetails.innerHTML = `
            <div class="row">
                <div class="col-6">
                    <small class="text-muted">Local Model</small>
                    <div>${health.llm_services.local_available ? 'Available' : 'Unavailable'}</div>
                </div>
                <div class="col-6">
                    <small class="text-muted">OpenAI</small>
                    <div>${health.llm_services.openai_available ? 'Available' : 'Unavailable'}</div>
                </div>
            </div>
        `;
    } else if (health.system_status === 'degraded') {
        indicator.className = 'status-indicator degraded';
        healthText.textContent = 'Degraded';
        healthDetails.innerHTML = `
            <div class="text-warning">
                <i class="fas fa-exclamation-triangle me-1"></i>
                Some services are unavailable
            </div>
        `;
    } else {
        indicator.className = 'status-indicator error';
        healthText.textContent = 'Error';
        healthDetails.innerHTML = `
            <div class="text-danger">
                <i class="fas fa-times-circle me-1"></i>
                System unavailable
            </div>
        `;
    }
    
    // Update LLM stats
    document.getElementById('local-success-rate').textContent = `${(llmStats.local_success_rate * 100).toFixed(1)}%`;
    document.getElementById('fallback-usage').textContent = llmStats.fallback_usage_count;
    document.getElementById('queue-size').textContent = llmStats.queue_size;
    document.getElementById('active-requests').textContent = llmStats.active_requests;
}

// Refresh dashboard
function refreshDashboard() {
    loadDashboard();
}

// Reset statistics
async function resetStats() {
    if (confirm('Are you sure you want to reset all statistics? This action cannot be undone.')) {
        try {
            const response = await fetch('/api/reset-stats', {
                method: 'POST'
            });
            
            const result = await response.json();
            
            if (response.ok) {
                alert('Statistics reset successfully');
                loadDashboard();
            } else {
                alert('Failed to reset statistics: ' + result.error);
            }
        } catch (error) {
            alert('Error resetting statistics: ' + error.message);
        }
    }
}

// Run quick evaluation
async function runQuickEvaluation() {
    if (confirm('Run a quick evaluation? This will test the system with a small sample of queries.')) {
        try {
            const response = await fetch('/api/evaluate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    mode: 'balanced',
                    samples: 3
                })
            });
            
            const result = await response.json();
            
            if (response.ok) {
                alert('Quick evaluation completed successfully!');
                // You could redirect to evaluation page or show results here
                window.location.href = '/evaluation';
            } else {
                alert('Evaluation failed: ' + result.error);
            }
        } catch (error) {
            alert('Error running evaluation: ' + error.message);
        }
    }
}

// Clean up on page unload
window.addEventListener('beforeunload', function() {
    if (updateInterval) {
        clearInterval(updateInterval);
    }
}); 