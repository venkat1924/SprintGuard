/**
 * SprintGuard PoC - Frontend Application
 * Handles all API interactions and UI updates
 */

// API Base URL
const API_BASE = '/api';

// Utility Functions
const Utils = {
    /**
     * Make API request with error handling
     */
    async apiRequest(endpoint, method = 'GET', data = null) {
        try {
            const options = {
                method: method,
                headers: {
                    'Content-Type': 'application/json'
                }
            };
            
            if (data) {
                options.body = JSON.stringify(data);
            }
            
            const response = await fetch(`${API_BASE}${endpoint}`, options);
            const result = await response.json();
            
            if (!response.ok) {
                throw new Error(result.error || 'Request failed');
            }
            
            return result.data;
        } catch (error) {
            console.error('API Error:', error);
            throw error;
        }
    },
    
    /**
     * Show element with fade-in animation
     */
    showElement(elementId) {
        const element = document.getElementById(elementId);
        if (element) {
            element.style.display = 'block';
            setTimeout(() => {
                element.style.opacity = '1';
            }, 10);
        }
    },
    
    /**
     * Hide element
     */
    hideElement(elementId) {
        const element = document.getElementById(elementId);
        if (element) {
            element.style.display = 'none';
        }
    },
    
    /**
     * Format date for display
     */
    formatDate(dateString) {
        const date = new Date(dateString);
        return date.toLocaleDateString('en-US', {
            weekday: 'long',
            year: 'numeric',
            month: 'long',
            day: 'numeric'
        });
    },
    
    /**
     * Get color based on score
     */
    getScoreColor(score) {
        if (score >= 80) return '#10b981';
        if (score >= 60) return '#3b82f6';
        if (score >= 40) return '#f59e0b';
        return '#ef4444';
    }
};

// ============================================================================
// Health Check Module
// ============================================================================
const HealthCheck = {
    async run() {
        const resultContainer = document.getElementById('health-check-result');
        resultContainer.innerHTML = '<div class="loading"></div> Analyzing data...';
        Utils.showElement('health-check-result');
        
        try {
            const data = await Utils.apiRequest('/health-check');
            this.renderResult(data);
        } catch (error) {
            resultContainer.innerHTML = `
                <div class="error">
                    <strong>Error:</strong> ${error.message}
                </div>
            `;
        }
    },
    
    renderResult(data) {
        const resultContainer = document.getElementById('health-check-result');
        
        const metricsHtml = `
            <div class="metric-card">
                <h4>Volume Score</h4>
                <div class="metric-score">${data.volume_score.toFixed(1)}</div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: ${data.volume_score}%; background-color: ${Utils.getScoreColor(data.volume_score)}"></div>
                </div>
            </div>
            <div class="metric-card">
                <h4>Completeness Score</h4>
                <div class="metric-score">${data.completeness_score.toFixed(1)}</div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: ${data.completeness_score}%; background-color: ${Utils.getScoreColor(data.completeness_score)}"></div>
                </div>
            </div>
            <div class="metric-card">
                <h4>Quality Score</h4>
                <div class="metric-score">${data.quality_score.toFixed(1)}</div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: ${data.quality_score}%; background-color: ${Utils.getScoreColor(data.quality_score)}"></div>
                </div>
            </div>
            <div class="metric-card">
                <h4>Consistency Score</h4>
                <div class="metric-score">${data.consistency_score.toFixed(1)}</div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: ${data.consistency_score}%; background-color: ${Utils.getScoreColor(data.consistency_score)}"></div>
                </div>
            </div>
        `;
        
        const recommendationsHtml = data.recommendations.map(rec => 
            `<li>${rec}</li>`
        ).join('');
        
        resultContainer.innerHTML = `
            <div style="text-align: center;">
                <div class="health-grade grade-${data.overall_grade}">${data.overall_grade}</div>
                <h3>Overall Score: ${data.overall_score.toFixed(1)}/100</h3>
                <p style="color: var(--text-secondary); margin-bottom: var(--spacing-lg);">
                    Based on ${data.story_count} historical stories
                </p>
            </div>
            
            <div class="metrics-grid">
                ${metricsHtml}
            </div>
            
            <div class="recommendations">
                <h4>📋 Recommendations:</h4>
                <ul>${recommendationsHtml}</ul>
            </div>
        `;
    }
};

// ============================================================================
// Risk Assessor Module
// ============================================================================
const RiskAssessor = {
    exampleStories: [
        "Implement OAuth2 authentication for the REST API endpoint",
        "Migrate user database from MySQL to PostgreSQL with zero downtime",
        "Add a button to toggle dark mode theme",
        "Integrate third-party payment gateway with webhook support",
        "Update copyright year in footer"
    ],
    
    fillExample() {
        const textarea = document.getElementById('story-description');
        const randomStory = this.exampleStories[
            Math.floor(Math.random() * this.exampleStories.length)
        ];
        textarea.value = randomStory;
    },
    
    async assess() {
        const description = document.getElementById('story-description').value.trim();
        const resultContainer = document.getElementById('risk-result');
        
        if (!description) {
            alert('Please enter a story description');
            return;
        }
        
        resultContainer.innerHTML = '<div class="loading"></div> Analyzing story...';
        Utils.showElement('risk-result');
        
        try {
            const data = await Utils.apiRequest('/assess-risk', 'POST', {
                description: description
            });
            this.renderResult(data);
        } catch (error) {
            resultContainer.innerHTML = `
                <div class="error">
                    <strong>Error:</strong> ${error.message}
                </div>
            `;
        }
    },
    
    renderResult(data) {
        const resultContainer = document.getElementById('risk-result');
        
        const similarStoriesHtml = data.similar_stories && data.similar_stories.length > 0
            ? `<p style="margin-top: var(--spacing-sm); color: var(--text-secondary);">
                Found ${data.similar_stories.length} similar historical stories
               </p>`
            : '';
        
        resultContainer.innerHTML = `
            <div>
                <span class="risk-badge risk-${data.risk_level}">
                    ${data.risk_level} Risk
                </span>
                
                <div class="confidence-indicator">
                    <strong>Confidence:</strong>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${data.confidence}%"></div>
                    </div>
                    <span style="margin-left: var(--spacing-sm); font-weight: 600;">
                        ${data.confidence.toFixed(1)}%
                    </span>
                </div>
                
                <div style="margin-top: var(--spacing-md);">
                    <h4>💡 Analysis:</h4>
                    <p style="line-height: 1.8; color: var(--text-primary);">
                        ${data.explanation}
                    </p>
                </div>
                
                ${similarStoriesHtml}
            </div>
        `;
    }
};

// ============================================================================
// Scope Simulator Module
// ============================================================================
const ScopeSimulator = {
    async simulate() {
        const endDate = document.getElementById('current-end-date').value;
        const currentPoints = parseInt(document.getElementById('current-points').value);
        const newPoints = parseInt(document.getElementById('new-points').value);
        const velocity = parseFloat(document.getElementById('team-velocity').value);
        const resultContainer = document.getElementById('scope-result');
        
        // Validation
        if (!endDate) {
            alert('Please select a sprint end date');
            return;
        }
        
        if (isNaN(currentPoints) || currentPoints < 0) {
            alert('Please enter valid current story points (>= 0)');
            return;
        }
        
        if (isNaN(newPoints) || newPoints < 1) {
            alert('Please enter valid new story points (>= 1)');
            return;
        }
        
        if (isNaN(velocity) || velocity <= 0) {
            alert('Please enter valid team velocity (> 0)');
            return;
        }
        
        resultContainer.innerHTML = '<div class="loading"></div> Simulating impact...';
        Utils.showElement('scope-result');
        
        try {
            const data = await Utils.apiRequest('/simulate-scope', 'POST', {
                current_end_date: endDate,
                current_story_points: currentPoints,
                new_story_points: newPoints,
                team_velocity: velocity
            });
            this.renderResult(data);
        } catch (error) {
            resultContainer.innerHTML = `
                <div class="error">
                    <strong>Error:</strong> ${error.message}
                </div>
            `;
        }
    },
    
    renderResult(data) {
        const resultContainer = document.getElementById('scope-result');
        
        const recommendationsHtml = data.recommendations.map(rec => 
            `<li>${rec}</li>`
        ).join('');
        
        resultContainer.innerHTML = `
            <div class="date-comparison">
                <div class="date-box">
                    <div class="label">Original End Date</div>
                    <div class="date">${Utils.formatDate(data.original_end_date)}</div>
                    <div style="margin-top: var(--spacing-xs); color: var(--text-secondary);">
                        ${data.original_points} points
                    </div>
                </div>
                
                <div class="date-arrow">→</div>
                
                <div class="date-box">
                    <div class="label">New End Date</div>
                    <div class="date" style="color: var(--danger-color);">
                        ${Utils.formatDate(data.new_end_date)}
                    </div>
                    <div style="margin-top: var(--spacing-xs); color: var(--text-secondary);">
                        ${data.new_points} points
                    </div>
                </div>
            </div>
            
            <div style="text-align: center;">
                <div class="impact-badge">
                    +${data.days_added} Business Day${data.days_added !== 1 ? 's' : ''} Delay
                </div>
            </div>
            
            <div style="margin-top: var(--spacing-md);">
                <h4>📊 Impact Summary:</h4>
                <p style="line-height: 1.8; color: var(--text-primary);">
                    ${data.impact_summary}
                </p>
            </div>
            
            <div class="recommendations">
                <h4>💡 Recommendations:</h4>
                <ul>${recommendationsHtml}</ul>
            </div>
        `;
    }
};

// ============================================================================
// Initialize
// ============================================================================
document.addEventListener('DOMContentLoaded', function() {
    console.log('SprintGuard PoC Loaded');
    
    // Set default date to 2 weeks from today
    const dateInput = document.getElementById('current-end-date');
    if (dateInput) {
        const twoWeeksFromNow = new Date();
        twoWeeksFromNow.setDate(twoWeeksFromNow.getDate() + 14);
        dateInput.value = twoWeeksFromNow.toISOString().split('T')[0];
    }
    
    // Auto-run health check on load
    HealthCheck.run();
});

