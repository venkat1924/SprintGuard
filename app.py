"""
SprintGuard PoC - Main Flask Application
Provides REST API for risk assessment, health checks, and scope simulation.
"""
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from config import DEBUG, HOST, PORT
from src.data_loader import get_data_loader
from src.analyzers import (
    HealthChecker,
    KeywordRiskAssessor,
    ScopeSimulator
)
from src.utils import format_success, format_error

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for API access

# Initialize data loader
data_loader = get_data_loader()

# Initialize analyzers with historical data
# NOTE: This is where the ML professor can plug in their implementation!
# Simply replace KeywordRiskAssessor with their custom class.
historical_stories = data_loader.get_all_stories()
risk_assessor = KeywordRiskAssessor(historical_stories)
scope_simulator = ScopeSimulator()


# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def index():
    """Serve the main web application"""
    return render_template('index.html')


@app.route('/api/health-check', methods=['GET'])
def health_check():
    """
    Endpoint: Data Health Check
    
    Analyzes the quality and quantity of historical data.
    Returns scores and recommendations for data improvement.
    """
    try:
        # Get all historical stories
        stories = data_loader.get_all_stories()
        
        # Perform health check
        checker = HealthChecker(stories)
        result = checker.assess()
        
        return jsonify(format_success(result.to_dict()))
    
    except Exception as e:
        return jsonify(format_error(f"Health check failed: {str(e)}", 500))


@app.route('/api/assess-risk', methods=['POST'])
def assess_risk():
    """
    Endpoint: Probabilistic Story Assessor
    
    Request Body:
        {
            "description": "User story description text"
        }
    
    Response:
        {
            "success": true,
            "data": {
                "risk_level": "High|Medium|Low",
                "confidence": 85.0,
                "explanation": "Detailed explanation...",
                "similar_stories": [1, 5, 12]
            }
        }
    """
    try:
        # Validate request
        data = request.get_json()
        
        if not data or 'description' not in data:
            return jsonify(format_error("Missing 'description' field in request body"))
        
        description = data['description'].strip()
        
        if not description:
            return jsonify(format_error("Description cannot be empty"))
        
        if len(description) < 3:
            return jsonify(format_error("Description is too short (minimum 3 characters)"))
        
        # Perform risk assessment
        result = risk_assessor.assess(description)
        
        return jsonify(format_success(result.to_dict()))
    
    except Exception as e:
        return jsonify(format_error(f"Risk assessment failed: {str(e)}", 500))


@app.route('/api/simulate-scope', methods=['POST'])
def simulate_scope():
    """
    Endpoint: Scope Impact Simulator
    
    Request Body:
        {
            "current_end_date": "2025-12-15",
            "current_story_points": 20,
            "new_story_points": 5,
            "team_velocity": 2.5  // Optional, defaults to 2.0
        }
    
    Response:
        {
            "success": true,
            "data": {
                "original_end_date": "2025-12-15",
                "new_end_date": "2025-12-17",
                "days_added": 2,
                "impact_summary": "...",
                "recommendations": [...]
            }
        }
    """
    try:
        # Validate request
        data = request.get_json()
        
        if not data:
            return jsonify(format_error("Request body is required"))
        
        # Required fields
        required_fields = ['current_end_date', 'current_story_points', 'new_story_points']
        missing_fields = [f for f in required_fields if f not in data]
        
        if missing_fields:
            return jsonify(format_error(
                f"Missing required fields: {', '.join(missing_fields)}"
            ))
        
        # Validate data types and values
        try:
            current_points = int(data['current_story_points'])
            new_points = int(data['new_story_points'])
        except ValueError:
            return jsonify(format_error("Story points must be integers"))
        
        if current_points < 0 or new_points < 1:
            return jsonify(format_error(
                "Current points must be >= 0 and new points must be >= 1"
            ))
        
        # Optional velocity parameter
        velocity = data.get('team_velocity', 2.0)
        try:
            velocity = float(velocity)
        except ValueError:
            return jsonify(format_error("Team velocity must be a number"))
        
        if velocity <= 0:
            return jsonify(format_error("Team velocity must be greater than 0"))
        
        # Perform simulation
        result = scope_simulator.simulate_scope_addition(
            current_end_date=data['current_end_date'],
            current_story_points=current_points,
            new_story_points=new_points,
            team_velocity=velocity
        )
        
        return jsonify(format_success(result.to_dict()))
    
    except ValueError as e:
        return jsonify(format_error(f"Invalid date format: {str(e)}"))
    except Exception as e:
        return jsonify(format_error(f"Simulation failed: {str(e)}", 500))


@app.route('/api/stories', methods=['GET'])
def get_stories():
    """
    Endpoint: Get historical stories (for debugging/exploration)
    
    Query Parameters:
        - risk_level: Filter by risk level (Low/Medium/High)
        - limit: Maximum number of stories to return
    """
    try:
        risk_level = request.args.get('risk_level')
        limit = request.args.get('limit', type=int)
        
        if risk_level:
            stories = data_loader.get_stories_by_risk_level(risk_level)
        else:
            stories = data_loader.get_all_stories()
        
        if limit:
            stories = stories[:limit]
        
        stories_dict = [story.to_dict() for story in stories]
        
        return jsonify(format_success({
            'stories': stories_dict,
            'count': len(stories_dict)
        }))
    
    except Exception as e:
        return jsonify(format_error(f"Failed to fetch stories: {str(e)}", 500))


@app.route('/api/info', methods=['GET'])
def get_info():
    """
    Endpoint: System information
    
    Returns information about the current risk assessor implementation.
    """
    try:
        story_count = data_loader.get_story_count()
        
        info = {
            'risk_assessor': risk_assessor.get_name(),
            'historical_story_count': story_count,
            'version': '1.0.0-PoC'
        }
        
        return jsonify(format_success(info))
    
    except Exception as e:
        return jsonify(format_error(f"Failed to get info: {str(e)}", 500))


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify(format_error("Resource not found", 404))


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify(format_error("Internal server error", 500))


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("SprintGuard PoC Starting...")
    print("=" * 60)
    print(f"Risk Assessor: {risk_assessor.get_name()}")
    print(f"Historical Stories: {len(historical_stories)}")
    print(f"Server: http://{HOST}:{PORT}")
    print("=" * 60)
    
    app.run(debug=DEBUG, host=HOST, port=PORT)

