"""Standardized API response formatting"""
from datetime import datetime
from typing import Any, Dict, Optional


def format_success(data: Any, message: Optional[str] = None) -> Dict:
    """
    Format a successful API response.
    
    Args:
        data: The response data
        message: Optional success message
        
    Returns:
        Standardized success response dictionary
    """
    response = {
        'success': True,
        'data': data,
        'error': None,
        'timestamp': datetime.utcnow().isoformat() + 'Z'
    }
    
    if message:
        response['message'] = message
    
    return response


def format_error(error_message: str, status_code: int = 400) -> tuple:
    """
    Format an error API response.
    
    Args:
        error_message: Description of the error
        status_code: HTTP status code (default: 400)
        
    Returns:
        Tuple of (error_dict, status_code)
    """
    response = {
        'success': False,
        'data': None,
        'error': error_message,
        'timestamp': datetime.utcnow().isoformat() + 'Z'
    }
    
    return response, status_code

