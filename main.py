import os
import uvicorn
from fastapi import FastAPI

# Import the API creation function
from nba_player_matcher import create_api

# Get the FastAPI application
app = create_api()

# This conditional ensures the server only starts when this file is run directly
if __name__ == "__main__":
    # Get port from environment variable or default to 8000
    port = int(os.environ.get("PORT", 8000))
    
    # Print startup information
    print(f"Starting NBA Player Shot Profile Matcher API on port {port}")
    print(f"API documentation available at http://localhost:{port}/docs")
    
    # Start the API server with uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0",  # Bind to all available network interfaces
        port=port,
        log_level="info"
    )