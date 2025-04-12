import os
import sys
import traceback

try:
    from nba_player_matcher import create_api, convert_numpy_types
    
    # Get the FastAPI application
    app = create_api()
    
    # Make sure app is not None
    if app is None:
        raise ValueError("ERROR: create_api() returned None instead of a FastAPI app instance")
        
    # This conditional ensures the server only starts when this file is run directly
    if __name__ == "__main__":
        import uvicorn
        
        # Get port from environment variable or default to 8000
        port = int(os.environ.get("PORT", 8000))
        
        # Print startup information
        print(f"Starting NBA Player Shot Profile Matcher API on port {port}")
        print(f"API documentation available at http://localhost:{port}/docs")
        
        # Start the API server with uvicorn
        uvicorn.run(
            app, 
            host="0.0.0.0",
            port=port,
            log_level="info"
        )
except Exception as e:
    print(f"CRITICAL ERROR initializing app: {str(e)}")
    traceback.print_exc()
    sys.exit(1)