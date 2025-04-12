import os
import traceback

# Import the API creation function
try:
    print("Importing nba_player_matcher...")
    from nba_player_matcher import create_api, convert_numpy_types
    
    # Create the FastAPI application
    print("Creating FastAPI app...")
    app = create_api()
    
    # For debugging
    print(f"App type: {type(app)}")
    
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
            host="0.0.0.0",  # Bind to all available network interfaces
            port=port,
            log_level="info"
        )
except Exception as e:
    print(f"CRITICAL ERROR in main.py: {str(e)}")
    traceback.print_exc()