from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles

# Importing script functions
from Hashtag_Generation import generator
from Reddit import scrape_reddit
from Sentiment_Analysis import sentiment_analysis
from Forecast import arima

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with allowed frontend domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request model
class ActionRequest(BaseModel):
    topic: str

# Mount multiple static directories
app.mount("/forecast_static", StaticFiles(directory="/home/captain/Desktop/NLP_FISAC/Backend/Forecast"), name="image_static")
app.mount("/keyword_static", StaticFiles(directory="/home/captain/Desktop/NLP_FISAC/Backend/Hashtag_Generation"), name="keyword_static")

@app.post("/process")
async def process_action(request: ActionRequest):
    """
    Handles topic input, runs all the required scripts sequentially in the same process.
    """
    topic = request.topic.strip()

    if not topic:
        return JSONResponse(content={"status": "failed", "message": "Topic is required"}, status_code=400)

    # Run each script exactly as written, in sequence
    try:
        generator.run(topic)  # Runs hashtag generation
        #scrape_reddit.run()  # Runs Reddit scraping
        #sentiment_analysis.run()  # Runs sentiment analysis
        arima.run()  # Runs ARIMA forecast
    except Exception as e:
        return JSONResponse(content={"status": "failed", "error": str(e)}, status_code=500)

    return JSONResponse(content={"status": "completed", "message": "All scripts executed successfully"})
