# config.py
# Global configuration for the bot and API

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Telegram Bot Token
BOT_TOKEN = os.getenv("BOT_TOKEN")
if BOT_TOKEN is None:
    raise ValueError("BOT_TOKEN is not set. Please set it as environment variable or in .env file.")

# URL of the FastAPI service (used by Telegram bot to send screenshots)
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000/upload-screenshot")
