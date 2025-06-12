# main.py
# Entry point → runs FastAPI + Telegram Bot in parallel

import threading
import logging

from bot_app.telegram_bot import run_bot
from api_app.fastapi_app import run_fastapi

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    try:
        logging.info("🚀 Starting FastAPI server thread...")
        threading.Thread(target=run_fastapi, daemon=True).start()

        logging.info("🤖 Starting Telegram bot...")
        run_bot()

    except KeyboardInterrupt:
        logging.info("🛑 Stopping services (Ctrl+C detected).")
