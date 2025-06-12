
# bot_app/telegram_bot.py
# Telegram Bot → configures handlers

import logging
import warnings
import telegram.error
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, CallbackQueryHandler, filters

from config import BOT_TOKEN
from bot_app.handlers import handle_photo, handle_command, handle_callback_query

# Create global Telegram bot application
app_bot = ApplicationBuilder().token(BOT_TOKEN).build()

# Register handlers
app_bot.add_handler(CommandHandler("start", handle_command))
app_bot.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, handle_photo))
app_bot.add_handler(CallbackQueryHandler(handle_callback_query))

# Run bot (called from main.py)
def run_bot():
    logging.info("🤖 Telegram Bot ready.")

    warnings.filterwarnings("ignore", category=RuntimeWarning, message="coroutine 'Application.shutdown' was never awaited")
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="coroutine 'Application.initialize' was never awaited")

    try:
        app_bot.run_polling()
    except KeyboardInterrupt:
        logging.info("🛑 Bot stopped by user (Ctrl+C)")
    except telegram.error.TimedOut:
        logging.info("🛑 Bot stopped with timeout (TimedOut on shutdown)")
    except Exception as e:
        logging.error(f"❌ Bot stopped with error: {e}")
