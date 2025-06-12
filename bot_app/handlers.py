
# bot_app/handlers.py

import traceback
import logging
import json
import requests
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton, InputFile
from telegram.ext import ContextTypes
from config import FASTAPI_URL

from bot_app.handlers_solver_core import solve_and_format
from bot_app.utils import compute_effective_stack, pick_closest_nodes_folder

# Global state variables
last_final_json = None
last_json_per_chat = {}
last_uploaded_image_bytes_per_chat = {}
last_node_per_chat = {}

def normalize_hero_cards(hero_cards):
    normalized = []
    for card in hero_cards:
        if card.startswith("10"):
            normalized.append("T" + card[2])
        else:
            normalized.append(card)
    return normalized

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global last_final_json, last_json_per_chat, last_uploaded_image_bytes_per_chat
    try:
        file = None
        source_type = None

        if update.message.photo:
            file = await update.message.photo[-1].get_file()
            source_type = "Photo (compressed)"
        elif update.message.document and update.message.document.mime_type.startswith("image/"):
            file = await update.message.document.get_file()
            source_type = "Document (original)"
        else:
            await update.message.reply_text("⚠️ Please send an image (photo or document).")
            return

        file_bytes = await file.download_as_bytearray()
        logging.info(f"📥 Received {source_type}, size={len(file_bytes)} bytes")

        # Save original uploaded image
        last_uploaded_image_bytes_per_chat[update.effective_chat.id] = file_bytes
        logging.info(f"✅ Stored uploaded image for chat {update.effective_chat.id}")

        # Send to FastAPI
        logging.info("➡️ Sending image to FastAPI...")
        response = requests.post(
            FASTAPI_URL,
            files={"file": ("input_image.png", file_bytes)},
            timeout=10
        )

        logging.info(f"⬅️ Response from FastAPI: status_code={response.status_code}")

        if response.status_code == 200:
            # Success → Fetch last_final_json
            json_response = requests.get(FASTAPI_URL.replace("/upload-screenshot", "/last-json"))
            if json_response.ok:
                last_final_json = json_response.json()
                if "hero_cards" in last_final_json:
                    last_final_json["hero_cards"] = normalize_hero_cards(last_final_json["hero_cards"])

                # Basic sanity check
                button_present = "button" in last_final_json and last_final_json["button"] is not None
                num_stacks = len(last_final_json.get("stacks", []))

                if not button_present or num_stacks == 0:
                    logging.warning(f"⚠️ Invalid poker table: button_present={button_present}, num_stacks={num_stacks}")
                    await update.message.reply_text(
                        "❌ This doesn't look like a valid poker table screenshot."
                        "Please send a clear screenshot of your table (must contain Button and Stacks)."
                    )
                    return

                # Save last_final_json
                last_json_per_chat[update.effective_chat.id] = last_final_json
                logging.info("✅ Saved last_final_json for solver.")

            else:
                logging.warning(f"⚠️ Failed to get last_final_json: {json_response.status_code}")
                await update.message.reply_text("❌ Internal error while processing the image.")
                return

        else:
            # Error from FastAPI
            logging.warning(f"⚠️ Invalid screen received from FastAPI: {response.text}")
            await update.message.reply_text(
                "❌ This doesn't look like a valid poker table screenshot."
                "Please send a clear screenshot of your table."
            )
            return

        # Solver logic
        reply_text, last_node = solve_and_format(last_final_json)
        last_node_per_chat[update.effective_chat.id] = last_node

        # Buttons
        buttons = [
            [InlineKeyboardButton("📋 JSON", callback_data="show_json")],
            [InlineKeyboardButton("🖼 Image", callback_data="show_image")],
            [InlineKeyboardButton("🎯 Range", callback_data="show_range")]
        ]
        reply_markup = InlineKeyboardMarkup(buttons)

        await update.message.reply_text(reply_text, reply_markup=reply_markup)

    except Exception as e:
        logging.error(f"❌ Error processing image: {e}")
        await update.message.reply_text(f"❌ Internal error while processing the image: {e}")

async def handle_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global last_json_per_chat, last_uploaded_image_bytes_per_chat

    query = update.callback_query
    chat_id = query.message.chat_id
    await query.answer()
    logging.info(f"CallbackQuery: {query.data} (chat_id={chat_id})")

    if query.data == "show_json":
        if chat_id in last_json_per_chat:
            json_text = json.dumps(last_json_per_chat[chat_id], indent=2, ensure_ascii=False)
            if len(json_text) > 4000:
                json_text = json_text[:4000] + "... (truncated)"
            await query.message.reply_text(f"<pre>{json_text}</pre>", parse_mode="HTML")
        else:
            await query.message.reply_text("⚠️ No data found.")

    elif query.data == "show_image":
        if chat_id in last_json_per_chat:
            try:
                from poker_pipeline.draw_table import draw_table_from_json_pil
                import io

                image = draw_table_from_json_pil(last_json_per_chat[chat_id])

                buf = io.BytesIO()
                image.save(buf, format="JPEG", quality=85)
                buf.seek(0)

                await query.message.reply_photo(photo=buf, caption="🖼️ Here is what I see on your screen")
                logging.info(f"✅ Sent rendered image for chat {chat_id}")

            except Exception as e:
                tb = traceback.format_exc()
                logging.error(f"❌ Error rendering image: {e}{tb}")
                await query.message.reply_text(f"❌ Error rendering image:{e}{tb}")

        else:
            await query.message.reply_text("⚠️ No image data found.")

    elif query.data == "show_range":
        try:
            from poker_pipeline.draw_range import draw_range_from_node
            import io

            node_dict = last_node_per_chat.get(chat_id)

            if node_dict is None:
                await query.message.reply_text("⚠️ No last node available.")
                return

            image = draw_range_from_node(node_dict)

            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=85)
            buf.seek(0)

            await query.message.reply_photo(photo=buf, caption="🎯 Here is Range for this spot")
            logging.info(f"✅ Sent range image for chat {chat_id}")

        except Exception as e:
            tb = traceback.format_exc()
            logging.error(f"❌ Error rendering range: {e}{tb}")
            await query.message.reply_text(f"❌ Error rendering range:{e}{tb}")

async def handle_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 I am an AI-powered poker bot."
        "Send a screenshot (as document) to analyze the hand."
        "You'll get a solution + options: JSON, Range, or Image."
    )
