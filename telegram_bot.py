import logging

import telegram
import torch
import torchvision.transforms as transforms
from utils import *

from PIL import Image
from telegram import Update, ForceReply
from telegram.ext import Updater, CommandHandler, MessageHandler, CallbackContext, ContextTypes, filters, Application
from io import BytesIO

# Set up logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
model = load_model()
print("model loaded")

with open('TOKEN.txt', 'r') as file:
    my_token = file.read()

# help command
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    await update.message.reply_text("Help!")


# first start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    await update.message.reply_html(
        rf"Hi {user.mention_html()}!\n"
        rf"Send me an image of pizza steak or sushi",
        reply_markup=ForceReply(selective=True),
    )


async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Stores the photo and asks for a location."""
    user = update.message.from_user
    photo_file = await update.message.photo[-1].get_file()
    await photo_file.download_to_drive("user_photo/user_photo.jpg")
    img = Image.open("user_photo/user_photo.jpg")
    predicted_class = make_predict(img)
    user_answer = f"{random.choice(guessing_phrases)} {predicted_class}"
    await update.message.reply_text(user_answer)


# take image -> make predict -> send answer
# async def handle_image(update, context) -> None:
#     photo_file = await update.message.photo[-1].get_gile()
#     await photo_file.download_to_drive('user_photo.jpg')


    # predicted_class = make_predict(img)
    # user_answer = f"{random.choice(guessing_phrases)} {predicted_class}"
    # update.message.reply_text(user_answer)
    # await update.message.reply_text(user_answer)

# Answer user to send a image if he send a text
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text('Please, send an image')


# make predictions on AI model
def make_predict(img):
    model.eval()
    with torch.inference_mode():
        img = image_transform(img)
        img = torch.unsqueeze(img, 0)
        pred_logtis = model(img)
        pred_class = pred_logtis.argmax(dim=1)
        class_name = class_names[pred_class]

    return class_name


def main():
    """ Start the bot """
    # Create the Application and pass it your bot's token.
    application = Application.builder().token(my_token).build()

    # start/help commands handler
    application.add_handler(CommandHandler('start', start))
    application.add_handler((CommandHandler('help', help_command)))

    # answer to the message
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    application.add_handler(MessageHandler(filters.PHOTO, handle_image))

    # Run the bot until the user presses Ctrl-C
    application.run_polling()


if __name__ == '__main__':
    main()
