import json
import os
import torch
import spacy

# Importing necessary modules from aiogram
from aiogram import Bot, Dispatcher, types
from aiogram.utils import executor
from aiogram.utils.callback_data import CallbackData
from aiogram.dispatcher import FSMContext
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
from aiogram.types import CallbackQuery, ReplyKeyboardMarkup, KeyboardButton 

# Importing necessary modules from your project
from database import session, User, Base, engine, UserInfo
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy import insert
from transformers import AutoModelWithLMHead, AutoTokenizer, AutoModel, AutoModelForCausalLM
from spacy.tokens import Doc
from spacy.lang.ru.examples import sentences 

# Create tables in the database (if they don't already exist)
Base.metadata.create_all(bind=engine)

# Configure the SQLAlchemy registry
scoped_session(sessionmaker(bind=engine)).configure(bind=engine)

data_dir = "./data"

# Initializing the bot object with its API token
bot = Bot(token='TG_TOKEN')

# Initializing the dispatcher object to handle incoming messages
dispatcher = Dispatcher(bot)

# Load the Russian language model with NER
nlp = spacy.load('ru_core_news_lg')

# Initializing the tokenizer and model for ruDialoGPT-medium
tokenizer = AutoTokenizer.from_pretrained('tinkoff-ai/ruDialoGPT-medium', pad_token_id=50256, padding_side='left')
model = AutoModelForCausalLM.from_pretrained('tinkoff-ai/ruDialoGPT-medium')

# The function to get the response from the ruDialoGPT-medium model
async def get_response(user_message):
    user_id = str(user_message.chat.id)

    # If the user data file doesn't exist or doesn't contain the user's message, use the ruDialoGPT-medium model to generate a response
    input_ids = tokenizer.encode(user_message.text, return_tensors="pt")
    # create attention mask
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
    for i, id in enumerate(input_ids[0]):
        if id == tokenizer.pad_token_id:
            attention_mask[0][i] = 0
    generated_token_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        top_k=50,
        top_p=1,
        num_beams=1,
        num_return_sequences=1,
        do_sample=True,
        no_repeat_ngram_size=2,
        temperature=0.2,
        repetition_penalty=1.0,
        length_penalty=0.5,
        pad_token_id=50256,
        eos_token_id=50256,
        max_new_tokens=40
    )

    response = [tokenizer.decode(sample_token_ids) for sample_token_ids in generated_token_ids]
    response_str = ' '.join(response)

    # Extract the part between the first occurrences of @@ВТОРОЙ@@ and @@ПЕРВЫЙ@@
    start_idx = response_str.find('@@ВТОРОЙ@@')
    end_idx = response_str.find('@@ПЕРВЫЙ@@')
    if start_idx != -1 and end_idx != -1:
        response = response_str[start_idx+len('@@ВТОРОЙ@@'):end_idx]
    else:
        response = "I'm sorry, I don't know what to say."

    # Remove any extra spaces or characters
    response = response.strip()

    return response

# The function to handle the '/start' command
@dispatcher.message_handler(commands=['start'])
async def start_command(message: types.Message):
    
    user_id = message.chat.id

    # Check if the user already exists in the database
    user = session.query(User).filter(User.chat_id == user_id).first()
    if not user:
        # If the user doesn't exist, create a new entry in the database
        user = User(chat_id=user_id, username=message.chat.username)
        session.add(user)
        session.commit()

        # Send a welcome message
        await message.reply("Hi, I'm an AI language model. Send me a message and I'll try to continue the conversation!")
    else:
        # If the user already exists in the database, send a welcome back message
        prompt = f"Hi, I'm back in touch!"
        response = tokenizer.decode(model.generate(tokenizer.encode(prompt, return_tensors="pt"), max_length=50, do_sample=True)[0], skip_special_tokens=True)
        await message.reply(response)  

# The function to handle the '/stop' command
@dispatcher.message_handler(commands=['stop'])
async def stop_command(message: types.Message):
    user_id = str(message.chat.id)

    # Send a goodbye message
    await message.reply("Goodbye!")

# The function to handle incoming chat messages
@dispatcher.message_handler()
async def chat_command(message: types.Message):
    # Get the user from the database based on their chat ID
    user = session.query(User).filter_by(chat_id=message.chat.id).first()

    # Extract and store the user's name from the message
    await extract_and_store_name(message, user)

    # Pass the whole message object instead of just the text
    response = await get_response(message)

    if response is not None and any(len(r.strip()) > 0 for r in response):
        await message.reply(response)
    else:
        await message.reply("I'm sorry, I don't know what to say.")

async def extract_and_store_name(user_message, user):
    # Check if the user's name is already in the database
    if user.info and user.info.name:
        return
    
    # Check if the message contains a name marker
    keywords = ['Меня зовут', 'Мое имя', 'Я', 'меня']
    has_marker = any(keyword in user_message.text for keyword in keywords)

    # Extract the user's name from the message using NER
    # Extract the user's name from the message using NER
    name = None
    if has_marker:
        doc = nlp(user_message.text)
        for ent in doc.ents:
            if ent.label_ == 'PER':
                name = ent.text
                break
    else:
        name = None
    
    # If a name was found, store it in the database
    if name:
        if not user.info:
            user.info = UserInfo(chat_id=user.chat_id, name=name, age=None, country=None, interests=None)
        session.commit()

        
        user.info.name = name
        session.commit()

if __name__ == '__main__':
    # Start the bot
    executor.start_polling(dispatcher)