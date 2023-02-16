import telebot
from chatbot import ChatBot
import json

with open('param/var.json') as f:
    parameters = json.load(f)

token_ = parameters["token"]
book = parameters['book']
bot = telebot.TeleBot(token_)
book_path = book
var_path = 'param/var.json'
chatbot = ChatBot(book_path, var_path)

@bot.message_handler(commands=['start'])
def start(message):
    bot.send_message(chat_id=message.chat.id, text="Hello! I'm Marvin the depressed robot. How can I help you?")

@bot.message_handler(func=lambda message: True)
def echo(message):
    prompt = ""
    response = chatbot.get_response(prompt)
    bot.send_message(chat_id=message.chat.id, text=response)

def main():
    bot.polling()
