from aiogram.types import ReplyKeyboardMarkup, KeyboardButton

btnMain = KeyboardButton('⬅️ Главное меню')

# --- Main Menu ---
btnRandom = KeyboardButton('FAQ')
btnOther = KeyboardButton('начать сначала')
mainMenu = ReplyKeyboardMarkup(resize_keyboard = True, one_time_keyboard=True).add(btnRandom, btnOther)


# --- Other Menu ---
btnInfo = KeyboardButton('')
btnMoney = KeyboardButton(' валют')
otherMenu = ReplyKeyboardMarkup(resize_keyboard = True).add(btnInfo, btnMoney, btnMain)
