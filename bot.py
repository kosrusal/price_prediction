import asyncio
import logging
from aiogram import Bot, Dispatcher, types
from aiogram.filters.command import Command
from aiogram import F

import datetime

import backtest as bt

# Включаем логирование, чтобы не пропустить важные сообщения
logging.basicConfig(level=logging.INFO)
# Объект бота
bot = Bot(token="6880124270:AAH03PE_nwMLfkRoGzNfy3vvdJFoeoY62Do")
# Диспетчер
dp = Dispatcher()

# Хэндлер на команду /start
@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer("Hello!")

@dp.message(Command("Choose_company"))
async def choose_company(message: types.Message):
    kb = [
        [types.KeyboardButton(text="AAPL")]
        #[types.KeyboardButton(text="Без пюрешки")]
    ]
    keyboard = types.ReplyKeyboardMarkup(keyboard=kb, resize_keyboard=True, one_time_keyboard=True)
    await message.answer("Выберите актив", reply_markup=keyboard)
    
@dp.message(F.text.lower() == "aapl")
async def with_puree(message: types.Message):
    tod = datetime.datetime.now()
    d = datetime.timedelta(days = 24)
    a = tod - d
    model = bt.load_model("AAPL")
    await message.reply(str(bt.back_test(model, "AAPL", str(a)[:10], str(datetime.date.today()))))
    #await message.reply("База")

@dp.message(Command("Predict"))
async def predict(message: types.Message):
    pass


# Запуск процесса поллинга новых апдейтов
async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())