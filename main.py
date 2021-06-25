TOKEN = os.environ['TOKEN']  # Берем токен из переменной окружения, которую добавили ранее
WEBHOOK_HOST = 'https://styletransfer-bot-my.herokuapp.com'  # Здесь указываем https://<название_приложения>.herokuapp.com
WEBAPP_HOST = '0.0.0.0'  # Слушаем все подключения к нашему приложению
WEBAPP_PORT = os.environ.get('PORT')  # тк в Procfile мы указали process_type web, heroku сгенерирует нам нужный порт, его достаточно взять из переменной окружения

# ...........Imports..............
import os
import logging
from aiogram import Bot, Dispatcher, executor, types
from aiogram.dispatcher.filters import Command

from states import Test
from states import Settings
from states import Ep
from states import Trans
from vgg import main_train
from res_net import resnet_train
from img_resize import main_resize

from aiogram.dispatcher import FSMContext
from aiogram.contrib.fsm_storage.memory import MemoryStorage

import json

list_id = []
# ................Vars................

API_TOKEN = '1883714727:AAF55trIrs0bCLhI9Pio1l_c7OZ7rkz4P4U'
logging.basicConfig(level=logging.INFO)
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot, storage=MemoryStorage())


# .....................Start........................
@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    await message.answer("Привет \nЭто телеграмм бот, который умеет переносить стиль с одного изображения на "
                         "другое.\nИспользуй команду /help для того, чтобы узнать функционал бота")


# .......................Get_ready......................
@dp.message_handler(commands=['go'])
async def going(message: types.Message):
    with open('data_file.json', 'r') as j:
        json_data = json.load(j)

    if str(message.from_user.id) in json_data:
        pass
    else:
        json_data[str(message.from_user.id)] = {"closs_factor": 1, "sloss_factor": 1, "is_big": 0, "num_epochs": 300,
                                                "im_size": 256}

    with open('data_file.json', 'w') as outfile:
        json.dump(json_data, outfile)

    list_id.append(str(message.from_user.id))

    await message.answer("Готово! Теперь вам доступны коммарнды /transfer, /transfer_2, /set_param, /set_epochs, "
                         "/set_size_128, "
                         "/set_size_256, /set_size_512")


# .......................Help.........................
@dp.message_handler(commands=['help'])
async def helper(message: types.Message):
    await message.answer("/go - необходимо прописать, чтобы иметь доступ к коммандам transfer, "
                         "set_param, set_epochs, set_size_128, set_size_256, set_size_512. Создаёт место под "
                         "параметры пользователя в БД.\n"
                         "/info - информация о боте \n/set_param - "
                         "установка параметров. Сначала отправьте команду, затем введите 2 целых числа через пробел. "
                         "Первое отвечает за то, "
                         "на сколько выходное изображение должно соответсвовать данному при комманде transform, "
                         "второе - на сколько изображение должно соответсвовать стилю (например, 3 15, сильно большие "
                         "числа желательно тоже не вводить). Комманда не обязательна, с помощью неё можно лишь "
                         "эксперементально подбирать параметры для наилучшего выхода нейросети. \n/set_epochs - "
                         "установка колличества эпох. Сначала отправьте команду, затем введите целое число - "
                         "колличество эпох. По умолчанию их 300 \n/set_size_128 - установка размера "
                         "выходного изображения 128х128. Обратите внимание, что размеры стиля и основного изображения "
                         "должны быть больше, чем 128х128 \n/set_size_256 - установка размера "
                         "выходного изображения 256х256. Обратите внимание, что размеры стиля и основного изображения "
                         "должны быть больше, чем 256х256  \n/set_size_512 - установка размера "
                         "выходного изображения 512х512. Обратите внимание, что размеры стиля и основного изображения "
                         "должны быть больше, чем 512х512. Кроме того, сеть при таком параметре будет обучаться "
                         "заметно дольше. \n/transfer - перенос стилей. Сначала пропишите комманду и "
                         "отправьте её, "
                         "затем по очереди отправьте 2 изображения (основное и стиля соответственно). Выводится "
                         "изображение 128*128. Помните, что изображение стиля должно быть достаточно однотонным (как, "
                         "например, стилистические картины художников), иначе достойного результата не получится. "
                         "Также обратите внимание, что оба изображения должны быть с разрешением более, чем 128 х 128 "
                         "\n/transfer_2 - альтернативная архитектура, замена команде transfer. Используйте ту, "
                         "которая больше понравится. Данная архитектура ОЧЕНЬ капризная и неустойчивая, поэтому "
                         "советую использовать параметры (1 3), колличество эпох 100 "
                         "\n/examples - примеры "
                         "работы программы")


# ...............................Info.................................
@dp.message_handler(commands=["info"])
async def information(message: types.Message):
    await message.answer("@AlienLF - разработчик \nОсновой является модель сверточной сети VGG19 (transfer) и ResNet "
                         "(transfer_2) \nНейросеть была "
                         "написана написана с помощюь статьи, в которой все понятно и доходчиво объясняется - "
                         "https://pytorch.org/tutorials/advanced/neural_style_tutorial.html \nОтдельное спасибо "
                         "@Noath и @Dmitry_SmRn за помощь")


# ............................Set_params.............................
@dp.message_handler(commands=['set_param'], state=None)
async def setting_param(message: types.Message):
    if str(message.from_user.id) not in list_id:
        await message.answer("Введите комманду /go")
        pass
    else:
        await message.answer("Введите 2 параметра - целых числа - через пробел")
        await Settings.S1.set()


@dp.message_handler(state=Settings.S1)
async def setting_param_1(message: types.Message, state: FSMContext):
    with open('data_file.json', 'r') as j:
        json_data = json.load(j)

    print(message.text.split())
    try:
        c, s = message.text.split()[0], message.text.split()[1]
        c, s = int(c), int(s)
        if c <= 0 or s <= 0:
            await message.answer("Каждое число должно быть больше 0. Введите команду заново")
            await state.finish()
        else:
            json_data[str(message.from_user.id)]['closs_factor'] = c
            json_data[str(message.from_user.id)]['sloss_factor'] = s

            with open('data_file.json', 'w') as outfile:
                json.dump(json_data, outfile)
            await message.answer("Параметры изменены!")
            await state.finish()

    except ValueError:
        await message.answer("Введите 2 ЦЕЛЫХ числа. Наберите команду заново.")
        await state.finish()


# ..................................Set_epochs..........................
@dp.message_handler(commands=['set_epochs'], state=None)
async def setting_epochs(message: types.Message):
    if str(message.from_user.id) not in list_id:
        await message.answer("Введите комманду /go")
        pass
    else:
        await message.answer("Введите целое число - колличество эпох")
        await Ep.E1.set()


@dp.message_handler(state=Ep.E1)
async def setting_epochs_1(message: types.Message, state: FSMContext):
    with open('data_file.json', 'r') as j:
        json_data = json.load(j)

    try:
        e = int(message.text)
        if e <= 0:
            await message.answer("Каждое число должно быть больше 0. Введите команду заново.")
            await state.finish()
        elif e >= 5000:
            await message.answer("Слижком много! Введите команду заново")
            await state.finish()
        else:
            json_data[str(message.from_user.id)]['num_epochs'] = e

            with open('data_file.json', 'w') as outfile:
                json.dump(json_data, outfile)
            await message.answer("Параметры изменены!")
            await state.finish()
    except ValueError:
        await message.answer("Введите ЦЕЛОЕ число, наберите команду заново")
        await state.finish()


# ..............................Transfer..........................
@dp.message_handler(Command("transfer"), state=None)
async def enter_test(message: types.Message):
    if str(message.from_user.id) not in list_id:
        await message.answer("Введите комманду /go")
        pass
    else:
        await message.answer("Хорошо. Отправьте первое фото - основное. На него будет накладываться стиль.")
        await Test.Q1.set()


@dp.message_handler(content_types=['photo'], state=Test.Q1)
async def answer_q1(message: types.Message, state: FSMContext):
    await message.photo[-1].download(str(message.from_user.id) + 'content.jpg')
    await message.answer("Отлично. Теперь отправьте второе изображение. Это будет стиль, который налоржится на первое "
                         "изображение.")
    await Test.next()


@dp.message_handler(content_types=['photo'], state=Test.Q2)
async def answer_q2(message: types.Message, state: FSMContext):
    await message.photo[-1].download(str(message.from_user.id) + 'style.jpg')
    await message.answer("Принято! Ждите, пока нейросеть обучится и выдаст результат")
    main_resize(str(message.from_user.id))
    main_train(str(message.from_user.id))
    await bot.send_photo(message.from_user.id, photo=open((str(message.from_user.id) + 'output.jpg'), 'rb'))
    await state.finish()


# ......................transfer_2..............................
@dp.message_handler(Command("transfer_2"), state=None)
async def tr21(message: types.Message, ):
    if str(message.from_user.id) not in list_id:
        await message.answer("Введите комманду /go")
        pass
    else:
        with open('data_file.json', 'r') as j:
            json_data = json.load(j)
        if json_data[str(message.from_user.id)]["num_epochs"] >= 300 or not ((1/7) < (json_data[str(message.from_user.id)]["closs_factor"]/json_data[str(message.from_user.id)]["sloss_factor"]) < 2):
            await message.answer("Предупреждение! Неподходящие параметры. Введите команду ещё раз.")
        else:
            await message.answer("Хорошо. Отправьте первое фото - основное. На него будет накладываться стиль.")
            await Trans.T1.set()


@dp.message_handler(content_types=['photo'], state=Trans.T1)
async def tr22(message: types.Message, state: FSMContext):
    await message.photo[-1].download(str(message.from_user.id) + 'content.jpg')
    await message.answer("Отлично. Теперь отправьте второе изображение. Это будет стиль, который налоржится на первое "
                         "изображение.")
    await Trans.next()


@dp.message_handler(content_types=['photo'], state=Trans.T2)
async def tr23(message: types.Message, state: FSMContext):
    await message.photo[-1].download(str(message.from_user.id) + 'style.jpg')
    await message.answer("Принято! Ждите, пока нейросеть обучится и выдаст результат")
    main_resize(str(message.from_user.id))
    resnet_train(str(message.from_user.id))
    await bot.send_photo(message.from_user.id, photo=open((str(message.from_user.id) + 'output.jpg'), 'rb'))
    await state.finish()


# .............................set_size_128...........................
@dp.message_handler(commands=["set_size_128"])
async def setting_128(message: types.Message):
    if str(message.from_user.id) not in list_id:
        await message.answer("Введите комманду /go")
        pass
    else:

        with open('data_file.json', 'r') as j:
            json_data = json.load(j)

        json_data[str(message.from_user.id)]["im_size"] = 128

        with open('data_file.json', 'w') as outfile:
            json.dump(json_data, outfile)

        await message.answer("Изменено!")


# ..........................set_size_256........................
@dp.message_handler(commands=["set_size_256"])
async def setting_256(message: types.Message):
    if str(message.from_user.id) not in list_id:
        await message.answer("Введите комманду /go")
        pass
    else:

        with open('data_file.json', 'r') as j:
            json_data = json.load(j)

        json_data[str(message.from_user.id)]["im_size"] = 256

        with open('data_file.json', 'w') as outfile:
            json.dump(json_data, outfile)

        await message.answer("Изменено!")


# ................................set_size_512.....................
@dp.message_handler(commands=["set_size_512"])
async def setting_128(message: types.Message):
    if str(message.from_user.id) not in list_id:
        await message.answer("Введите комманду /go")
        pass
    else:
        with open('data_file.json', 'r') as j:
            json_data = json.load(j)
        json_data[str(message.from_user.id)]["im_size"] = 512
        with open('data_file.json', 'w') as outfile:
            json.dump(json_data, outfile)
        await message.answer("Изменено!")


# ..............................Examples.........................
@dp.message_handler(commands=['examples'])
async def show_example(message: types.Message):
    await message.answer("Пример работы /transfer")
    await bot.send_photo(message.from_user.id, photo=open('examples/vgg_1.jpg', 'rb'))
    await bot.send_photo(message.from_user.id, photo=open('examples/vgg_2.jpg', 'rb'))
    await message.answer("Пример работы /transfer_2")
    await bot.send_photo(message.from_user.id, photo=open('examples/resnet_1.jpg', 'rb'))
    await bot.send_photo(message.from_user.id, photo=open('examples/resnet_2.jpg', 'rb'))


# ................................Error..............................
@dp.message_handler()
async def error_command(message: types.Message):
    if message.text[0] == '/':
        await message.reply("Такой команды нет. Используйте /help для того, чтобы увидеть все команды.")


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
