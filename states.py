from aiogram.dispatcher.filters.state import StatesGroup, State


class Test(StatesGroup):
    Q1 = State()
    Q2 = State()


class Settings(StatesGroup):
    S1 = State()


class Ep(StatesGroup):
    E1 = State()

class Trans(StatesGroup):
    T1 = State()
    T2 = State()
