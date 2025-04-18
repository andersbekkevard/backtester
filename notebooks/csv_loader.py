class CsvLoader:
    def __init__(self, name: str, age: int) -> None:
        self.__name = name
        self.__age = age

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, name):
        self.__name = name
