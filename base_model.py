from abc import ABC, abstractmethod

class Model(ABC):
    def __init__(self):
        ...

    @abstractmethod
    def generate(self, input_data, **kwargs):
        ...