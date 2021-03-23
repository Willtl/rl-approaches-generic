from abc import ABC, abstractmethod


class BaseNetwork(ABC):
    @abstractmethod
    def feed(self):
        pass

    @abstractmethod
    def mutate(self):
        pass