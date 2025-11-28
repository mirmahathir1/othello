from abc import ABC, abstractmethod

class Agent(ABC):
    @abstractmethod
    def get_move(self, game, verbose):
        pass
