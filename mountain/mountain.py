import torch

from enum import Enum
from typing import Tuple


class MountainOfDeathAction(Enum):
    LEFT = 0
    RIGHT = 1


class MountainOfDeath:
    """
    A simple symbolic environment.

    States:
    0, 1, ... , h , ... , 2h

    Player starts at state h and makes a move. Once the move has been made, the player
    is trapped on the path to either 0 or 2h+1. State 0 has no reward but is safe; state 2h+1
    has a reward but is unsafe.
    """

    def __init__(self, height=3):
        self._height = height
        self._starting_state = height
        self._state = self._starting_state

    def reset(self):
        self._state = self._starting_state

    def step(self, action: MountainOfDeathAction) -> Tuple[float, int]:
        if self._state > self._starting_state and self._state < 2 * self._height + 1:
            self._state += 1
        elif self._state < self._starting_state and self._state > 0:
            self._state -= 1
        else:
            if action == MountainOfDeathAction.LEFT and self._state > 0:
                self._state -= 1
            elif action == MountainOfDeathAction.RIGHT and self._state < 2 * self._height:
                self._state += 1
        if self._state == 2 * self._height:
            self.reset()
            print('Went right')
            return 5, 1
        elif self._state == 0:
            self.reset()
            print('Went left')
            return -10, 0
        else:
            return 0, 0

    def render(self):
        return torch.tensor([[self._state]]).float()

    def close(self):
        pass


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    g = MountainOfDeath()
    g.render()
    while True:
        action = input('Pls give an action: ')
        if action == 'exit':
            exit()
        reward = g.step(MountainOfDeathAction(int(action)))
        print(f'Received reward of {reward}')
        g.render()
