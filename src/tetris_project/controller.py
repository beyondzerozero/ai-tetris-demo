import copy
from abc import ABC, abstractmethod
from typing import List, Mapping, Tuple

import numpy as np
from gymnasium import Env

from tetris_gym import Action


class Controller(ABC):
    def __init__(self, actions: set[Action]) -> None:
        self.actions = actions
        self.action_map = {action.id: action for action in actions}

    def get_possible_actions(self, env: Env) -> List[Tuple[Action, np.ndarray]]:
        # Env 정보를 보고 가능한 행동을 return (행동결정을 controller에 일임함)
        actions = []
        if env.unwrapped.action_mode == 0:
            # actions = [0, 1, 2, 3, 4, 5, 6]
            # ※ 현재 기계학습에 사용하지 않음 
            pass
        elif env.unwrapped.action_mode == 1:
            for action in self.actions:
                y, rotate, hold = action.convert_to_tuple(
                    env.unwrapped.tetris.board.width
                )
                if hold:
                    tetris_copy = copy.deepcopy(env.unwrapped.tetris)
                    if tetris_copy.hold():
                        actions.append((action, tetris_copy.observe()))
                    continue
                tetris_copy = copy.deepcopy(env.unwrapped.tetris)
                flag = tetris_copy.move_and_rotate_and_drop(y, rotate)
                if flag:
                    actions.append((action, tetris_copy.observe()))
        return actions

    @abstractmethod
    def get_action(self, env: Env) -> Action:
        pass


class HumanController(Controller):
    def __init__(self, actions: set[Action], input_map: Mapping[str, Action]) -> None:
        super().__init__(actions)
        self.input_map = input_map

    def get_action(self, _: Env) -> Action:
        while True:
            try:
                action_input = input("Enter action: ")
                action = self.input_map[action_input]
                return self.action_map[action.id]
            except KeyError:
                print("Invalid action")
