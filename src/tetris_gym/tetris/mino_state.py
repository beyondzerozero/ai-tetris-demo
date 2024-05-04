import copy

import numpy as np

from .mino import Mino


class MinoState:
    def __init__(self, mino: Mino, height: int, width: int, origin: tuple):
        self.mino = mino
        self.height = height
        self.width = width
        self.origin = origin

    def __repr__(self) -> str:
        return f"MinoState(mino={self.mino}, origin={self.origin})"

    def __str__(self) -> str:
        return str(self.mino)

    def __eq__(self, other) -> bool:
        return self.mino == other.mino and self.origin == other.origin

    def __hash__(self) -> int:
        return hash((self.mino, self.origin))

    def rotate_left(self, field: np.array) -> bool:
        # 왼쪽 회전 (시계방향))
        prev_shape = copy.deepcopy(self.mino.shape)
        self.mino.shape = np.rot90(prev_shape)
        # invalid이면 rollback
        if self.is_invalid(field):
            self.mino.shape = prev_shape
            return False
        return True

    def rotate_right(self, field: np.array) -> bool:
        # 오른쪽 회전 (반시계방향)
        prev_shape = copy.deepcopy(self.mino.shape)
        self.mino.shape = np.rot90(prev_shape, -1)
        # invalid이면 rollback
        if self.is_invalid(field):
            self.mino.shape = prev_shape
            return False
        return True

    def move(self, dx: int, dy: int, field: np.array) -> bool:
        # 이동 
        prev_origin = copy.deepcopy(self.origin)
        self.origin = (prev_origin[0] + dx, prev_origin[1] + dy)
        # invalid이면 rollback
        if self.is_invalid(field):
            self.origin = prev_origin
            return False
        return True

    def is_invalid(self, field: np.array) -> bool:
        for i in range(self.mino.shape.shape[0]):
            for j in range(self.mino.shape.shape[1]):
                if self.mino.shape[i][j] == 0:
                    continue
                # 오프사이트 판단 
                if (
                    self.origin[0] + i < 0
                    or self.origin[0] + i >= self.height
                    or self.origin[1] + j < 0
                    or self.origin[1] + j >= self.width
                ):
                    return True
                # 중첩판정 
                if field[self.origin[0] + i][self.origin[1] + j] != 0:
                    return True
        return False

    def to_tensor(self) -> np.array:
        return np.array([self.mino.id, self.origin[0], self.origin[1]])
