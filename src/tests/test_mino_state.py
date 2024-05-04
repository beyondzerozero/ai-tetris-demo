import numpy as np
import pytest

from tetris_gym import Mino, MinoState, TetrisBoard

mock_mino_shape = np.array(
    [
        # J shape
        [0, 1, 0],
        [0, 1, 0],
        [1, 1, 0],
    ]
)
mock_mino_shape_2 = np.array(
    [
        # L shape
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 1],
    ]
)


class TestMinoState:
    def test_mino_rotate_left(self):
        mino = Mino(1, mock_mino_shape, "J")
        mino_state = MinoState(mino, 20, 10, (0, 0))
        field = TetrisBoard(20, 10, {mino})

        mino_state.rotate_left(field.board)  # 90 degrees
        assert np.array_equal(mino.shape, np.array([[0, 0, 0], [1, 1, 1], [0, 0, 1]]))
        mino_state.rotate_left(field.board)  # 180 degrees
        assert np.array_equal(mino.shape, np.array([[0, 1, 1], [0, 1, 0], [0, 1, 0]]))
        mino_state.rotate_left(field.board)  # 270 degrees
        assert np.array_equal(mino.shape, np.array([[1, 0, 0], [1, 1, 1], [0, 0, 0]]))
        mino_state.rotate_left(field.board)  # 0 degrees
        assert np.array_equal(mino.shape, np.array([[0, 1, 0], [0, 1, 0], [1, 1, 0]]))

    def test_mino_rotate_right(self):
        mino = Mino(1, mock_mino_shape, "J")
        mino_state = MinoState(mino, 20, 10, (0, 0))
        field = TetrisBoard(20, 10, {mino})

        mino_state.rotate_right(field.board)  # 270 degrees
        assert np.array_equal(mino.shape, np.array([[1, 0, 0], [1, 1, 1], [0, 0, 0]]))
        mino_state.rotate_right(field.board)  # 180 degrees
        assert np.array_equal(mino.shape, np.array([[0, 1, 1], [0, 1, 0], [0, 1, 0]]))
        mino_state.rotate_right(field.board)  # 90 degrees
        assert np.array_equal(mino.shape, np.array([[0, 0, 0], [1, 1, 1], [0, 0, 1]]))
        mino_state.rotate_right(field.board)  # 0 degrees
        assert np.array_equal(mino.shape, np.array([[0, 1, 0], [0, 1, 0], [1, 1, 0]]))

    def test_mino_move(self):
        mino = Mino(1, mock_mino_shape, "J")
        mino_state = MinoState(mino, 20, 10, (0, 0))
        field = TetrisBoard(20, 10, {mino})

        mino_state.move(1, 0, field.board)
        assert mino_state.origin == (1, 0)
        mino_state.move(0, 1, field.board)
        assert mino_state.origin == (1, 1)
        mino_state.move(-1, 0, field.board)
        assert mino_state.origin == (0, 1)
        mino_state.move(0, -1, field.board)
        assert mino_state.origin == (0, 0)

    def test_mino_is_invalid(self):
        mino = Mino(1, mock_mino_shape, "J")
        mino_state = MinoState(mino, 20, 10, (0, 0))
        field = TetrisBoard(20, 10, {mino})

        # 장외로 나가는 경우
        mino_state.move(-1, 0, field.board)
        assert mino_state.origin == (0, 0)
        mino_state.move(0, -1, field.board)
        assert mino_state.origin == (0, 0)

        # 180 degrees rotation
        mino_state.rotate_left(field.board)
        mino_state.rotate_left(field.board)

        # origin에 -1이 포함된 경우 
        mino_state.move(0, -1, field.board)
        assert mino_state.origin == (0, -1)

        # 회전에 의해 장외로 나가는 경우
        mino_state.rotate_left(field.board)
        assert np.array_equal(mino.shape, np.array([[0, 1, 1], [0, 1, 0], [0, 1, 0]]))
