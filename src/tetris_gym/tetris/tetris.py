import copy
import random
from collections import deque

import numpy as np

from tetris_project.config import EDGE_CHAR, VOID_CHAR

from .board import TetrisBoard
from .mino import Mino
from .mino_state import MinoState

WALL_WIDTH = 1
NEXT_MINO_NUM = 3
NEXT_MINO_LIST_WIDTH = 6
LINE_CLEAR_SCORE = [0, 100, 300, 500, 800]


class Tetris:
    def __init__(self, height: int, width: int, minos: set[Mino]) -> None:
        self.board = TetrisBoard(height, width, minos)
        self.minos = minos
        self.mino_permutation = deque()

        self.hold_used = False  # hold사용 
        self.hold_mino = MinoState(
            mino=Mino(0, np.array([[0]]), VOID_CHAR),
            height=height,
            width=width,
            origin=(0, 0),
        )

        self.pre_mino_state = None
        self.latest_clear_mino_state = None
        self.latest_clear_lines = 0
        self.line_total_count = 0
        self.score = 0

        # 초기상태 mino 생성
        self.current_mino_state = self._generate_mino_state()
        self.game_over = False

    def _generate_mino_state(self) -> MinoState:
        # len(permutation) < 7에서 다음 순서의 minoi추가 
        if len(self.mino_permutation) < 7:
            add_permutation = copy.deepcopy(list(self.minos))
            random.shuffle(add_permutation)
            for mino in add_permutation:
                self.mino_permutation.append(mino)

        selected_mino = self.mino_permutation.popleft()

        return MinoState(
            mino=selected_mino,
            height=self.board.height,
            width=self.board.width,
            origin=(0, self.board.width // 2 - selected_mino.shape.shape[1] // 2),
        )

    def hold(self) -> bool:
        if self.hold_used:
            return False
        self.hold_used = True
        self.pre_mino_state = copy.deepcopy(self.current_mino_state)

        if self.hold_mino.mino.id == 0:
            self.hold_mino = self.current_mino_state
            self.current_mino_state = self._generate_mino_state()
        else:  # swap
            self.hold_mino, self.current_mino_state = (
                self.current_mino_state,
                self.hold_mino,
            )
        return True

    def place(self) -> None:
        self.score += 1  # 설치할 수 있으면 +1점
        self.hold_used = False  # hold상태 reset
        self.board.set_mino(self.current_mino_state)  # mino를 board에 설치

        self.latest_clear_lines = self.board.clear_lines()  # Line지우기 
        self.pre_mino_state = copy.deepcopy(
            self.current_mino_state
        )  # 마지막 mino 지우기 
        if len(self.latest_clear_lines) > 0:
            self.latest_clear_mino_state = copy.deepcopy(
                self.current_mino_state
            )  # Line 삭제시 mino 저장
        self.line_total_count += len(self.latest_clear_lines)  # Line 소거수 누적
        self.score += LINE_CLEAR_SCORE[
            len(self.latest_clear_lines)
        ]  # Line지우기 점수 추가
        self.current_mino_state = self._generate_mino_state()  # 다음 mino 생성

        # Game Over 판정 
        for i in range(self.current_mino_state.mino.shape.shape[0]):
            for j in range(self.current_mino_state.mino.shape.shape[1]):
                if (
                    self.current_mino_state.mino.shape[i][j] == 1
                    and self.board.board[self.current_mino_state.origin[0] + i][
                        self.current_mino_state.origin[1] + j
                    ]
                    != 0
                ):
                    self.game_over = True

    def move_and_rotate_and_drop(self, y: int, rotate: int) -> bool:
        # (y좌표 변위, 회전횟수) -> 이동가능flag
        prev_state = copy.deepcopy(self.current_mino_state)
        flag = True
        # rotate
        while rotate > 0:
            flag = self.current_mino_state.rotate_left(self.board.board)
            rotate -= 1
            if not flag:
                self.current_mino_state = prev_state
                return False
        # move y
        while y != self.current_mino_state.origin[1]:
            if y < self.current_mino_state.origin[1]:
                flag = self.current_mino_state.move(0, -1, self.board.board)
            elif y > self.current_mino_state.origin[1]:
                flag = self.current_mino_state.move(0, 1, self.board.board)
            if not flag:
                self.current_mino_state = prev_state
                return False
        # drop
        while flag:
            flag = self.current_mino_state.move(1, 0, self.board.board)
        self.place()
        return True

    def observe(self) -> np.ndarray:
        return np.concatenate(
            [
                [
                    self.get_hole_count(),
                    self.get_above_block_squared_sum(),
                    self.get_latest_clear_mino_heght(),
                    self.get_row_transitions(),
                    self.get_column_transitions(),
                    self.get_bumpiness(),
                    self.get_eroded_piece_cells(),
                    self.get_cumulative_wells(),
                    self.get_aggregate_height(),
                ],
                self.current_mino_state.mino.to_tensor().flatten(),
                np.concatenate(
                    [mino.to_tensor().flatten() for mino in self.mino_permutation][
                        :NEXT_MINO_NUM
                    ]
                ),
                self.hold_mino.mino.to_tensor().flatten(),
            ]
        )

    def get_above_block_squared_sum(self) -> int:
        # ========== above_block_squared_sum ========== #
        # 빈 매스로 자신보다 상단에 있는 블록 수의 제곱합(직접 특징량)

        # get_hole_count와의 차별화
        # get_hole_count : 기본적으로 구멍이 없다는 것이 좋다는 상태를 표현 
        # above_block_squared_sum : 구멍이 있을때 구멍위에 블록이 없는 쪽이 복귀하기 쉬운 상태를 표현 
        res = 0
        for i in range(self.board.height):
            for j in range(self.board.width):
                if self.board.board[i][j] != 0:
                    continue
                cnt = 0
                for k in range(i - 1, -1, -1):
                    if self.board.board[k][j] != 0:
                        cnt += 1
                res += cnt**2
        return res

    def get_hole_count(self) -> int:
        # ========== hole_count ========== #
        # 빈 매스로 자신보다 상단의 블록매스 총수 
        res = 0
        for i in range(self.board.height):
            for j in range(self.board.width):
                if self.board.board[i][j] != 0:
                    continue
                for k in range(i - 1, -1, -1):
                    if self.board.board[k][j] != 0:
                        res += 1
                        break
        return res

    def get_latest_clear_mino_heght(self) -> int:
        # ========== latest_clear_mino_heght ========== #
        # 최근 Line 지우는 미노 높이 
        if self.latest_clear_mino_state is None:
            return 0
        return self.board.height - self.latest_clear_mino_state.origin[0]

    def get_row_transitions(self) -> int:
        # ========== row_transitions ========== #
        # 각 행에서 블록 -> 빈 or 빈 -> 블록으로 변화하는 횟수
        res = 0
        for i in range(self.board.height):
            for j in range(self.board.width - 1):
                if (
                    (self.board.board[i][j] != 0) and (self.board.board[i][j + 1] != 0)
                ) or (
                    (self.board.board[i][j] == 0) and (self.board.board[i][j + 1] == 0)
                ):
                    res += 1
        return res

    def get_column_transitions(self) -> int:
        # ========== column_transitions ========== #
        # 각 열에서 블록 -> 빈 or 빈 -> 블록으로 변화하는 횟수
        res = 0
        for j in range(self.board.width):
            for i in range(self.board.height - 1):
                if (
                    (self.board.board[i][j] != 0) and (self.board.board[i + 1][j] != 0)
                ) or (
                    (self.board.board[i][j] == 0) and (self.board.board[i + 1][j] == 0)
                ):
                    res += 1
        return res

    def get_bumpiness(self) -> int:
        # ========== bumpiness ========== #
        # 높이 차이 (변화)의 합 
        res = 0
        prev_height = self.board.height
        for j in range(self.board.width):
            height = 0
            for i in range(self.board.height):
                if self.board.board[i][j] != 0:
                    height = self.board.height - i
                    break
            if j != 0:
                res += abs(prev_height - height)
            prev_height = height
        return res

    def get_eroded_piece_cells(self) -> int:
        # ========== eroded_piece_cells ========== #
        # 최근 지워진 줄수 * 그것에 기여한 최근 미노 셀수 
        res = 0
        if self.latest_clear_mino_state is not None:
            for i in range(self.latest_clear_mino_state.mino.shape.shape[0]):
                for j in range(self.latest_clear_mino_state.mino.shape.shape[1]):
                    if (
                        self.latest_clear_mino_state.mino.shape[i][j] == 1
                        and self.latest_clear_lines.count(
                            self.latest_clear_mino_state.origin[0] + i
                        )
                        > 0
                    ):
                        res += 1
        return res

    def get_cumulative_wells(self) -> int:
        # ========== cumulatve_well ========== #
        # 좌우가 블록인 빈 매스에서 위로 k연속 빈 매스가 계속될 때 
        # well(i,j) = ∑_{i=1}^{k} i = k(k+1)/2
        # cumulatve_well = ∑ well(i,j)
        res = 0
        for j in range(self.board.width):
            for i in range(self.board.height - 1, -1, -1):
                well_flag = self.board.board[i][j] == 0
                well_flag &= j == 0 or self.board.board[i][j - 1] != 0
                well_flag &= (
                    j == self.board.width - 1 or self.board.board[i][j + 1] != 0
                )
                if well_flag:
                    k = 0
                    while i - k >= 0 and self.board.board[i - k][j] == 0:
                        k += 1
                    res += k * (k + 1) // 2
                    i -= k
        return res

    def get_aggregate_height(self) -> int:
        # ========== aggregate_height ========== #
        # aggregate_height = ∑_{j=1}^{w} height(j)
        res = 0
        for j in range(self.board.width):
            for i in range(self.board.height):
                if self.board.board[i][j] != 0:
                    res += self.board.height - i
                    break
        return res

    def render(self) -> str:
        all_fields = []
        s = EDGE_CHAR * (self.board.width + 2 * WALL_WIDTH)
        all_fields.append(s)

        for i in range(self.board.height):
            s = EDGE_CHAR
            for j in range(self.board.width):
                mino_x = i - self.current_mino_state.origin[0]
                mino_y = j - self.current_mino_state.origin[1]

                if self.board.board[i][j] in self.board.mino_id_map:
                    s += self.board.mino_id_map[self.board.board[i][j]].char
                elif (
                    0 <= mino_x < self.current_mino_state.mino.shape.shape[0]
                    and 0 <= mino_y < self.current_mino_state.mino.shape.shape[1]
                    and self.current_mino_state.mino.shape[mino_x][mino_y] == 1
                ):
                    s += self.current_mino_state.mino.char
                else:
                    s += VOID_CHAR
            s += EDGE_CHAR
            all_fields.append(s)

        s = EDGE_CHAR * (self.board.width + 2 * WALL_WIDTH)
        all_fields.append(s)

        # Next mino 생성 (4개까지)
        all_fields[0] += VOID_CHAR + "Ｎｅｘｔ" + VOID_CHAR
        now_line = 1
        for i in range(min(NEXT_MINO_NUM, len(self.mino_permutation))):
            all_fields[now_line] += VOID_CHAR * NEXT_MINO_LIST_WIDTH
            now_line += 1  # 빈 줄 

            for j in range(self.mino_permutation[i].shape.shape[0]):
                s = VOID_CHAR
                if self.mino_permutation[i].id == 4:
                    s += VOID_CHAR  # O shape의 경우 공백 추가 

                for k in range(self.mino_permutation[i].shape.shape[1]):
                    if self.mino_permutation[i].shape[j][k] == 1:
                        s += self.mino_permutation[i].char
                    else:
                        s += VOID_CHAR
                s += VOID_CHAR
                all_fields[now_line] += s
                now_line += 1

        # Next mino 생성 (4개까지)
        all_fields[now_line] += VOID_CHAR * NEXT_MINO_LIST_WIDTH
        now_line += 1  # 빈줄
        all_fields[now_line] += VOID_CHAR + "Ｈｏｌｄ" + VOID_CHAR
        now_line += 1
        all_fields[now_line] += VOID_CHAR * NEXT_MINO_LIST_WIDTH
        now_line += 1  # 빈줄 

        if self.hold_mino is not None:
            for i in range(self.hold_mino.mino.shape.shape[0]):
                s = VOID_CHAR
                if self.hold_mino.mino.id == 4:
                    s += VOID_CHAR
                for j in range(self.hold_mino.mino.shape.shape[1]):
                    if self.hold_mino.mino.shape[i][j] == 1:
                        s += self.hold_mino.mino.char
                    else:
                        s += VOID_CHAR
                s += VOID_CHAR
                all_fields[now_line] += s
                now_line += 1

        # 나머지 줄 채우기 
        while now_line < self.board.height + 2 * WALL_WIDTH:
            all_fields[now_line] += VOID_CHAR * NEXT_MINO_LIST_WIDTH
            now_line += 1

        # 화면하단 점수와 줄수 표시 
        s = VOID_CHAR + "Score " + VOID_CHAR + "Line" + VOID_CHAR * 11
        all_fields.append(s)
        s = (
            VOID_CHAR
            + f"{self.score:0>6}"
            + VOID_CHAR
            + f"{self.line_total_count:0>6}"
            + VOID_CHAR * 10
        )
        all_fields.append(s)

        # 下하단을 쉽게 볼 수 있도록 빈행 추가
        s = VOID_CHAR * (self.board.width + 2 * WALL_WIDTH)
        all_fields.append(s)

        s = ""
        for field in all_fields:
            s += field + "\n"
        return s
