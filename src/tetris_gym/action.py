from typing import Tuple


class Action:
    def __init__(self, id: int, name="") -> None:
        self.id = id
        self.name = name

    # action_mode = 1용 id -> (y, rotate, hold) 변환 메소드
    def convert_to_tuple(self, width: int) -> Tuple[int, int, bool]:
        hold = self.id == ((width + 1) * 4)
        if hold:
            return 0, 0, True
        y = (self.id % (width + 1)) - 2
        rotate = self.id // (width + 1)
        return y, rotate, False
