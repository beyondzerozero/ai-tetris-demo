from statistics import mean, median

import gymnasium as gym

from .ai import NN, NNPlayerController, NNTrainerController, WEIGHT_OUT_PATH
from .config import (
    ALL_HARDDROP_ACTIONS,
    HUMAN_CONTROLLER_ORDINARY_TETRIS_ACTIONS_INPUT_MAP,
    ORDINARY_TETRIS_ACTIONS,
    ORDINARY_TETRIS_MINOS,
    TETRIS_HEIGHT,
    TETRIS_WIDTH,
)
from .controller import HumanController


def overwrite_print(text, line):
    print("\033[{};0H\033[K{}".format(line + 1, text))


def start():
    env = gym.make(
        "tetris-v1",
        height=TETRIS_HEIGHT,
        width=TETRIS_WIDTH,
        minos=ORDINARY_TETRIS_MINOS,
        action_mode=0,
    )
    env.reset()
    done = False
    controller = HumanController(
        ORDINARY_TETRIS_ACTIONS,
        HUMAN_CONTROLLER_ORDINARY_TETRIS_ACTIONS_INPUT_MAP,
    )
    while not done:
        overwrite_print(env.render(), 0)
        action = controller.get_action(env)
        _, _, done, _, _ = env.step(action)
    # GameOver
    print(env.render())


def train_cuda():
    train("cuda")


def train_mps():
    train("mps")


def train(device="cpu"):
    print("Device: ", device)

    env = gym.make(
        "tetris-v1",
        height=TETRIS_HEIGHT,
        width=TETRIS_WIDTH,
        minos=ORDINARY_TETRIS_MINOS,
        action_mode=1,
    )
    env.reset()

    # input: 상태 특징량
    # output: 앞으로 보상 기대치 
    input_size = env.observation_space.shape[0]
    output_size = 1
    model = NN(input_size, output_size).to(device)
    controller = NNTrainerController(
        ALL_HARDDROP_ACTIONS,
        model,
        discount=1.00,
        epsilon=1.00,
        epsilon_min=0.001,
        epsilon_decay=0.999,
        device=device,
    )

    # 기존 parametor를 읽어들이는 경우 파일명 지정
    # model.load("param/NN4.weights.h5")

    running = True
    total_games = 0
    total_steps = 0
    while running:
        steps, rewards = controller.train(env, episodes=20)
        total_games += len(rewards)
        total_steps += steps
        model.save()  # 도중 경과 저장 

        print(env.render())
        print("==================")
        print("* Total Games: ", total_games)
        print("* Total Steps: ", total_steps)
        print("* Epsilon: ", controller.epsilon)
        print("*")
        print("* Average: ", sum(rewards) / len(rewards))
        print("* Median: ", median(rewards))
        print("* Mean: ", mean(rewards))
        print("* Min: ", min(rewards))
        print("* Max: ", max(rewards))
        print("==================")


def simulate():
    env = gym.make(
        "tetris-v1",
        height=TETRIS_HEIGHT,
        width=TETRIS_WIDTH,
        minos=ORDINARY_TETRIS_MINOS,
        action_mode=1,
    )
    env.reset()

    # input: 상태 특징량
    # output: 앞으로 보상 기대치
    input_size = env.observation_space.shape[0]
    output_size = 1
    model = NN(input_size, output_size)
    controller = NNPlayerController(ALL_HARDDROP_ACTIONS, model)

    # 기존 parametor를 읽은 경우, param 부하 파일명 지정 
    model.load(WEIGHT_OUT_PATH)

    running = True
    while running:
        action = controller.get_action(env)
        _, _, done, _, _ = env.step(action)
        print(overwrite_print(env.render(), 0))
        if done:
            _, _ = env.reset()
            input("Press Enter to continue...")


if __name__ == "__main__":
    train()
