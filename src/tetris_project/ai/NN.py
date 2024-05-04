import os
import random
from collections import deque
from pathlib import Path
from typing import List, Tuple

import numpy as np
from gymnasium import Env

import torch
import torch.nn as nn

from tetris_gym import Action
from tetris_gym.tetris import LINE_CLEAR_SCORE
from tetris_project.controller import Controller

WEIGHT_OUT_PATH = os.path.join(os.path.dirname(__file__), "out.pth")


class ExperienceBuffer:
    def __init__(self, buffer_size=20000):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, experience):
        # Replay Buffer에 (observe, action, reward, next_observe, done)을 추가함 
        self.buffer.append(experience)

    def sample(
        self, size: int
    ) -> List[Tuple[np.ndarray, int, float, np.ndarray, bool]]:
        idx = np.random.choice(len(self.buffer), size, replace=False)
        return [self.buffer[i] for i in idx]

    def len(self) -> int:
        return len(self.buffer)


class NN(nn.Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def save(self) -> None:
        torch.save(self.state_dict(), WEIGHT_OUT_PATH)

    def load(self, path: str) -> None:
        path = os.path.join(os.path.dirname(__file__), path)
        if Path(path).is_file():
            self.load_state_dict(torch.load(path))


class NNTrainerController(Controller):
    def __init__(
        self,
        actions: set[Action],
        model: nn.Module,
        discount=0.95,
        epsilon=0.50,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        device="cpu",
    ) -> None:
        super().__init__(actions)
        self.model = model
        self.discount = discount  # 할인율 
        self.epsilon = epsilon  # ε-greedy방법의 ε
        self.epsilon_min = epsilon_min  # ε-greedy방법의 ε 최소값 
        self.epsilon_decay = epsilon_decay  # ε-greedy방법의 ε 감쇠율

        # Experience Replay Buffer (상단하단 2개)
        self.lower_experience_buffer = ExperienceBuffer()
        self.upper_experience_buffer = ExperienceBuffer()

        self.device = device

    def get_action(self, env: Env) -> Action:
        possible_states = self.get_possible_actions(env)
        if random.random() < self.epsilon:  # ε-greedy 방법 
            return random.choice(possible_states)[0]
        else:  # 최적행동
            states = [state for _, state in possible_states]
            states_tensor = torch.tensor(np.array(states)).float().to(self.device)
            rating = self.model(states_tensor)
            action = possible_states[rating.argmax().item()][0]
            return action

    def train(self, env: Env, episodes=1):
        # 통계 
        rewards = []
        steps = 0

        for _ in range(episodes):
            state, _ = env.reset()
            done = False
            total_reward = 0
            while not done:
                action = self.get_action(env)  # 행동선택 (ε-greedy방법)
                next_state, reward, done, _, info = env.step(action)  # 행동실행
                if info["is_lower"]:
                    self.lower_experience_buffer.add(
                        (state, action, reward, next_state, done)
                    )
                else:
                    self.upper_experience_buffer.add(
                        (state, action, reward, next_state, done)
                    )

                if reward >= LINE_CLEAR_SCORE[4]:  # Line Clear일 때 
                    print("★★★★★★★★★★ 4 Line Clear! ★★★★★★★★★★")
                elif reward >= LINE_CLEAR_SCORE[3]:
                    print("★★★★★★★★★★ 3 Line Clear! ★★★★★★★★★★")
                elif reward >= LINE_CLEAR_SCORE[2]:
                    print("★★★★★★★★★★ 2 Line Clear! ★★★★★★★★★★")
                elif reward >= LINE_CLEAR_SCORE[1]:
                    print("★★★★★★★★★★ 1 Line Clear! ★★★★★★★★★★")

                state = next_state
                total_reward += reward
                steps += 1

            rewards.append(total_reward)
            self.learn()

        return [steps, rewards]

    def learn(self, batch_size=128, epochs=8):
        # 상하단으로 batch_size 개의 데이터를 가져옴 
        if (
            self.lower_experience_buffer.len() < batch_size // 2
            or self.upper_experience_buffer.len() < batch_size - batch_size // 2
        ):
            print("lower experience buffer size: ", self.lower_experience_buffer.len())
            print(
                "upper experience buffer size: ",
                self.upper_experience_buffer.len(),
                "\n",
            )
            return

        # 훈련 데이터 
        lower_batch = self.lower_experience_buffer.sample(batch_size // 2)
        upper_batch = self.upper_experience_buffer.sample(batch_size - batch_size // 2)
        all_batch = lower_batch + upper_batch

        # 현재와 다음 상태의 Q(s, a)를 모아 배치처리하여 효율화함 
        states = np.array([sample[0] for sample in all_batch])
        next_states = np.array([sample[3] for sample in all_batch])
        cancat_states_tensor = (
            torch.tensor(np.concatenate([states, next_states])).float().to(self.device)
        )
        all_targets = self.model(cancat_states_tensor)

        targets = all_targets[:batch_size]
        next_targets = all_targets[batch_size:]

        # batch에서 가장 높은 보상 기대치 Q(s, a)와 즉시보상 r을 표시함
        # idx: 가장 높은 보상 기대치 인덱스 
        idx = np.argmax([sample[2] for sample in all_batch])
        print(f"Immediate max reward in batch: {all_batch[idx][2]}")
        print(
            f"Action max value for the first sample in batch: {targets[idx].item()}\n"
        )

        # Q(s, a) 업데이트 
        for i, (_, _, reward, _, done) in enumerate(all_batch):
            targets[i] = reward
            if not done:
                targets[i] += self.discount * next_targets[i]

        targets_tensor = torch.tensor(targets).float().to(self.device)

        # 학습 
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        for _ in range(epochs):
            optimizer.zero_grad()
            states_tensor = torch.tensor(states).float().to(self.device)
            outputs = self.model(states_tensor)
            loss = criterion(outputs, targets_tensor)

            loss.backward()
            optimizer.step()

        # 학습후 다시 batch에서 가장 높은 보상 기대치 Q(s, a)를 표시함 (확인용)
        targets = self.model(torch.tensor(states).float().to(self.device))
        print(
            f"Action max value for the first sample in batch after learning: {targets[idx].item()}\n"
        )

        # 학습할 때마다 ε 감쇠 
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)


class NNPlayerController(Controller):
    def __init__(self, actions: set[Action], model) -> None:
        super().__init__(actions)
        self.model = model

    def get_action(self, env: Env) -> Action:
        possible_states = self.get_possible_actions(env)
        # 상태에서 최적 행동 선택 
        states = [state for _, state in possible_states]
        rating = self.model(torch.tensor(np.array(states)).float())
        action = possible_states[rating.argmax().item()][0]
        return action
