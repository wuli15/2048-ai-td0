import numpy as np
import random
import os
from multiprocessing import Pool, Manager, cpu_count
import time

directions = ["up", "down", "left", "right"]
n_tuples = [(0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 6), (4, 5, 6, 8, 9, 10), (4, 5, 6, 7, 8, 9)]
EPISODES = 1000
NUM_PROCESSES = cpu_count() - 2  # 自动检测核心数，预留2个核心给系统


class Game:
    def __init__(self):
        self.reset()

    def generate(self):
        empty_cells = np.argwhere(self.state == 0)
        if len(empty_cells) > 0:
            location = random.choice(empty_cells)
            self.state[location[0], location[1]] = np.random.choice([2, 4], p=[0.9, 0.1])

    def if_lose(self):
        if np.any(self.state == 0):
            return False
        if np.any(self.state[:, :-1] == self.state[:, 1:]):
            return False
        if np.any(self.state[:-1, :] == self.state[1:, :]):
            return False
        return True

    def reset(self):
        self.state = np.zeros((4, 4), dtype=int)
        self.total_score = 0
        self.score_add = 0
        self.generate()
        self.generate()
        # 每次重置时初始化win_count
        self.win_count = {512: 0, 1024: 0, 2048: 0, 4096: 0, 8192: 0}

    @staticmethod
    def move(name, state):
        score_add = 0
        dta = np.copy(state)
        if name == "left":
            for i in range(4):
                row = dta[i]
                temp = np.zeros(4, dtype=np.int32)
                pos = 0
                last = -1
                for j in range(4):
                    if row[j] != 0:
                        if last == row[j]:
                            temp[pos - 1] *= 2
                            score_add += temp[pos - 1]
                            last = -1
                        else:
                            temp[pos] = dta[i, j]
                            last = temp[pos]
                            pos += 1
                dta[i, :] = temp
        elif name == "right":
            for i in range(4):
                row = dta[i, :][::-1]
                temp = np.zeros(4, dtype=np.int32)
                pos = 0
                last = -1
                for j in range(4):
                    if row[j] != 0:
                        if last == row[j]:
                            temp[pos - 1] *= 2
                            score_add += temp[pos - 1]
                            last = -1
                        else:
                            temp[pos] = row[j]
                            last = temp[pos]
                            pos += 1
                dta[i, :] = temp[::-1]
        elif name == "up":
            dta = dta.T
            for i in range(4):
                row = dta[i]
                temp = np.zeros(4, dtype=np.int32)
                pos = 0
                last = -1
                for j in range(4):
                    if row[j] != 0:
                        if last == row[j]:
                            temp[pos - 1] *= 2
                            score_add += temp[pos - 1]
                            last = -1
                        else:
                            temp[pos] = dta[i, j]
                            last = temp[pos]
                            pos += 1
                dta[i, :] = temp
            dta = dta.T
        elif name == "down":
            dta = dta.T
            for i in range(4):
                row = dta[i, :][::-1]
                temp = np.zeros(4, dtype=np.int32)
                pos = 0
                last = -1
                for j in range(4):
                    if row[j] != 0:
                        if last == row[j]:
                            temp[pos - 1] *= 2
                            score_add += temp[pos - 1]
                            last = -1
                        else:
                            temp[pos] = row[j]
                            last = temp[pos]
                            pos += 1
                dta[i, :] = temp[::-1]
            dta = dta.T
        return dta, score_add

    @staticmethod
    def transform_to_log2(dta):
        dta_log2 = np.zeros([4, 4], dtype=int)
        non_zero = dta != 0
        dta_log2[non_zero] = np.log2(dta[non_zero]).astype(int)
        return dta_log2

    @staticmethod
    def directions_are_valid(state):
        valid = set()
        mask = (state[:-1, :] == state[1:, :]) & (state[:-1, :] != 0)
        zero = (state[:-1, :] == 0) & (state[1:, :] != 0)
        if np.any(mask | zero):
            valid.add("up")
        mask = (state[:, 1:] == state[:, :-1]) & (state[:, 1:] != 0)
        zero = (state[:, 1:] == 0) & (state[:, :-1] != 0)
        if np.any(mask | zero):
            valid.add("right")
        mask = (state[1:, :] == state[:-1, :]) & (state[1:, :] != 0)
        zero = (state[1:, :] == 0) & (state[:-1, :] != 0)
        if np.any(mask | zero):
            valid.add("down")
        mask = (state[:, :-1] == state[:, 1:]) & (state[:, :-1] != 0)
        zero = (state[:, :-1] == 0) & (state[:, 1:] != 0)
        if np.any(mask | zero):
            valid.add("left")
        return list(valid)

    def estimate(self):
        max_val = np.max(self.state)
        if max_val >= 512: self.win_count[512] = 1
        if max_val >= 1024: self.win_count[1024] = 1
        if max_val >= 2048: self.win_count[2048] = 1
        if max_val >= 4096: self.win_count[4096] = 1
        if max_val >= 8192: self.win_count[8192] = 1


class RL:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.init_model()
        return cls._instance

    def init_model(self):
        self.model_path = 'models/model.npy'
        if os.path.exists(self.model_path):
            self.R_table = np.load(self.model_path, mmap_mode='r')
        else:
            self.R_table = np.zeros((4, 2 ** 24))
        self.isom = [
            [[0, 1, 2, 3, 4, 5], [0, 4, 8, 12, 1, 5], [3, 2, 1, 0, 7, 6], [12, 8, 4, 0, 13, 9], [12, 13, 14, 15, 8, 9],
             [3, 7, 11, 15, 2, 6], [15, 14, 13, 12, 11, 10], [15, 11, 7, 3, 14, 10]],
            [[0, 1, 2, 4, 5, 6], [0, 4, 8, 1, 5, 9], [3, 2, 1, 7, 6, 5], [12, 8, 4, 13, 9, 5], [12, 13, 14, 8, 9, 10],
             [3, 7, 11, 2, 6, 10], [15, 14, 13, 11, 10, 9], [15, 11, 7, 14, 10, 6]],
            [[4, 5, 6, 8, 9, 10], [1, 5, 9, 2, 6, 10], [7, 6, 5, 11, 10, 9], [13, 9, 5, 14, 10, 6], [8, 9, 10, 4, 5, 6],
             [2, 6, 10, 1, 5, 9], [11, 10, 9, 7, 6, 5], [14, 10, 6, 13, 9, 5]],
            [[4, 5, 6, 7, 8, 9], [1, 5, 9, 13, 2, 6], [7, 6, 5, 4, 11, 10], [13, 9, 5, 1, 14, 10], [8, 9, 10, 11, 4, 5],
             [2, 6, 10, 14, 1, 5], [11, 10, 9, 8, 7, 6], [14, 10, 6, 2, 13, 9]]]

    def calculate_value(self, dta) -> float:
        dta = (dta & 0xf).reshape(16)
        value = 0
        for tp in range(len(self.isom)):
            for iso in self.isom[tp]:
                index = 0
                for num in iso:
                    x = dta[num]
                    index = index << 4
                    index += x
                value += self.R_table[tp][index]
        return value

    def choose_action(self, state) -> str:
        best_action = 'up'
        best_reward = -np.inf
        valid_dirs = Game.directions_are_valid(state)

        if not valid_dirs:
            return random.choice(directions)

        for direction in valid_dirs:
            new_state, score_add = Game.move(direction, state)
            value = self.calculate_value(Game.transform_to_log2(new_state))
            reward = score_add + value
            if reward > best_reward:
                best_action = direction
                best_reward = reward
        return best_action


def run_episode(_):
    game = Game()
    agent = RL()
    game.reset()

    while True:
        action = agent.choose_action(game.state)
        game.state, score_add = Game.move(action, game.state)
        game.total_score += score_add
        game.generate()
        if game.if_lose():
            break

    game.estimate()
    return game.win_count


if __name__ == "__main__":
    print(f"Starting parallel processing with {NUM_PROCESSES} processes...")
    start_time = time.time()

    # 使用进程池并行执行每个episode
    with Pool(processes=NUM_PROCESSES) as pool:
        results = pool.map(run_episode, range(EPISODES))

    # 聚合所有结果
    win_count = {512: 0, 1024: 0, 2048: 0, 4096: 0, 8192: 0}
    for result in results:
        for key in win_count:
            win_count[key] += result[key]

    # 打印最终结果
    print(f"\nTotal episodes: {EPISODES}")
    print(f"Total time: {time.time() - start_time:.2f} seconds")
    print(f'512: {win_count[512] / (EPISODES / 100):.2f}%')
    print(f'1024: {win_count[1024] / (EPISODES / 100):.2f}%')
    print(f'2048: {win_count[2048] / (EPISODES / 100):.2f}%')
    print(f'4096: {win_count[4096] / (EPISODES / 100):.2f}%')
    print(f'8192: {win_count[8192] / (EPISODES / 100):.2f}%')