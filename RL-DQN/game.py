from pynput import keyboard
from numba import jit
import numpy as np
import random

class Game:
    def __init__(self):
        self.state = np.zeros((4, 4), dtype=int)
        self.generate_start()
        self.total_score=0
        self.score_add=0

    def generate_start(self):
        for _ in range(2):
            empty_cells = np.argwhere(self.state == 0)
            location = random.choice(empty_cells)
            self.state[location[0], location[1]] = 2

    def generate(self):
        empty_cells = np.argwhere(self.state == 0)
        if len(empty_cells) > 0:
            location = random.choice(empty_cells)
            self.state[location[0], location[1]] = np.random.choice([2, 4], p=[0.9, 0.1])

    def if_win(self):
        return np.any(self.state == 1024)

    def if_lose(self):
        if np.any(self.state == 0):
            return False
        if np.any(self.state[:, :-1] == self.state[:, 1:]):
            return False
        if np.any(self.state[:-1, :] == self.state[1:, :]):
            return False
        return True

    def process_move(self, direction):
        state_cpy = np.copy(self.state)
        self.state,score_add = move(direction,self.state)
        self.total_score+=score_add
        if not np.array_equal(self.state, state_cpy):
            self.generate()
            Game.print_state(self.state)
        return True

    def get_score(self):
        return self.total_score

    def reset(self):
        self.state = np.zeros((4, 4), dtype=int)
        self.total_score=0
        self.generate_start()

    @staticmethod
    def print_state(state):
        #os.system('cls')  # 清空控制台
        for row in state:
            print(' '.join(map(str, row)))
        print()

    @staticmethod
    def get_key():
        keys = []
        def on_press(key):
            nonlocal keys
            try:
                keys.append(key.char.upper())
            except AttributeError:
                pass
            return False
        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()
        if keys[0]=='W': return 'up'
        elif keys[0]=='S': return 'down'
        elif keys[0]=='A': return 'left'
        elif keys[0]=='D': return 'right'

    '''
    以下是ai部分的接口===============================================================================================================================================
    '''

    @staticmethod   #转换成log2并且归一化
    def transform_to_log2(dta):
        dta0 = (dta == 0).astype(int)
        dta_final = dta + dta0
        dta_log2=np.log2(dta_final)
        return dta_log2/np.max(dta_log2)

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

@jit(nopython=True)
def move(name,state):
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
            for i in range(4):  # 遍历每一行（反向）
                row = dta[i, :][::-1]  # 先反转行
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
            dta=dta.T
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
            dta=dta.T
        elif name == "down":
            dta=dta.T
            for i in range(4):  # 遍历每一行（反向）
                row = dta[i, :][::-1]  # 先反转行
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
            dta=dta.T
        return dta, score_add

if __name__ == "__main__":
    game = Game()
    Game.print_state(game.state)
    while True:
        key=game.get_key()
        game.process_move(key)