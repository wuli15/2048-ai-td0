import numpy as np
import random
import os

directions=["up","down","left","right"]
n_tuples=[(0,1,2,3,4,5),(0,1,2,4,5,6),(4,5,6,8,9,10),(4,5,6,7,8,9)]
GAMMA=1

class RL:
    _instance = None  # 类变量保存单例

    def __new__(cls):
        """单例模式确保只加载一次,这个时候就不需要__init__了，不然还是会重复初始化。
        单例是确保全局只存在一个RL类，加上不__init__，可以省去重复加载"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.init_model()
        return cls._instance

    def init_model(self):
        self.model_path=r'RL\models\model.npy'
        if os.path.exists(self.model_path):
            self.R_table = np.load(self.model_path, mmap_mode='r')   #这里是内存映射，也可以改成全内存加载，都能达到1ms以内响应
        else:
            self.R_table=np.zeros((4,2**24))
        self.isom=[[[0, 1, 2, 3, 4, 5], [0, 4, 8, 12, 1, 5], [3, 2, 1, 0, 7, 6], [12, 8, 4, 0, 13, 9], [12, 13, 14, 15, 8, 9], [3, 7, 11, 15, 2, 6], [15, 14, 13, 12, 11, 10], [15, 11, 7, 3, 14, 10]],
                   [[0, 1, 2, 4, 5, 6], [0, 4, 8, 1, 5, 9], [3, 2, 1, 7, 6, 5], [12, 8, 4, 13, 9, 5], [12, 13, 14, 8, 9, 10], [3, 7, 11, 2, 6, 10], [15, 14, 13, 11, 10, 9], [15, 11, 7, 14, 10, 6]],
                   [[4, 5, 6, 8, 9, 10], [1, 5, 9, 2, 6, 10], [7, 6, 5, 11, 10, 9], [13, 9, 5, 14, 10, 6], [8, 9, 10, 4, 5, 6], [2, 6, 10, 1, 5, 9], [11, 10, 9, 7, 6, 5], [14, 10, 6, 13, 9, 5]],
                   [[4, 5, 6, 7, 8, 9], [1, 5, 9, 13, 2, 6], [7, 6, 5, 4, 11, 10], [13, 9, 5, 1, 14, 10], [8, 9, 10, 11, 4, 5], [2, 6, 10, 14, 1, 5], [11, 10, 9, 8, 7, 6], [14, 10, 6, 2, 13, 9]]]

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
        valid_directions = RL.directions_are_valid(state)
        best_action = None
        best_score = -np.inf
        for direction in valid_directions:
            new_state, reward = RL.move(direction, state)
            value = self.calculate_value(RL.transform_to_log2(new_state))
            score = reward + value
            if score > best_score:
                best_action = direction
                best_score = score
        return best_action

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
            dta = dta.T
        return dta, score_add

    @staticmethod  # 转换成log2
    def transform_to_log2(dta):
        dta_log2 = np.zeros([4, 4], dtype=int)
        non_zero = dta != 0
        dta_log2[non_zero] = np.log2(dta[non_zero]).astype(int)
        return dta_log2

    @staticmethod
    def directions_are_valid(state):
        valid = []
        top = state[:-1, :]
        bottom = state[1:, :]
        left = state[:, :-1]
        right = state[:, 1:]
        vertical_merge = (top == bottom) & (top != 0)
        horizontal_merge = (left == right) & (left != 0)
        if np.any(vertical_merge | ((top == 0) & (bottom != 0))):
            valid.append("up")
        if np.any(horizontal_merge | ((right == 0) & (left != 0))):
            valid.append("right")
        if np.any(vertical_merge | ((bottom == 0) & (top != 0))):
            valid.append("down")
        if np.any(horizontal_merge | ((left == 0) & (right != 0))):
            valid.append("left")
        return valid