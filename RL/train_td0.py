from collections import deque
import numpy as np
import argparse
import random
import os

class TrainingConfig:
    def __init__(self,args):
        self.episodes=args.ep
        self.alpha=args.lr
        self.alpha_decay=args.lr_decay
        self.epsilon=args.epsilon
        self.gamma=args.g
        self.lamda=args.lam
        self.seed=args.s
        self.model_path=r'models\model.npy'
        self.progress_file = "models/training_progress.npy"

class Game:
    '''这里的训练中有reset，所以省略初始生成'''
    def __init__(self):
        self.state = np.zeros((4, 4), dtype=int)
        self.total_score=0

    def __str__(self):
        rows = []
        for row in range(4):
            cells = [f"{self.state[row, col]:>5}" for col in range(4)]
            row_str = f"│{'┊'.join(cells)}│"
            rows.append(row_str)
            if row < 3:
                sep = "├" + "┼".join(["┄" * 5 for _ in range(4)]) + "┤"
                rows.append(sep)
        top = "╭" + "┬".join(["─" * 5 for _ in range(4)]) + "╮"
        bottom = "╰" + "┴".join(["─" * 5 for _ in range(4)]) + "╯"
        return (f"{top}\n" + "\n".join(rows) + f"\n{bottom}")

    def generate(self):
        flat_indices = np.flatnonzero(self.state == 0)
        if flat_indices.size > 0:
            flat_idx = random.choice(flat_indices)
            row, col = np.unravel_index(flat_idx, self.state.shape)
            self.state[row, col] = 2 if random.random() < 0.9 else 4

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
        self.total_score=0
        self.generate()
        self.generate()

    @staticmethod
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

class RL:
    def __init__(self,config):
        self.model_path=config.model_path
        self.progress_file=config.progress_file
        self.alpha=config.alpha
        self.gamma=config.gamma
        self.lamda=config.lamda
        self.episode_count = 0
        self.trajectory=deque(maxlen=100000)
        self.isom=RL.tuple_isom([(0,1,2,3,4,5),(0,1,2,4,5,6),(4,5,6,8,9,10),(4,5,6,7,8,9)])

        if os.path.exists(self.model_path):
            print("model exists,loading saved model\n")
            self.R_table=np.load(self.model_path)
        else:
            print("model does not exist,creating new model\n")
            self.R_table = np.zeros((4, 2**24), dtype=np.float32)
            np.save(self.model_path, self.R_table)

        if os.path.exists(self.progress_file):
            self.episode_count = np.load(self.progress_file)
        else:
            np.save(self.progress_file, self.episode_count)

    @staticmethod
    def tuple_isom(n_tuples):
        isom = []
        dta = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]])
        dta_t = dta.T
        states = np.array(
            [dta, dta_t, np.flip(dta, axis=1), np.flip(dta_t, axis=1), np.flip(dta, axis=0), np.flip(dta_t, axis=0)
                , np.flip(np.flip(dta, axis=0), axis=1), np.flip(np.flip(dta, axis=0), axis=1).T]).reshape(8, -1)
        for tp in n_tuples:
            temp = []
            for state in states:
                t = [state[x] for x in tp]
                temp.append(t)
            isom.append(temp)
        return isom

    @staticmethod  # 转换成log2
    def transform_to_log2(dta):
        dta_log2 = np.zeros([4, 4], dtype=int)
        non_zero = dta != 0
        dta_log2[non_zero] = np.log2(dta[non_zero]).astype(int)
        return dta_log2

    '''
    传入的是取对数之后的数组,传出价值和索引表，索引表方便更新
    '''
    def calculate_value(self,dta)->(float,list):
        dta=(dta & 0xf).reshape(16)
        value=0
        index_lst=[]
        for tp in range(4):
            index_tp=[]
            for iso in self.isom[tp]:
                index = 0
                for num in iso:
                    x=dta[num]
                    index = index << 4
                    index += x
                value+=self.R_table[tp][index]
                index_tp.append(index)
            index_lst.append(index_tp)
        return value,index_lst

    '''传入的是原始数组'''
    def choose_action(self,state)->str:
        valid_directions=Game.directions_are_valid(state)
        best_action = None
        best_score=-np.inf
        for direction in valid_directions:
            new_state,reward=Game.move(direction,state)
            value,_=self.calculate_value(RL.transform_to_log2(new_state))
            score=reward+value
            if score > best_score:
                best_action=direction
                best_score=score
        return best_action

    '''
    传入的是取对数之后的数组
    循环里面用到的常数是具体计算出来的，但是为了省时间，改成直接用数值
    更新：传入了结束状态，用来纠正1024以下的情况
    '''
    def learn(self):
        target=0
        while self.trajectory:
            record=self.trajectory.pop()
            state=record[0]
            reward=record[1]
            old_value,index_lst=self.calculate_value(state)
            error=target-old_value
            for tp in range(4):
                for index in index_lst[tp]:
                    self.R_table[tp, index]+=self.alpha*(error/32)
            target=reward+(old_value+self.alpha*error)#后面括号里就是new_value

class Trainer():
    def __init__(self,game,agent,config):
        self.game=game
        self.agent=agent
        self.config=config
        self.win_count = {2 ** i: 0 for i in range(9, 14)}

    def train(self):
        try:
            for episode in range(self.agent.episode_count,self.config.episodes):
                self.game.reset()
                while True:
                    action = self.agent.choose_action(self.game.state)
                    self.game.state, score_add = Game.move(action, self.game.state)
                    self.game.total_score += score_add
                    after_state = np.copy(self.game.state)
                    self.game.generate()
                    self.agent.trajectory.append([RL.transform_to_log2(after_state), score_add])
                    if self.game.if_lose():
                        break
                self.estimate()
                self.agent.learn()

                self.agent.episode_count += 1
                if self.agent.episode_count % 10 == 0:
                    print(self.game)
                    print(f"episode:{self.agent.episode_count}, score:{self.game.total_score}\n")
                    if self.agent.episode_count % 100 == 0:
                        for i in range(9,14):
                            print(f'{2**i}:{self.win_count[2**i]}%')
                        print('\n')
                        self.win_count={2**i:0 for i in range(9,14)}
                    if self.agent.episode_count%10000==0:
                        self.save_model()
            self.save_model()

        except KeyboardInterrupt:
            print("\n检测到手动退出信号")
            print(f"保存当前训练进度，共完成{self.agent.episode_count}轮训练")
            self.save_model()
            print("模型已保存，程序退出")

    def save_model(self):
        temp_path = r'models\model_temp.npy'
        progress_temp = r"models\training_progress_temp.npy"
        try:
            np.save(temp_path, self.agent.R_table)
            np.save(progress_temp, self.agent.episode_count)
            os.replace(temp_path, self.agent.model_path)
            os.replace(progress_temp, self.agent.progress_file)
        except Exception as e:
            print(f"保存失败: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            if os.path.exists(progress_temp):
                os.remove(progress_temp)

    def estimate(self):
        for i in range(9,14):
            if np.max(self.game.state)>=2**i:self.win_count[2**i]+=1

def parse_arguments():
    parser = argparse.ArgumentParser(description="这是一个机器学习训练程序，支持设置训练轮数、学习率等参数")
    parser.add_argument('--ep', type=int, default=300000, help="训练总轮数（默认150000轮）", metavar="episodes")
    parser.add_argument('--lr', type=float, default=0.1, help="学习率（默认0.1）", metavar="alpha")
    parser.add_argument('--epsilon', type=float, default=0, help="ε-贪心（随时间衰减，默认不启用）", metavar="epsilon")
    parser.add_argument('--lr_decay', type=float, default=1, help="学习率衰减（默认1，即不启用动态衰减）", metavar="decay")
    parser.add_argument('--g', type=float, default=1, help="折扣因子", metavar="γ")
    parser.add_argument('--lam', type=float, default=1, help="衰减系数", metavar="λ")
    parser.add_argument('--s', type=int, default=15, help="随机数种子", metavar="random_seed")
    return parser.parse_args()

if __name__=='__main__':
    args = parse_arguments()
    config=TrainingConfig(args)
    game=Game()
    agent=RL(config)
    random.seed(config.seed)
    trainer=Trainer(game,agent,config)

    trainer.train()