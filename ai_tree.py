import math
import random
import numpy as np

directions=["left","up","right","down"]

class Node:
    def __init__(self,state,score,action,parent):
        self.state=state
        self.score=score
        self.parent=parent
        self.action=action
        self.children=[]
        self.visits=0
        self.total_score=0

    def expand(self):
        valid=self.directions_are_valid()
        explored=[child.action for child in self.children]
        unexplored=[d for d in valid if d not in explored]
        if unexplored and not Node.is_terminal(self.state):
            direction=random.choice(unexplored)
            new_state,new_score=self.move(direction,self.state,self.score)
            new_node=Node(new_state,new_score,direction,self)
            self.children.append(new_node)
            return new_node
        return None

    def directions_are_valid(self):
        state = self.state
        valid = set()

        mask = (state[:, :-1] == state[:, 1:]) & (state[:, :-1] != 0)
        zero = (state[:, :-1] == 0) & (state[:, 1:] != 0)
        if np.any(mask | zero):
            valid.add("left")

        mask = (state[:, 1:] == state[:, :-1]) & (state[:, 1:] != 0)
        zero = (state[:, 1:] == 0) & (state[:, :-1] != 0)
        if np.any(mask | zero):
            valid.add("right")

        mask = (state[:-1, :] == state[1:, :]) & (state[:-1, :] != 0)
        zero = (state[:-1, :] == 0) & (state[1:, :] != 0)
        if np.any(mask | zero):
            valid.add("up")

        mask = (state[1:, :] == state[:-1, :]) & (state[1:, :] != 0)
        zero = (state[1:, :] == 0) & (state[:-1, :] != 0)
        if np.any(mask | zero):
            valid.add("down")

        return list(valid)

    def update(self,score):
        self.visits+=1
        self.total_score+=score

    def best(self):
        ucts=[]
        for child in self.children:
            c=2
            if child.visits==0:
                ucts.append(float('inf'))
            else:
                ucts.append(child.total_score/child.visits+c*math.sqrt(math.log(self.visits)/child.visits))
        return self.children[np.argmax(ucts)]

    @staticmethod
    def is_terminal(state):
        if np.any(state == 0):
            return False
        if np.any(state[:, :-1] == state[:, 1:]):
            return False
        if np.any(state[:-1, :] == state[1:, :]):
            return False
        return True

    @staticmethod
    def move(name, state, score):
        dta = np.copy(state)
        if name == "left":
            for i in range(4):
                row = dta[i, :]
                row1 = row[row != 0].tolist()
                for j in range(len(row1) - 1):
                    if row1[j] == row1[j + 1]:
                        row1[j] *= 2
                        score += row1[j]
                        row1[j + 1] = 0
                row1 = [x for x in row1 if x != 0]
                row1 += [0] * (4 - len(row1))
                dta[i, :] = row1
        elif name == "right":
            for i in range(4):  # 遍历每一行（反向）
                row = dta[i, :][::-1]  # 先反转行
                row1 = row[row != 0].tolist()
                for j in range(len(row1) - 1):
                    if row1[j] == row1[j + 1]:
                        row1[j] *= 2
                        score += row1[j]
                        row1[j + 1] = 0
                row1 = [x for x in row1 if x != 0]
                row1 += [0] * (4 - len(row1))
                dta[i, :] = row1[::-1]
        elif name == "up":
            for j in range(4):
                col = dta[:, j]
                row1 = col[col != 0].tolist()
                for k in range(len(row1) - 1):
                    if row1[k] == row1[k + 1]:
                        row1[k] *= 2
                        score += row1[k]
                        row1[k + 1] = 0
                row1 = [x for x in row1 if x != 0]
                row1 += [0] * (4 - len(row1))
                dta[:, j] = row1
        elif name == "down":
            for j in range(4):
                col = dta[:, j][::-1]
                row1 = col[col != 0].tolist()
                for k in range(len(row1) - 1):
                    if row1[k] == row1[k + 1]:
                        row1[k] *= 2
                        score += row1[k]
                        row1[k + 1] = 0
                row1 = [x for x in row1 if x != 0]
                row1 += [0] * (4 - len(row1))
                dta[:, j] = row1[::-1]
        return dta, score

    @staticmethod
    def generate(state):
        kongge = np.argwhere(state == 0)
        if len(kongge) != 0:
            location = random.choice(kongge)
            state[location[0], location[1]] = np.random.choice([2, 4], p=[0.9, 0.1])
        return state

class MCTS:
    def __init__(self,state,score):
        self.root=Node(state,score,None,None)
        self.iteration_limit=300
        self.max_move=10

    def search(self):
        for _ in range(self.iteration_limit):
            node = self.select(self.root)
            new_node = node.expand()
            if new_node:
                score = self.simulate(new_node.state, new_node.score)
                self.back(new_node, score)
            else:
                score = self.simulate(node.state, node.score)
                self.back(node, score)
        return self.root.best().action

    def select(self,node):
        while True:
            if Node.is_terminal(node.state):
                return node
            valid = node.directions_are_valid()
            expanded = [child.action for child in node.children]
            unexpanded = [a for a in valid if a not in expanded]
            if unexpanded:
                return node
            if not node.children:
                return node
            node = node.best()

    def simulate(self, state, score):
        step = 0
        while not Node.is_terminal(state) and step < self.max_move:
            state = Node.generate(state)
            directions_valid = [direction for direction in directions if
                                np.any(Node.move(direction, state, score)[0] != state)]
            if not directions_valid:
                break

            def best_move(state,score,directions_valid):
                scores=[]
                for direction in directions_valid:
                    scores.append(Node.move(direction, state, score)[1])
                return directions_valid[scores.index(max(scores))]

            state, score = Node.move(best_move(state,score,directions_valid), state, score)
            step += 1
        return score

    def back(self,node,score):
        while node:
            node.update(score/node.state.max()/35000)       #80%以上成功率
            node=node.parent