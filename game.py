from PIL import Image, ImageTk
import numpy as np
from ai_llm import ai_llm
from ai_tree import MCTS
from ai_reinforce import RL
import tkinter as tk
import random
import ctypes
import os

colors = {
    0: '#F5EDE5',
    2: '#EDE0D4',
    4: '#E0D0C0',
    8: '#D4C0AC',
    16: '#C8B098',
    32: '#BCA084',
    64: '#B09070',
    128: '#A4805C',
    256: '#987048',
    512: '#8C6034',
    1024: '#805020',
    2048: '#74400C',
    4096: '#682000',
    8192: '#5C0000',
    16384:'#500000'
}

directions=["left","up", "right", "down" ]

class Game:
    best = 0
    def __init__(self, root):
        self.root = root
        self.root.title('2048')
        self.root.resizable(False, False)
        self.dpi_scaling = self.get_dpi_scaling()
        self.root.tk.call('tk', 'scaling', self.dpi_scaling)
        self.canvas = tk.Canvas(root, width=400, height=600, bg='#F5F5EF')
        self.canvas.pack()
        self.flag_continue = False
        self._btn_restart = tk.Button(
            root, text="重新开始", command=self.restart, bg="#cdc1b4", fg="white",
            relief="flat", highlightthickness=0, bd=5, width=10, height=2
        )
        self.btn_ai1 = tk.Button(
            root, text="贪心算法", command=self.ai_greed, bg="#cdc1b4", fg="white",
            relief="flat", highlightthickness=0, bd=5, width=10, height=2
        )
        self.btn_ai2 = tk.Button(
            root, text="蒙特卡洛", command=self.ai_mcts, bg="#cdc1b4", fg="white",
            relief="flat", highlightthickness=0, bd=5, width=10, height=2
        )
        # self.btn_ai3 = tk.Button(
        #     root, text="LLM", command=self.ai3, bg="#cdc1b4", fg="white",
        #     relief="flat", highlightthickness=0, bd=5, width=10, height=2
        # )
        self.btn_ai3 = tk.Button(
            root, text="强化学习", command=self.ai_rl, bg="#cdc1b4", fg="white",
            relief="flat", highlightthickness=0, bd=5, width=10, height=2
        )
        self.score = 0
        self.dta = np.zeros((4, 4), dtype=int)
        self.start()

    def draw(self):
        for i in range(4):
            for j in range(4):
                color = colors.get(self.dta[i, j])
                self.canvas.create_rectangle(100 * j, 100 * i, 100 * j + 100, 100 * i + 100, outline='#E5D9D0',
                                             fill=color, width=5)
                if self.dta[i, j] != 0:
                    self.canvas.create_text(100 * j + 50, 100 * i + 50, text=str(self.dta[i, j]), font=('Arial', 25),
                                            fill='white')
        self.canvas.create_text(15, 435, text=f'分数:{self.score}', font=('Montserrat', 15), fill='#3E2E23', anchor='w')
        self.canvas.create_text(15, 470, text=f'最高:{Game.best}', font=('Montserrat', 15), fill='#3E2E23',
                                anchor='w')

    def clear(self):
        items_to_delete = self.canvas.find_withtag("all")
        for item in items_to_delete:
            tags = self.canvas.gettags(item)
            if "keep" not in tags:
                self.canvas.delete(item)

    def start(self):
        self.flag_continue = False
        self.play()

    def restart(self):
        if self.score > Game.best:
            Game.best = self.score
        self.score = 0
        self.flag_continue = False
        self.btn_ai1.config(state='normal')
        self.btn_ai2.config(state='normal')
        self.btn_ai3.config(state='normal')
        self._btn_restart.config(state='normal')
        self.dta = np.zeros((4, 4), dtype=int)
        self.clear()
        self.draw()
        self.play()

    def cont(self):
        if self.score > Game.best:
            Game.best = self.score
        self.flag_continue = True
        self.btn_ai1.config(state='normal')
        self.btn_ai2.config(state='normal')
        self.btn_ai3.config(state='normal')
        self._btn_restart.config(state='normal')
        return self.score

    def get_dpi_scaling(self):
        """获取系统DPI缩放比例"""
        if os.name == 'nt':
            try:
                from ctypes import windll
                return windll.user32.GetDpiForSystem() / 96.0
            except:
                return 1.0
        return 1.0

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
        flat_indices = np.flatnonzero(state == 0)
        if flat_indices.size > 0:
            flat_idx = np.random.choice(flat_indices)
            row, col = np.unravel_index(flat_idx, state.shape)
            state[row, col] = 2 if np.random.random() <= 0.9 else 4
        return state

    def if_win(self):
        if self.flag_continue:
            return
        return np.any(self.dta == 2048)

    def if_lose(self):
        if np.any(self.dta == 0):
            return False
        if np.any(self.dta[:, :-1] == self.dta[:, 1:]):
            return False
        if np.any(self.dta[:-1, :] == self.dta[1:, :]):
            return False
        return True

    def run(self, event):
        key = event.keysym.lower()
        if key in directions:
            if self.if_lose() or (self.if_win() and not self.flag_continue):
                return
            dta_cpy = np.copy(self.dta)
            self.dta, self.score = Game.move(key, self.dta, self.score)
            if not np.array_equal(self.dta, dta_cpy):
                self.dta=Game.generate(self.dta)
                self.clear()
                self.draw()
            if self.if_win():
                Popup(self.root, self.score, self.restart, self.cont, self.flag_continue, "Win")
            if self.if_lose():
                Popup(self.root, self.score, self.restart, self.cont, self.flag_continue, "Win" if self.flag_continue else "Lose")

    def play(self):
        self.dta=Game.generate(self.dta)
        self.dta=Game.generate(self.dta)
        self.canvas.create_window(200, 450, window=self._btn_restart, tags=('keep',))
        self.canvas.create_window(100, 530, window=self.btn_ai1, tags=('keep',))
        self.canvas.create_window(200, 530, window=self.btn_ai2, tags=('keep',))
        self.canvas.create_window(300, 530, window=self.btn_ai3, tags=('keep',))
        self.draw()
        self.root.bind("<Key>", self.run)

    def evaluate(self):
        empty = np.sum(self.dta == 0)
        monotonicity = 0
        for i in range(4):
            row = self.dta[i, self.dta[i] != 0]
            if len(row) > 1:
                if np.all(row[:-1] <= row[1:]) or np.all(row[:-1] >= row[1:]):
                    monotonicity += np.sum(row)
        for j in range(4):
            col = self.dta[self.dta[:, j] != 0, j]
            if len(col) > 1:
                if np.all(col[:-1] <= col[1:]) or np.all(col[:-1] >= col[1:]):
                    monotonicity += np.sum(col)

        max1 = np.max(self.dta)

        maxposition = 0
        if self.dta[0, 0] == max1:
            maxposition += max1 * 10
            if self.dta[0, 1] != 0 and self.dta[0, 1] < max1:
                maxposition += self.dta[0, 1] * 3
            if self.dta[1, 0] != 0 and self.dta[1, 0] < max1:
                maxposition += self.dta[1, 0] * 3

        distribution = 0
        weights = np.array([
            [16, 15, 14, 13],
            [11, 12, 10, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])
        distribution = np.sum(self.dta * weights)

        totalscore = (empty + 2 * monotonicity + 3 * distribution + maxposition + 1) * self.score
        return totalscore

    '''贪心'''
    def ai_greed(self):
        self.btn_ai1.config(state='disabled')
        self.btn_ai2.config(state='disabled')
        self.btn_ai3.config(state='disabled')
        self._btn_restart.config(state='disabled')
        tag = False

        def trytomove():
            scores = []
            tryscore = self.score
            dta = np.copy(self.dta)
            for direction in directions:
                self.dta, self.score = Game.move(direction, self.dta, self.score)
                scores.append(self.evaluate() if not np.array_equal(self.dta, dta) else -99999999999999999999)
                self.dta = np.copy(dta)
                self.score = tryscore
            return directions[scores.index(max(scores))]

        if self.if_win():
            Popup(self.root, self.score, self.restart, self.cont, self.flag_continue, "Win")
            tag = True
        if self.if_lose():
            Popup(self.root, self.score, self.restart, self.cont, self.flag_continue, "Win" if self.flag_continue else "Lose")
            tag = True
        if Game.best < self.score:
            Game.best = self.score
        dta_cpy = np.copy(self.dta)
        self.dta, self.score = Game.move(trytomove(), self.dta, self.score)
        if not np.array_equal(self.dta, dta_cpy):
            self.dta=Game.generate(self.dta)
            self.clear()
            self.draw()
        if not tag:
            self.root.after(1, self.ai_greed)

    '''蒙特卡洛'''
    def ai_mcts(self):
        self.btn_ai1.config(state='disabled')
        self.btn_ai2.config(state='disabled')
        self.btn_ai3.config(state='disabled')
        self._btn_restart.config(state='disabled')
        tag = False
        if self.if_win():
            Popup(self.root, self.score, self.restart, self.cont, self.flag_continue, "Win")
            tag = True
        if self.if_lose():
            Popup(self.root, self.score, self.restart, self.cont, self.flag_continue, "Win" if self.flag_continue else "Lose")
            tag = True
        tree=MCTS(self.dta, self.score)
        self.dta,self.score=Game.move(tree.search(),self.dta, self.score)
        self.dta=Game.generate(self.dta)
        if Game.best < self.score:
            Game.best = self.score
        self.clear()
        self.draw()
        if not tag:
            self.root.after(1, self.ai_mcts)

    '''大语言模型（太慢，已隐藏）'''
    # def ai3(self):
    #     if hasattr(self, 'btn_ai1') and self.btn_ai1.winfo_exists():
    #         self.btn_ai1.config(state='disabled')
    #     if hasattr(self, 'btn_ai2') and self.btn_ai2.winfo_exists():
    #         self.btn_ai2.config(state='disabled')
    #     if hasattr(self, 'btn_ai3') and self.btn_ai3.winfo_exists():
    #         self.btn_ai3.config(state='disabled')
    #     if hasattr(self, '_btn_restart') and self._btn_restart.winfo_exists():
    #         self._btn_restart.config(state='disabled')
    #     if hasattr(self, 'btn_start') and self.btn_start.winfo_exists():
    #         self.btn_start.config(state='disabled')
    #
    #     self.loading_window = tk.Toplevel(self.root)
    #     self.loading_window.overrideredirect(True)
    #     self.loading_window.geometry('200x100+200+300')
    #     self.loading_window.attributes("-alpha", 0.75)
    #     tk.Label(
    #         self.loading_window, text="Loading...",
    #         font=("Arial", 10),
    #         bg="#F0F0F0"
    #     ).pack(expand=True)
    #     self.loading_window.attributes("-topmost", True)
    #     self.root.update_idletasks()
    #     self.root.update()
    #
    #     if self.if_win():
    #         Popup(self.root, self.score, self.restart, self.cont, self.flag_continue, "Win")
    #     if self.if_lose():
    #         Popup(self.root, self.score, self.restart, self.cont, self.flag_continue,"Win" if self.flag_continue else "Lose")
    #     self.dta, self.score = Game.move(ai_llm(self.dta), self.dta, self.score)
    #     self.dta = Game.generate(self.dta)
    #     if Game.best < self.score:
    #         Game.best = self.score
    #     self.clear()
    #     self.draw()
    #
    #     if hasattr(self, 'loading_window') and self.loading_window.winfo_exists():
    #         self.loading_window.destroy()
    #     if hasattr(self, 'btn_ai1') and self.btn_ai1.winfo_exists():
    #         self.btn_ai1.config(state='normal')
    #     if hasattr(self, 'btn_ai2') and self.btn_ai2.winfo_exists():
    #         self.btn_ai2.config(state='normal')
    #     if hasattr(self, 'btn_ai3') and self.btn_ai3.winfo_exists():
    #         self.btn_ai3.config(state='normal')
    #     if hasattr(self, '_btn_restart') and self._btn_restart.winfo_exists():
    #         self._btn_restart.config(state='normal')
    #     if hasattr(self, 'btn_start') and self.btn_start.winfo_exists():
    #         self.btn_start.config(state='normal')

    '''强化学习'''
    def ai_rl(self):
        self.btn_ai1.config(state='disabled')
        self.btn_ai2.config(state='disabled')
        self.btn_ai3.config(state='disabled')
        self._btn_restart.config(state='disabled')
        tag = False
        if self.if_win():
            Popup(self.root, self.score, self.restart, self.cont, self.flag_continue, "Win")
            tag = True
        if self.if_lose():
            Popup(self.root, self.score, self.restart, self.cont, self.flag_continue,
                  "Win" if self.flag_continue else "Lose")
            tag = True
        agent=RL()
        action=agent.choose_action(self.dta)
        self.dta, self.score = Game.move(action,self.dta, self.score)
        self.dta = Game.generate(self.dta)
        if Game.best < self.score:
            Game.best = self.score
        self.clear()
        self.draw()
        if not tag:
            self.root.after(1, self.ai_rl)

class Popup:
    def __init__(self,root,score,restart,cont,flag,condition):
        self.root = root
        self.popup = tk.Toplevel(root)
        self.popup.grab_set()
        self.popup.title("游戏胜利！" if condition=='Win' else '游戏失败……')
        self.popup.geometry("300x250+500+300")
        self.btn_restart = tk.Button(
            self.popup, text="重新开始", command=lambda: [restart(), self.close()], bg="#cdc1b4", fg="white",
            relief="flat", highlightthickness=0, bd=5, width=10, height=1
        )
        self.btn_continue = tk.Button(
            self.popup, text="继续游戏", command=lambda:[cont(),self.close()], bg="#cdc1b4", fg="white",
            relief="flat", highlightthickness=0, bd=5, width=10, height=1
        )
        self.btn_quit = tk.Button(
            self.popup, text="退出游戏", command=root.quit, bg="#cdc1b4", fg="white",
            relief="flat", highlightthickness=0, bd=5, width=10, height=1
        )
        self.label1 = tk.Label(self.popup, text="胜利", font=('Arial', 16))
        self.label2 = tk.Label(self.popup, text="失败", font=('Arial', 16))
        self.label3 = tk.Label(self.popup, text=f"分数: {score}", font=('Arial', 16))

        if condition=='Win':
            self.label1.pack(pady=15)
        else:
            self.label2.pack(pady=10)
        self.label3.pack(pady=15)
        if condition=='Win' and not flag:
            self.btn_continue.pack(pady=10)
        else:
            self.btn_restart.pack(pady=15)
        self.btn_quit.pack(pady=15)
        self.popup.protocol("WM_DELETE_WINDOW", lambda: [restart(), self.close()])

    def close(self):
        self.popup.grab_release()
        self.popup.destroy()

if __name__ == '__main__':
    root = tk.Tk()
    game = Game(root)
    root.mainloop()