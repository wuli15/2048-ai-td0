import random
import re
import numpy as np
from openai import OpenAI

direction_pattern = re.compile(r"\b(up|down|left|right)\b", re.I)

client= OpenAI(
    api_key='',
    base_url="",
    timeout=5
)

def ai_without_type(dta, type):
    directions = directions_are_valid(dta)
    try:
        prompt = f'''你是一个专业的2048玩家。当前棋盘状态（0为空格）是：{dta}。请考虑以下策略来选择最有效的移动方向：
                    - 优先合并较大的数字，以加速向2048目标前进。
                    - 保持棋盘的单调性，例如将大数字集中在一角或一侧。
                    - 避免移动后导致棋盘阻塞，即没有足够的空间进行后续合并。
                    - 考虑未来几步的潜力，尽量为后续合并创造机会。
                    基于以上策略，请从{directions}中选择一个最优的移动方向。输出格式：方向: 方向，理由: [简要说明]'''
        response = client.chat.completions.create(
            model=type,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            #max_tokens=5
        )
        if match := direction_pattern.search(response.choices[0].message.content):
            return match.group(1).lower()
        result=ai_without_type(dta,'qwen-max')
        if result in directions:
            return ai_without_type(dta,'qwen-max')
        return random.choice(directions)
    except Exception as e:
        result = ai_without_type(dta, 'qwen-max')
        if result in directions:
            return ai_without_type(dta, 'qwen-max')
        return random.choice(directions)

def ai_llm(dta):
    result=ai_without_type(dta,'qwen-turbo')
    return result

def directions_are_valid(dta):
    state = np.array(dta)
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

if __name__ == '__main__':
    dta = [
        [2, 0, 4, 0],
        [0, 0, 4, 0],
        [0, 0, 4, 0],
        [2, 0, 4, 0]
    ]
    print(ai_llm(dta))