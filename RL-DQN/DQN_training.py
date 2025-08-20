import numpy as np
from ai_llm import directions_are_valid
from game import Game,move
from DQN_tensorflow import DQN

def take_action(action,game):
    directions = ["up","down","left","right"]
    state_cpy=np.copy(game.state)
    game.state,game.score_add=move(directions[action],game.state)
    if not np.array_equal(state_cpy, game.state):
        game.generate()

def action_judge(game,action):
    directions = ["up", "down", "left", "right"]
    if game.if_win():
        reward=1000
        done=1
    elif game.if_lose():
        reward=-1000
        done=1
    else:
        reward=game.score_add if game.score_add!=0  else 1
        done=0

    directions_valid=directions_are_valid(game.state)
    if directions[action] not in directions_valid:
        reward-=50

    return reward,done

DQN_model_path = "models"

WIDTH = 4
HEIGHT = 4

action_size=4
EPISODES=100000
BATCH_SIZE=128
target_step=0
DIRECTIONS = ["up","down","left","right"]

if __name__=='__main__':
    game=Game()
    agent = DQN(WIDTH, HEIGHT, action_size, DQN_model_path)
    for episode in range(agent.episode_count, EPISODES):
        agent.episode_count+=1
        target_step = 0
        total_reward = 0
        while True:
            state = np.array(Game.transform_to_log2(game.state)).reshape(-1, WIDTH, HEIGHT, 1)[0]
            target_step += 1
            action=agent.choose_action(state)
            take_action(action,game)
            reward,done=action_judge(game,action)
            next_state=np.array(Game.transform_to_log2(game.state)).reshape(-1, WIDTH, HEIGHT, 1)[0]

            agent.store(state,action,reward,next_state,done)
            if len(agent.replay_buffer)>BATCH_SIZE:
                agent.train(BATCH_SIZE)

            total_reward += reward
            if done==1:
                break

        if episode % 100==0:
            agent.save_model()
        decay_rate = 0.9995
        agent.epsilon = max(0.01, agent.epsilon * decay_rate)

        print(game.state)
        print(f'episode {episode} done. Evaluated reward: {total_reward / target_step:.4f}')
        game.reset()