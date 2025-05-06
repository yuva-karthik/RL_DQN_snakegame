import numpy as np
import pygame
import matplotlib.pyplot as plt

from wrapped_snake_game import SnakeGame
from dqn_model import DQNModel
from dqn_agent import DQNAgent

env = SnakeGame()
model = DQNModel()
agent = DQNAgent(model)

epoch = 2000
batch_size = 64
steps_epoch = 500

plt.ion()
fig, ax = plt.subplots()
line, = ax.plot([], [], label='Reward')
ax.set_title("Training Progress")
ax.set_xlabel("Episode")
ax.set_ylabel("Total Reward")
ax.grid(True)
rewards = []

for e in range(epoch):
    state = env.reset()
    total_reward = 0
    done = False

    for step in range(steps_epoch):
        if e % 10 == 0:
            env.render()
            pygame.time.delay(30)

        action = agent.act(state)
        next_state, reward, done = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        agent.replay(batch_size)

        state = next_state
        total_reward += reward

        if done:
            print(f"Epoch: {e+1}, Reward: {total_reward}")
            break

    rewards.append(total_reward)

    line.set_data(range(len(rewards)), rewards)
    ax.set_xlim(0, max(10, len(rewards)))
    ax.set_ylim(min(rewards)-10, max(rewards)+10)
    plt.pause(0.01)

    if e % 100 == 0:
        model.save(f"saved_model/snake_model_{e}.keras")

env.close()
plt.ioff()
plt.show()