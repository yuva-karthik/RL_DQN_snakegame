import tensorflow as tf
import numpy as np
import random as rm
from collections import deque

class DQNAgent:
    def __init__(self, model, gamma=0.99, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995, mem_size=2000):
        self.model = model
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=mem_size)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    def act(self,state):
        if rm.random() <= self.epsilon:
            return rm.randint(0,2)
        else:
            state = np.expand_dims(state,axis=0)
            q_vals = self.model(state)
            return np.argmax(q_vals[0])
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self , batch_size=64):
        if len(self.memory) < batch_size:
            return
        
        batch = rm.sample(self.memory,batch_size)
        states , actions , rewards , next_states , dones = zip(*batch)

        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)

        q_values = self.model(states)
        next_q_values = self.model(next_states)
        target_q_values = q_values.numpy()

        for i in range(batch_size):
            target_q_values[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i]) * (1 - dones[i])
        
        with tf.GradientTape() as tape:
            predictions = self.model(states)
            loss = tf.keras.losses.mean_squared_error(target_q_values,predictions)
        grads = tape.gradient(loss,self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads,self.model.trainable_variables))

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay