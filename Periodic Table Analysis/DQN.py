# Deep Q-Network (DQN)
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
class DQNAgent:
    def __init__(self, input_size, output_size, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, lr=0.001):
        self.input_size = input_size
        self.output_size = output_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.lr = lr
        self.model = DQN(input_size, output_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.memory = []
        self.batch_size = 64
    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.output_size)
        with torch.no_grad():
            q_values = self.model(torch.tensor(state).float())
            return torch.argmax(q_values).item()
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.tensor(states).float()
        actions = torch.tensor(actions).unsqueeze(1)
        rewards = torch.tensor(rewards).float()
        next_states = torch.tensor(next_states).float()
        dones = torch.tensor(dones).bool()
        q_values = self.model(states)
        next_q_values = self.model(next_states)
        target_q_values = q_values.clone()
        with torch.no_grad():
            target_q_values[range(self.batch_size), actions.squeeze()] = rewards + self.gamma * next_q_values.max(dim=1)[0] * ~dones
        loss = self.criterion(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
periodic_table_data = pd.read_csv("Periodic_Table.csv").dropna()
state_space_size = len(periodic_table_data.columns) - 1  
action_space_size = 10  
agent = DQNAgent(input_size=state_space_size, output_size=action_space_size)
episodes = 100
for episode in range(episodes):
    state = periodic_table_data.sample().values[0][:-1]
    done = False
    total_reward = 0
    while not done:
        action = agent.select_action(state)
        next_state = periodic_table_data.sample().values[0][:-1]  
        reward = 1 if action == next_state[-1] else 0
        done = True 
        total_reward += reward
        agent.memory.append((state, action, reward, next_state, done))
        state = next_state
        agent.replay()
        agent.decay_epsilon()
    if episode % 1 == 0:
        print(f"Episode: {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon}")
