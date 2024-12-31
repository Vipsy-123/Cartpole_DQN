import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# Neural Network for DQN
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Hyperparameters
env = gym.make("CartPole-v1", render_mode="human")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
learning_rate = 0.001
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 64
memory_size = 100000
episodes = 1000
target_update_freq = 10

# Experience Replay Memory
memory = deque(maxlen=memory_size)

# Initialize networks
policy_net = DQN(state_size, action_size)
target_net = DQN(state_size, action_size)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# Optimizer and Loss Function
optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# Helper Functions
def remember(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))

def act(state, epsilon):
    if np.random.rand() <= epsilon:
        return random.randrange(action_size)
    state = torch.FloatTensor(state).unsqueeze(0)
    with torch.no_grad():
        q_values = policy_net(state)
    return torch.argmax(q_values).item()

def replay():
    if len(memory) < batch_size:
        return None

    minibatch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*minibatch)

    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones)

    # Calculate Q-values
    q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q_values = target_net(next_states).max(1)[0]
    target_q_values = rewards + gamma * next_q_values * (1 - dones)

    # Update Policy Network
    loss = criterion(q_values, target_q_values.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

# Training Loop
for episode in range(episodes):
    state = env.reset()[0]
    total_reward = 0
    episode_loss = 0

    for t in range(500):
        action = act(state, epsilon)
        next_state, reward, done, _, _ = env.step(action)
        remember(state, action, reward, next_state, done)

        # Render the environment
        env.render()  # Add this line to see the CartPole in action


        state = next_state
        total_reward += reward

        # Perform a replay (train the model)
        loss = replay()
        if loss is not None:
            episode_loss += loss

        if done:
            break

    # Decay epsilon (for exploration vs exploitation)
    epsilon = max(epsilon_min, epsilon_decay * epsilon)

    # Update target network periodically
    if episode % target_update_freq == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # Print training progress
    if episode % 10 == 0:  # Print every 10 episodes
        print(f"Episode {episode}/{episodes}, Total Reward: {total_reward}, Loss: {episode_loss:.4f}, Epsilon: {epsilon:.4f}")

env.close()
