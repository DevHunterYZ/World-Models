import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# Hiperparametreler
EPISODES = 50
LEARNING_RATE = 1e-4
GAMMA = 0.99
BATCH_SIZE = 64
MEMORY_SIZE = 100000
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.995
TARGET_UPDATE = 10

# Eylem uzayını ayrıklaştırma
DISCRETE_ACTIONS = [
    np.array([0, 1, 0]),   # Gaz
    np.array([0, 0, 0.8]), # Fren
    np.array([-1, 0, 0]),  # Sol
    np.array([1, 0, 0]),   # Sağ
    np.array([0, 1, 0.8]), # Gaz + Fren (kayma)
    np.array([-1, 1, 0]),  # Sol + Gaz
    np.array([1, 1, 0]),   # Sağ + Gaz
    np.array([0, 0, 0]),   # Hiçbir şey yapma
]

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        
        # Çıkış boyutunu dinamik olarak hesaplayın
        self._create_linear_input(input_shape)
        
        self.fc = nn.Sequential(
            nn.Linear(self.linear_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
        
    def _create_linear_input(self, input_shape):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            dummy_output = self.features(dummy_input)
            self.linear_input_size = dummy_output.view(1, -1).size(1)
    
    def forward(self, x):
        x = x / 255.0  # Normalize edin
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
        
    def __len__(self):
        return len(self.memory)

def preprocess_state(state):
    # Durumu ön işleyin: Gri tonlamaya çevir ve yeniden boyutlandır
    state = state[0:84, :, :]  # Alt kısmı kes
    state = np.mean(state, axis=2).astype(np.uint8)  # Gri tonlama
    state = np.expand_dims(state, axis=0)  # Kanal boyutunu ekle
    return state

def select_action(state, policy_net, epsilon, num_actions):
    if random.random() < epsilon:
        return random.randrange(num_actions)
    else:
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = policy_net(state)
            return q_values.max(1)[1].item()

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = list(zip(*transitions))
    
    states = torch.FloatTensor(np.array(batch[0])).to(device)
    actions = torch.LongTensor(batch[1]).unsqueeze(1).to(device)
    rewards = torch.FloatTensor(batch[2]).unsqueeze(1).to(device)
    next_states = torch.FloatTensor(np.array(batch[3])).to(device)
    dones = torch.FloatTensor(batch[4]).unsqueeze(1).to(device)
    
    current_q_values = policy_net(states).gather(1, actions)
    next_q_values = target_net(next_states).max(1)[0].detach().unsqueeze(1)
    expected_q_values = rewards + (GAMMA * next_q_values * (1 - dones))
    
    loss = nn.MSELoss()(current_q_values, expected_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Cihaz ayarı (GPU kullanımı için)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ortamı oluştur
env = gym.make('CarRacing-v2', render_mode='human')

# Ağları oluştur
num_actions = len(DISCRETE_ACTIONS)
state_shape = (1, 84, 96)  # Ön işlenmiş durum şekli

policy_net = DQN(state_shape, num_actions).to(device)
target_net = DQN(state_shape, num_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
memory = ReplayMemory(MEMORY_SIZE)

epsilon = EPSILON_START

for episode in range(EPISODES):
    state, _ = env.reset()
    state = preprocess_state(state)
    total_reward = 0
    done = False
    
    while not done:
        action_idx = select_action(state, policy_net, epsilon, num_actions)
        action = DISCRETE_ACTIONS[action_idx]
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state_processed = preprocess_state(next_state)
        memory.push(state, action_idx, reward, next_state_processed, done)
        state = next_state_processed
        total_reward += reward
        
        optimize_model()
        
        if done:
            print(f"Episode {episode+1}/{EPISODES}, Reward: {total_reward:.2f}")
            break
    
    # Epsilon'ı azalt
    if epsilon > EPSILON_END:
        epsilon *= EPSILON_DECAY
    
    # Hedef ağı güncelle
    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

env.close()
