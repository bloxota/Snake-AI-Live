import pygame
import random
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from gymnasium import spaces
import os

# ==============================
# Snake Environment
# ==============================
class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()
        self.grid_size = 20
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=1, shape=(8,), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.snake = [(self.grid_size//2, self.grid_size//2)]
        self.direction = (1, 0)
        self.food = self._spawn_food()
        self.done = False
        self.score = 0
        return self._get_obs(), {}

    def step(self, action):
        if action == 0: self.direction = (0, -1)
        elif action == 1: self.direction = (0, 1)
        elif action == 2: self.direction = (-1, 0)
        elif action == 3: self.direction = (1, 0)

        head_x, head_y = self.snake[0]
        dx, dy = self.direction
        new_head = (head_x + dx, head_y + dy)
        reward = 0

        if new_head[0] < 0 or new_head[0] >= self.grid_size or \
           new_head[1] < 0 or new_head[1] >= self.grid_size or \
           new_head in self.snake:
            self.done = True
            reward = -10
        else:
            self.snake.insert(0, new_head)
            if new_head == self.food:
                reward = 10
                self.score += 1
                self.food = self._spawn_food()
            else:
                old_dist = abs(head_x - self.food[0]) + abs(head_y - self.food[1])
                new_dist = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])
                reward = 1 if new_dist < old_dist else 0
                self.snake.pop()

        return self._get_obs(), reward, self.done, False, {}

    def _get_obs(self):
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        danger_up = 1 if (head_x, head_y-1) in self.snake or head_y-1 < 0 else 0
        danger_down = 1 if (head_x, head_y+1) in self.snake or head_y+1 >= self.grid_size else 0
        danger_left = 1 if (head_x-1, head_y) in self.snake or head_x-1 < 0 else 0
        danger_right = 1 if (head_x+1, head_y) in self.snake or head_x+1 >= self.grid_size else 0
        food_up = 1 if food_y < head_y else 0
        food_down = 1 if food_y > head_y else 0
        food_left = 1 if food_x < head_x else 0
        food_right = 1 if food_x > head_x else 0
        return np.array([danger_up,danger_down,danger_left,danger_right,
                         food_up,food_down,food_left,food_right], dtype=np.float32)

    def _spawn_food(self):
        while True:
            food = (random.randint(0,self.grid_size-1), random.randint(0,self.grid_size-1))
            if food not in self.snake:
                return food

# ==============================
# Setup Environment & Model
# ==============================
env = DummyVecEnv([lambda: SnakeEnv()])
MODEL_PATH = "snake_dqn_smart_live"

if os.path.exists(MODEL_PATH + ".zip"):
    model = DQN.load(MODEL_PATH, env=env)
    print("Schlaues Modell geladen")
else:
    model = DQN("MlpPolicy", env, verbose=1,
                learning_rate=0.001, buffer_size=5000,
                learning_starts=100, batch_size=32, gamma=0.95)
    print("Kein trainiertes Modell gefunden, erstes kurzes Training...")
    model.learn(total_timesteps=10000)
    model.save(MODEL_PATH)
    print("Modell gespeichert.")

# ==============================
# Pygame Setup
# ==============================
pygame.init()
cell_size = 30
grid_size = 20
window_size = grid_size*cell_size
screen = pygame.display.set_mode((window_size, window_size))
pygame.display.set_caption("Snake AI Smart Live")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 24)

# ==============================
# Main Loop
# ==============================
obs = env.reset()
highscore = 0
running = True
speed = 10  # normale Geschwindigkeit

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # KI Aktion
    action,_ = model.predict(obs, deterministic=True)
    obs,reward,dones,infos = env.step(action)
    done = dones[0]
    snake_env = env.envs[0]

    # Highscore aktualisieren
    if snake_env.score > highscore:
        highscore = snake_env.score

    # Reset bei Tod
    if done:
        obs = env.reset()

    # Zeichnen
    screen.fill((0,0,0))
    for segment in snake_env.snake:
        pygame.draw.rect(screen,(0,255,0),(segment[0]*cell_size,segment[1]*cell_size,cell_size,cell_size))
    pygame.draw.rect(screen,(255,0,0),(snake_env.food[0]*cell_size,snake_env.food[1]*cell_size,cell_size,cell_size))

    # Score und Highscore anzeigen
    score_text = font.render(f"Score: {snake_env.score}", True, (255,255,255))
    high_text = font.render(f"Highscore: {highscore}", True, (255,255,0))
    screen.blit(score_text,(10,10))
    screen.blit(high_text,(10,40))

    pygame.display.flip()
    clock.tick(speed)

pygame.quit()
model.save(MODEL_PATH)
print("Model gespeichert.")
