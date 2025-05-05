import pygame
import random
import numpy as np

class SnakeGame:
    def __init__(self):
        self.width = 640
        self.height = 480
        self.cell_size = 20
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.reset()

    def reset(self):
        """ Resets the game to the initial state. """
        self.snake = [(100, 100)] 
        self.food = self._get_random_food()
        self.direction = (self.cell_size, 0) 
        self.score = 0
        self.done = False
        self.state = self._get_state()
        return self.state

    def step(self, action):
        """ Takes an action and returns the next state, reward, and done flag. """
        if action == 1:  
            self.direction = (-self.direction[1], self.direction[0])
        elif action == 2: 
            self.direction = (self.direction[1], -self.direction[0])

        head_x, head_y = self.snake[0]
        new_head = (head_x + self.direction[0], head_y + self.direction[1])
        self.snake.insert(0, new_head)

        reward = -0.1 
        if new_head == self.food:
            self.score += 1
            reward = 10 
            self.food = self._get_random_food() 
        else:
            self.snake.pop() 

        if self._is_collision(new_head):
            self.done = True
            reward = -100 

        self.state = self._get_state()
        return self.state, reward, self.done

    def _get_state(self):
        """ Get the current state (snake's head, food, direction, etc.) """
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        return np.array([head_x, head_y, food_x, food_y, self.direction[0], self.direction[1]])

    def _is_collision(self, head):
        """ Check if the snake collides with the wall or itself. """
        if (head[0] < 0 or head[0] >= self.width or 
            head[1] < 0 or head[1] >= self.height or 
            head in self.snake[1:]):
            return True
        return False

    def _get_random_food(self):
        """ Return a random food position """
        x = random.randint(0, (self.width // self.cell_size) - 1) * self.cell_size
        y = random.randint(0, (self.height // self.cell_size) - 1) * self.cell_size
        return (x, y)

    def render(self):
        """ Display the game (for debugging or showing the agent play). """
        self.screen.fill((0, 0, 0))  # Black background
        for segment in self.snake:
            pygame.draw.rect(self.screen, (0, 255, 0), (segment[0], segment[1], self.cell_size, self.cell_size))
        pygame.draw.rect(self.screen, (255, 0, 0), (self.food[0], self.food[1], self.cell_size, self.cell_size))
        pygame.display.flip()

    def close(self):
        """ Close the pygame window when done. """
        pygame.quit()
