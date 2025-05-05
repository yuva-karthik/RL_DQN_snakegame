import pygame
import random
import sys

pygame.init()

CELL_SIZE = 20
GRID_WIDTH = 35
GRID_HEIGHT = 20
SCREEN_WIDTH = CELL_SIZE * GRID_WIDTH
SCREEN_HEIGHT = CELL_SIZE * GRID_HEIGHT
FPS = 10

WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
DARK_GREEN = (0, 155, 0)
RED = (255, 0, 0)
BLACK = (0, 0, 0)

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Snake Game")
clock = pygame.time.Clock()

font = pygame.font.SysFont(None, 35)

def draw_text(text, color, x, y):
    img = font.render(text, True, color)
    screen.blit(img, (x, y))

def draw_snake(snake_list):
    for segment in snake_list:
        pygame.draw.rect(screen, GREEN, (segment[0], segment[1], CELL_SIZE, CELL_SIZE))

def get_random_food():
    x = random.randint(0, GRID_WIDTH - 1) * CELL_SIZE
    y = random.randint(0, GRID_HEIGHT - 1) * CELL_SIZE
    return (x, y)

def main():
    snake = [(100, 100)]
    direction = (CELL_SIZE, 0) 
    food = get_random_food()
    score = 0

    run = True
    while run:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP and direction != (0, CELL_SIZE):
                    direction = (0, -CELL_SIZE)
                elif event.key == pygame.K_DOWN and direction != (0, -CELL_SIZE):
                    direction = (0, CELL_SIZE)
                elif event.key == pygame.K_LEFT and direction != (CELL_SIZE, 0):
                    direction = (-CELL_SIZE, 0)
                elif event.key == pygame.K_RIGHT and direction != (-CELL_SIZE, 0):
                    direction = (CELL_SIZE, 0)

        head_x, head_y = snake[0]
        new_head = (head_x + direction[0], head_y + direction[1])
        snake.insert(0, new_head)

        if new_head == food:
            score += 1
            food = get_random_food()
        else:
            snake.pop()

        if (new_head[0] < 0 or new_head[0] >= SCREEN_WIDTH or
            new_head[1] < 0 or new_head[1] >= SCREEN_HEIGHT):
            run = False

        if new_head in snake[1:]:
            run = False

        screen.fill(BLACK)
        draw_snake(snake)
        pygame.draw.rect(screen, RED, (food[0], food[1], CELL_SIZE, CELL_SIZE))
        draw_text(f"Score: {score}", WHITE, 10, 10)
        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
