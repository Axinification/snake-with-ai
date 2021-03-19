import pygame
import random
import math
from enum import Enum
from collections import namedtuple

pygame.init()
font = pygame.font.SysFont('arial', 25)
#Touple for points on the grid
Point = namedtuple('Point', 'x, y')

#VARIABLES
SIZE = 500
ROWS = 25
BLOCK_SIZE = SIZE/ROWS
TIME_DELAY = 50
CLOCK_TICK = 10
START_ROW = 10
START_COLUMN = 10
SNAKE_POSITION = Point(START_COLUMN*BLOCK_SIZE, START_ROW*BLOCK_SIZE)


#colors
SNAKE_COLOR = (0,255,0) #GREEN
WHITE = (255,255,255)
BLACK = (0,0,0)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


class Game:
    def __init__(self, width=SIZE, height=SIZE, rows=ROWS):
        #Set initial values
        self.width = width
        self.height = height
        self.rows = rows

        #Set initial display
        self.display = pygame.display.set_mode((width, height))
        pygame.display.set_caption('Snake Evironment')

        #Set the clock
        self.clock = pygame.time.Clock()

        #Initial game state
        self.direction = Direction.RIGHT

        self.head = SNAKE_POSITION
        self.snake = [self.head, 
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.snack = None
        self.randomColor = (random.randrange(20,255),random.randrange(20,255),random.randrange(20,255))
        self.placeSnack()
    
    def takeAction(self): # Collect input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                
            keys = pygame.key.get_pressed()
            for key in keys: #reversing is blocked for now
                if keys[pygame.K_LEFT] and self.direction is not Direction.RIGHT:
                    self.direction = Direction.LEFT
                elif keys[pygame.K_RIGHT] and self.direction is not Direction.LEFT:
                    self.direction = Direction.RIGHT
                elif keys[pygame.K_UP] and self.direction is not Direction.DOWN:
                    self.direction = Direction.UP
                elif keys[pygame.K_DOWN] and self.direction is not Direction.UP:
                    self.direction = Direction.DOWN
        # Move the head
        self.moveSnake(self.direction) # update the head
        self.snake.insert(0, self.head) # update snake array
        
        # check if game over
        gameOver = False #First assume that game is not over
        if self.onCollision():
            gameOver = True #If collision is detected switch gameOver to true
            return gameOver, self.score
            
        # 4. place new snack or just move
        if self.head == self.snack:
            self.score += 1 #Add point if position of head is the same 
            self.placeSnack() #Place new snack if eaten
        else:
            self.snake.pop() #Pop the tail if no snack is eaten, otherwise it will stay in one place and the snake will grow constantly
        
        # 5. update ui and clock
        self.redrawWindow()
        self.clock.tick(CLOCK_TICK)
        pygame.time.delay(TIME_DELAY)
        # 6. return game over and score
        return gameOver, self.score
    
    def moveSnake(self, direction):
        x = self.head.x
        y = self.head.y
        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE
            
        self.head = Point(x, y)
    
    def onCollision(self):
        #WALLS VERSION
        if (self.head.x > self.width - BLOCK_SIZE or  
            self.head.x < 0 or 
            self.head.y > self.height - BLOCK_SIZE or 
            self.head.y < 0):
            return True
        
        #NO WALLS VERSION
        # if self.direction is Direction.LEFT and self.head.x < 0: self.head = Point(SIZE, self.head.y)
        # elif self.direction is Direction.RIGHT and self.head.x > self.width - BLOCK_SIZE: self.head = Point(-BLOCK_SIZE, self.head.y)
        # elif self.direction is Direction.UP and self.head.y < 0: self.head = Point(self.head.x, SIZE)
        # elif self.direction is Direction.DOWN and self.head.y > self.height - BLOCK_SIZE: self.head = Point(self.head.x, -BLOCK_SIZE)
        # hits itself
        if self.head in self.snake[1:]:
            return True
        
        return False
    
    def placeSnack(self):
        x = random.randint(0, ROWS-1)*BLOCK_SIZE
        y = random.randint(0, ROWS-1)*BLOCK_SIZE
        self.randomColor = (random.randrange(20,255),random.randrange(20,255),random.randrange(20,255))
        self.snack = Point(x, y)
        if self.snack in self.snake:
            self.placeSnack()
    
    def redrawWindow(self):
        self.display.fill(BLACK)
    
        for segment in self.snake:
            pygame.draw.rect(self.display, SNAKE_COLOR, pygame.Rect(segment.x, segment.y, BLOCK_SIZE, BLOCK_SIZE))
        
        pygame.draw.rect(self.display, self.randomColor, pygame.Rect(self.snack.x, self.snack.y, BLOCK_SIZE, BLOCK_SIZE))
        
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

if __name__ == '__main__':
            game = Game()
            
            # game loop
            while True:
                gameOver, score = game.takeAction() 
                
                if gameOver == True:
                    break
                
            print('Final Score', score)
                
                
            pygame.quit()