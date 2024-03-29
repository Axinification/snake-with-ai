from numpy.core.numeric import array_equal
import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
from variables import (PENALTY_MULTIPLIER, SIZE, TILES, TIME_DELAY, CLOCK_TICK, 
                        START_ROW, START_COLUMN, REWARD, 
                        REWARD_MULTIPLIER, LOOP_TIME, TIME_PENALTY, 
                        EASY_MODE, DIRECTION_REWARD, STRAIGHT_LINE_REWARD, COILING_PENALTY)
import matplotlib.pyplot as plt

pygame.init()

font = pygame.font.SysFont('arial', 25)
#Touple for points on the grid
Point = namedtuple('Point', 'x, y')

#VARIABLES
BLOCK_SIZE = SIZE/TILES
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
    def __init__(self, width=SIZE, height=SIZE, tiles=TILES):
        #Set initial values

        self.width = width
        self.height = height
        self.tiles = tiles

        #Set initial display
        self.display = pygame.display.set_mode((width, height))
        pygame.display.set_caption('Snake Evironment')

        #Snake turn counters
        self.rightTurns = 0 # Int for right turns
        self.leftTurns = 0 # Int for left turns

        #Set the clock
        self.clock = pygame.time.Clock()
        self.reset() #Initializing zero state

    def reset(self): # Restart function
        #Initial game state
        self.direction = Direction.RIGHT # Set the starting direction
        self.head = SNAKE_POSITION # Set the snake starting position
        # Create List for snake body parts
        self.snake = [self.head, 
                    Point(self.head.x-BLOCK_SIZE, self.head.y),
                    Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]

        self.score = 0 # Score
        self.frameIteration = 0 # Used to prevent endless looping
        self.snack = None # Snack position initiation
        self.randomColor = (random.randrange(20,255),random.randrange(20,255),random.randrange(20,255))
        self._placeSnack() # Update snack position
    
    def takeAction(self, action): # Check input result
        self.frameIteration += 1 # Update frameIteration every step
        self.snackFrameIteration += 1 # Update sncak time counter

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                
        # Move the head
        self._moveSnake(action)
        # Check the reward for step
        self.reward = 0
        self._directionReward()
        self._turnReward(action)
        # Update snake array
        self.snake.insert(0, self.head) 
        # print('Reward for step: ', self.reward)

        # Check if game over
        gameOver = False # First assume that game is not over
        # Game over on collision or when time runs out
        if self.onCollision() or self.frameIteration > LOOP_TIME*len(self.snake):
            gameOver = True # If collision is detected switch gameOver to true
            self.reward += -REWARD*PENALTY_MULTIPLIER # Return -REWARD if game is lost
            return self.reward, gameOver, self.score
            
        # Place new snack or just move
        if self.head == self.snack:
            self.score += 1 # Add point if position of head is the same 
            self._placeSnack() # Place new snack if eaten
            if REWARD*REWARD_MULTIPLIER-self.frameIteration*TIME_PENALTY>REWARD:
                self.reward += REWARD*REWARD_MULTIPLIER-self.snackFrameIteration*TIME_PENALTY # Give out the reward // Reward lost with time
            else:
                self.reward += REWARD
        else:
            # Pop the tail if no snack is eaten, 
            # otherwise it will stay in one place and the snake will grow constantly
            self.snake.pop()
        
        # Update window, clock and add delay
        self._redrawWindow()
        self.clock.tick(CLOCK_TICK)
        pygame.time.delay(TIME_DELAY)

        # Return game over and score
        return self.reward, gameOver, self.score
    
    def _moveSnake(self, action): # Determining the next move based on action  
        #- [1,0,0] straight
        #- [0,1,0] right turn
        #- [0,0,1] left turn

        directions = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP] # Clockwise order for iterating
        index = directions.index(self.direction) # Get the index of move in clockwise order -> [0,1,2,3]

        # Turning logic
        if np.array_equal(action, [1,0,0]): 
            newDirection = directions[index] # No change
        elif np.array_equal(action, [0,1,0]):
            # Modulo used for iteration -> (2+1)%4 = 3 -> (3+1)%4 = 0 
            nextIndex = (index + 1) % 4 # Choose next action
            newDirection = directions[nextIndex] # Turn right
        else: 
            # Modulo used for iteration -> (1-1)%4 = 0 -> (0-1)%4 = 3
            nextIndex = (index - 1) % 4 # Choose next action
            newDirection = directions[nextIndex] # Turn left
        # Set the direction to newDirection
        self.direction = newDirection #Set the direction to the one choosen 
        
        x = self.head.x #Get the x coordinate of head
        y = self.head.y #Get the y coordinate of head

        #Moving logic
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE #Move right
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE #Move left
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE #Move up
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE #Move down
            
        self.head = Point(x, y) #Update head coordinates
    
    def onCollision(self, head = None): # Collision detection
        if EASY_MODE:
            if self.direction is Direction.LEFT and head.x < 0: self.head = Point(SIZE, head.y)
            elif self.direction is Direction.RIGHT and head.x > self.width - BLOCK_SIZE: head = Point(-BLOCK_SIZE, head.y)
            elif self.direction is Direction.UP and head.y < 0: self.head = Point(head.x, SIZE)
            elif self.direction is Direction.DOWN and head.y > self.height - BLOCK_SIZE: self.head = Point(head.x, -BLOCK_SIZE)
        else:
            if head == None:
                head = self.head
            if (head.x > self.width - BLOCK_SIZE or  #If head collides with right side wall
                head.x < 0 or #If head collides with left side wall
                head.y > self.height - BLOCK_SIZE or  #If head collides with up side wall
                head.y < 0): #If head collides with down side wall
                return True
        
        if head in self.snake[1:]: #If the point of head is the same as the point of snake starting from index 1 of the array
            return True
        
        return False
    
    def _placeSnack(self): # Place snack
        x = random.randint(0, TILES-1)*BLOCK_SIZE # Get the new x coordinate of snack
        y = random.randint(0, TILES-1)*BLOCK_SIZE # Get the new y coordinate of snack
        self.snackFrameIteration = 0
        self.randomColor = (random.randrange(20,255),random.randrange(20,255),random.randrange(20,255)) #Pick random color for the snack
        self.snack = Point(x, y) # Set the coordinates of the snack
        # If snack is in snake list spawn snack
        if self.snack in self.snake:
            self._placeSnack()
    
    def _directionReward(self): # Reward for going in the snack direction
        reward = 0
        #Rewards for movement
        if self.direction is Direction.RIGHT and self.head.x <= self.snack.x and self.head.y == self.snack.y:
            reward = STRAIGHT_LINE_REWARD
        elif self.direction is Direction.RIGHT and self.head.x <= self.snack.x:
            reward = DIRECTION_REWARD
        if self.direction is Direction.LEFT and self.head.x >= self.snack.x and self.head.y == self.snack.y:
            reward = STRAIGHT_LINE_REWARD
        elif self.direction is Direction.LEFT and self.head.x >= self.snack.x:
            reward = DIRECTION_REWARD
        if self.direction is Direction.DOWN and self.head.y <= self.snack.y and self.head.x == self.snack.x:
            reward = STRAIGHT_LINE_REWARD
        elif self.direction is Direction.DOWN and self.head.y <= self.snack.y:
            reward = DIRECTION_REWARD
        if self.direction is Direction.UP and self.head.y >= self.snack.y and self.head.x == self.snack.x:
            reward = STRAIGHT_LINE_REWARD
        elif self.direction is Direction.UP and self.head.y >= self.snack.y:
            reward = DIRECTION_REWARD
        self.reward += reward
    
    def _turnReward(self, action): # Turning reward // coiling prevention
        # [0,1,0] right turn
        # [0,0,1] left turn

        # If snake made full loop set turns to 0
        reward = 0

        pointLeft = Point(self.head.x - BLOCK_SIZE, self.head.y)
        pointUp = Point(self.head.x, self.head.y - BLOCK_SIZE)
        pointRight = Point(self.head.x + BLOCK_SIZE, self.head.y)
        pointDown = Point(self.head.x, self.head.y + BLOCK_SIZE)

        # Count the number of turns in a given direction
        if np.array_equal(action, [0,0,1]): # Turn left
            self.leftTurns += 1
            if self.rightTurns != 0:
                self.rightTurns -= 1
        elif np.array_equal(action, [0,1,0]): # Turn right
            self.rightTurns += 1
            if self.leftTurns != 0:
                self.leftTurns -= 1
        
        # Checks if there was more right or left turns and if action that will be taken increments the higher one
        checker = (self.rightTurns > self.leftTurns and np.array_equal(action, [0,1,0])) or (self.leftTurns > self.rightTurns and np.array_equal(action, [0,0,1]))

        # If the point ahead is in the snake body
        if self.direction == Direction.LEFT and pointLeft in self.snake[1:]:
            if checker:
                reward = COILING_PENALTY
        if self.direction == Direction.UP and pointUp in self.snake[1:]:
            if checker:
                reward = COILING_PENALTY
        if self.direction == Direction.RIGHT and pointRight in self.snake[1:]:
            if checker:
                reward = COILING_PENALTY
        if self.direction == Direction.DOWN and pointDown in self.snake[1:]:
            if checker:
                reward = COILING_PENALTY

        if self.rightTurns == 4 or self.leftTurns == 4:
            self.rightTurns = 0
            self.leftTurns = 0

        self.reward += reward
    
    def _redrawWindow(self): # Update display
        self.display.fill(BLACK)
    
        for segment in self.snake:
            # Draw eyes for the first segment
            if segment == self.snake[0]:
                centre = BLOCK_SIZE//2
                radius = BLOCK_SIZE//10
                circleMiddle = (segment.x+centre-radius, segment.y+centre-radius)
                circleMiddle2 = (segment.x+BLOCK_SIZE-radius*2, segment.y+centre-radius)
                pygame.draw.rect(self.display, SNAKE_COLOR, pygame.Rect(segment.x, segment.y, BLOCK_SIZE, BLOCK_SIZE))
                pygame.draw.circle(self.display, (0,0,0), circleMiddle, radius)
                pygame.draw.circle(self.display, (0,0,0), circleMiddle2, radius)
            else:
                pygame.draw.rect(self.display, SNAKE_COLOR, pygame.Rect(segment.x, segment.y, BLOCK_SIZE, BLOCK_SIZE))
        
        pygame.draw.rect(self.display, self.randomColor, pygame.Rect(self.snack.x, self.snack.y, BLOCK_SIZE, BLOCK_SIZE))
        
        text = font.render("Score: " + str(self.score), True, WHITE) # Set the text to score
        self.display.blit(text, [0, 0]) # Display score
        pygame.display.flip()