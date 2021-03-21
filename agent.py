import torch 
import random
import numpy as np
from collections import deque
from environmentAI import Game, Direction, Point, BLOCK_SIZE
from model import LinearQNet, QTrainer
from plotter import plot
from variables import MAX_MEMORY, BATCH_SIZE, LEARNING_RATE, EPSILON_DELTA, GAMMA, GAMMA_LOW, IS_INCREMENTING

INPUT_SIZE = 11 # Amount of inputs in state
HIDDEN_SIZE = 256 # Amount of hidden nodes
OUTPUT_SIZE = 3 # Amount of actions that AI can take

# Agent class
class Agent: 
    def __init__(self):
        # Counting of games
        self.numberOfGames = 0 
        # Randomned seeding
        self.epsilon = 0 
        # Discount rate
        self.gamma = GAMMA
        # Incrementing discount rate
        self.gammaIncrementing = GAMMA_LOW
        # Function to call popleft to rewrite memory
        self.memory = deque(maxlen=MAX_MEMORY) 

        self.model = LinearQNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
        if IS_INCREMENTING and self.gammaIncrementing < self.gamma:
            self.trainer = QTrainer(self.model, LEARNING_RATE, self.gammaIncrementing)
        else:
            self.trainer = QTrainer(self.model, LEARNING_RATE, self.gamma)

    def getState(self, game):
        head = game.snake[0]

        # We have to remember that game grid gives us 
        # incrementing "x" to the RIGHT 
        # and "y" in the DOWN direction
        pointLeft = Point(head.x - BLOCK_SIZE, head.y)
        pointRight = Point(head.x + BLOCK_SIZE, head.y)
        pointUp = Point(head.x, head.y - BLOCK_SIZE)
        pointDown = Point(head.x, head.y + BLOCK_SIZE)
        # Check in which direction snake is going
        directionLeft = game.direction == Direction.LEFT
        directionRight = game.direction == Direction.RIGHT
        directionUp = game.direction == Direction.UP
        directionDown = game.direction == Direction.DOWN
        # List for collision detection
        state = [ 
            #Danger ahead
            (directionLeft and game.onCollision(pointLeft)) or
            (directionRight and game.onCollision(pointRight)) or
            (directionUp and game.onCollision(pointUp)) or
            (directionDown and game.onCollision(pointDown)),

            #Danger to the right
            (directionLeft and game.onCollision(pointUp)) or
            (directionRight and game.onCollision(pointDown)) or
            (directionUp and game.onCollision(pointRight)) or
            (directionDown and game.onCollision(pointLeft)),

            #Danger to the left
            (directionLeft and game.onCollision(pointDown)) or
            (directionRight and game.onCollision(pointUp)) or
            (directionUp and game.onCollision(pointLeft)) or
            (directionDown and game.onCollision(pointRight)),

            #Move direction
            directionLeft,
            directionRight,
            directionUp,
            directionDown,

            #Snack detection
            game.snack.x < game.head.x, #Snack to the left
            game.snack.x > game.head.x, #Snack to the right
            game.snack.y > game.head.y, #Snack down
            game.snack.y < game.head.y  #Snack up
            ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, nextState, gameOver): # Save inputs 
        # If MAX_MEMORY is reached popleft // return Tuple
        self.memory.append((state, action, reward, nextState, gameOver))

    def trainLongMemory(self):  # If lenght of the memory exceeds BATCH_SIZE we can train long memory more efficiently
        if len(self.memory) > BATCH_SIZE: 
            # Take random sample from memory as a tuple
            miniSample = random.sample(self.memory, BATCH_SIZE) 
        else:
            miniSample = self.memory

        # Put states, actions etc. together in one Tuple
        states, actions, rewards, nextStates, gameOvers = zip(*miniSample)
        self.trainer.trainStep(states, actions, rewards, nextStates, gameOvers)

    def trainShortMemory(self, state, action, reward, nextState, gameOver):
        self.trainer.trainStep(state, action, reward, nextState, gameOver)

    def getAction(self, state):
        # Generating random moves -> Tradeoff exploration / exploitantion
        # Epsilon value will be lowered every game
        self.epsilon = EPSILON_DELTA - self.numberOfGames 
        move = [0,0,0]

        # The higher the epsilon the more random moves will the snake make
        if random.randint(0,200) < self.epsilon: 
            # If the statement is fullfiled, random move will be generated
            moveIndex = random.randint(0,2) 
            # Assign 1 to the move on chosen index to decide what will be the next move
            move[moveIndex] = 1 
        # Otherwise choose the move based on model
        else: 
            # Convert state to tensor using torch library [5.0 , 2.7 , 0.5] -> [1,0,0] 
            # Chooses the biggest number
            inputState = torch.tensor(state, dtype=torch.float) 
            # Get the prediction
            prediction = self.model(inputState) 
            # Convert tensor to one number // Based on index
            predictedMove = torch.argmax(prediction).item() 
            # Assign 1 to the move on chosen index to decide what will be the next move
            move[predictedMove] = 1
        return move

def train():
    plotScores = []
    plotMeanScores = []
    totalScore = 0
    record = 0
    agent = Agent()
    game = Game()
    
    while True:
        # Get old state
        oldState = agent.getState(game)

        # Get move
        move = agent.getAction(oldState)

        # Perform move and return state
        reward, gameOver, score = game.takeAction(move)

        # Save new state
        newState = agent.getState(game)

        # Train the short memory
        agent.trainShortMemory(oldState, move, reward, newState, gameOver)

        # Remember retrurned values
        agent.remember(oldState, move, reward, newState, gameOver)

        if gameOver:
            # Train replay memory and plot the results
            game.reset()
            agent.numberOfGames += 1 # Increment the number of games each game
            agent.gammaIncrementing *= agent.numberOfGames # Gamma Incrementing
            agent.trainLongMemory()

            # Highscore logic -> save only better scored games
            if score > record:
                record = score # Set record to score
                agent.model.save() # Save the model
            
            print('Game:', agent.numberOfGames, 'Score:', score, 'Record:', record)
            
            plotScores.append(score) # Append the plot scores list with score
            totalScore += score # Add current score to score total
            meanScore = totalScore / agent.numberOfGames # Calculate mean score using total
            plotMeanScores.append(meanScore) # Append the mean score plot with current mean score
            #print('Scores:', plotScores, 'Mean Scores:', plotMeanScores) #Debugging
            # plot(plotScores, plotMeanScores) # Plotting of the scores
            plot(plotScores, plotMeanScores)

if __name__ == '__main__':
    train()

