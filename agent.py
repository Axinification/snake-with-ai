import torch 
import random
import numpy as np
import os
from collections import deque
from environmentAI import Game, Direction, Point, BLOCK_SIZE
from model import LinearQNet, QTrainer
from plotter import plot
from variables import (MAX_MEMORY, BATCH_SIZE, LEARNING_RATE, 
                        EPSILON_DELTA, GAMMA, GAMMA_LOW, GAMMA_INCREMENT,
                        IS_INCREMENTING, CHECKPOINT_PATH, LOAD)

#Constants
INPUT_SIZE = 11 # Amount of inputs in state
HIDDEN_SIZE = 256 # Amount of hidden nodes // can be changed
OUTPUT_SIZE = 3 # Amount of actions that AI can take

# Agent class
class Agent: 
    def __init__(self):
        # Size of tile
        self.blockSize = BLOCK_SIZE
        # Counting of games
        self.numberOfGames = 0 
        # Randomness
        self.epsilon = 0
        # Randomness change 
        self.epsilonChange = EPSILON_DELTA
        # Discount rate
        self.gamma = GAMMA
        # Set gamma check
        self.isIncrementing = IS_INCREMENTING
        # Delta of gamma
        self.gammaIncrement = GAMMA_INCREMENT
        # Current gamma
        self.currentGamma = GAMMA_LOW
        # Learning Rate
        self.learningRate = LEARNING_RATE
        # Function to call popleft to rewrite memory
        self.memory = deque(maxlen=MAX_MEMORY) 
        # Size of batch for learning
        self.batchSize = BATCH_SIZE
        #Variables for plotting
        self.plotScores=[]
        self.plotMeanScores=[]
        self.totalScore = 0

        self.model = LinearQNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
        if LOAD:
            checkpoint = torch.load(os.path.join(CHECKPOINT_PATH, 'checkpoint.pth'))
            model = torch.load(CHECKPOINT_PATH + '/model.pth')
            self.numberOfGames = checkpoint['numberOfGames']
            self.learningRate = checkpoint['learningRate']
            self.currentGamma = checkpoint['gamma']
            self.plotScores = checkpoint['scores']
            self.plotMeanScores = checkpoint['meanScores']
            self.totalScore = checkpoint['totalScore']
            self.model.load_state_dict(model)
            self.model.eval()
            self.trainer = QTrainer(self.model, self.learningRate, self.currentGamma)
            checkpoint = torch.load(os.path.join(CHECKPOINT_PATH, 'checkpoint.pth'))
            self.trainer.optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            self.trainer = QTrainer(self.model, self.learningRate, self.currentGamma)

    def getState(self, game):
        head = game.snake[0]

        # We have to remember that game grid gives us 
        # incrementing "x" to the RIGHT 
        # and "y" in the DOWN direction
        pointLeft = Point(head.x - self.blockSize, head.y)
        pointRight = Point(head.x + self.blockSize, head.y)
        pointUp = Point(head.x, head.y - self.blockSize)
        pointDown = Point(head.x, head.y + self.blockSize)
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
        if len(self.memory) > self.batchSize: 
            # Take random sample from memory as a tuple
            miniSample = random.sample(self.memory, self.batchSize) 
        else:
            miniSample = self.memory

        # Put states, actions etc. together in one Tuple
        states, actions, rewards, nextStates, gameOvers = zip(*miniSample)
        self.trainer.trainStep(states, actions, rewards, nextStates, gameOvers)

    def trainShortMemory(self, state, action, reward, nextState, gameOver):
        self.trainer.trainStep(state, action, reward, nextState, gameOver)

    def useMemory(self, state, move):
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

    def getAction(self, state):
        # Generating random moves -> Tradeoff exploration / exploitantion
        # Epsilon value will be lowered every game
        self.epsilon = self.epsilonChange - self.numberOfGames 
        move = [0,0,0]
        if not LOAD:
            # The higher the epsilon the more random moves will the snake make
            if random.randint(0,self.epsilonChange) < self.epsilon: 
                # If the statement is fullfiled, random move will be generated
                moveIndex = random.randint(0,2) 
                # Assign 1 to the move on chosen index to decide what will be the next move
                move[moveIndex] = 1 
            # Otherwise choose the move based on model
            else: 
                self.useMemory(state, move)
            return move
        else:
            return self.useMemory(state, move)

def train():
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
            if agent.isIncrementing and agent.currentGamma < agent.gamma:
                if agent.numberOfGames > agent.epsilonChange:
                    agent.currentGamma += agent.gammaIncrement # Gamma Incrementing
                agent.currentGamma = round(agent.currentGamma, 3)
                agent.trainer = QTrainer(agent.model, agent.learningRate, agent.currentGamma)
            else:
                agent.currentGamma = agent.gamma
                agent.trainer = QTrainer(agent.model, agent.learningRate, agent.gamma)
            agent.trainLongMemory()

            # Highscore logic -> save only better scored games
            if score > record:
                record = score # Set record to score
                
            #Saving
            agent.model.saveModel(CHECKPOINT_PATH) # Save the model
            agent.trainer.saveParameters( agent.plotScores, agent.plotMeanScores, agent.totalScore, agent.numberOfGames, CHECKPOINT_PATH) # Save parameters    
            
            print('Game:', agent.numberOfGames, 'Score:', score, 'Record:', record, 'Gamma:', agent.currentGamma)
            
            agent.plotScores.append(score) # Append the plot scores list with score
            agent.totalScore += score # Add current score to score total
            meanScore = agent.totalScore / agent.numberOfGames # Calculate mean score using total
            agent.plotMeanScores.append(meanScore) # Append the mean score plot with current mean score
            # print('Scores:', plotScores, 'Mean Scores:', plotMeanScores) #Debugging
            # plot(plotScores, plotMeanScores) # Plotting of the scores
            plot(agent.plotScores, agent.plotMeanScores)

if __name__ == '__main__':
    train()

