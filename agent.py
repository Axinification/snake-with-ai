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
                        IS_INCREMENTING, LOAD)

#Define Version

def returnInputVersion():
    print("""Define input version. 
    f -> Far seeing
    s -> Short seeing
    fn -> Far no direction
    sn -> Short no direction""")

    inputVersion = input("Type one of the options: ")

    if(inputVersion == 'f' or inputVersion == 's' or inputVersion == 'fn' or inputVersion == 'sn'):
        return inputVersion
    else:
        print("Input invalid. Try again.")
        return returnInputVersion()

INPUT_VERSION = returnInputVersion()

def returnInputSize():
    if(INPUT_VERSION == 'f'):
        return 22 # Amount of inputs in state
    elif(INPUT_VERSION == 'fn'):
        return 18
    elif(INPUT_VERSION == 's'):
        return 15
    elif(INPUT_VERSION == 'sn'):
        return 12

INPUT_SIZE = returnInputSize()
HIDDEN_SIZE = 256 # Amount of hidden nodes // can be changed
OUTPUT_SIZE = 3 # Amount of actions that AI can take

def returnHiddenLayersAmount():
    hiddenLayersAmount = input("Define the number of hidden layers, minimum 0: ")
    if(int(hiddenLayersAmount)<0):
        print("The number can't be lower than 0!")
        return returnHiddenLayersAmount()
    else:
        return hiddenLayersAmount

HIDDEN_LAYERS_AMOUNT = returnHiddenLayersAmount()

def returnCheckpoint(hiddenAmount, inputVersion):
    return "checkpoints/version-"+inputVersion+"-hidden-"+hiddenAmount

CHECKPOINT_PATH = returnCheckpoint(HIDDEN_LAYERS_AMOUNT,INPUT_VERSION)

def loadInfo(self):
        if LOAD:
            if os.path.exists(CHECKPOINT_PATH+'/checkpoint.pth') & os.path.exists(CHECKPOINT_PATH+'/model.pth') :
                checkpoint = torch.load(CHECKPOINT_PATH+'/checkpoint.pth')
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
        else:
                self.trainer = QTrainer(self.model, self.learningRate, self.currentGamma)

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
        loadInfo(self)

    def getState(self, game):
        head = game.snake[0]
        # We have to remember that game grid gives us 
        # incrementing "x" to the RIGHT 
        # and "y" in the DOWN direction
        pointLeft = Point(head.x - self.blockSize, head.y)
        pointLeftUp = Point(head.x - self.blockSize, head.y - self.blockSize)
        pointUp = Point(head.x, head.y - self.blockSize)
        pointRightUp = Point(head.x + self.blockSize, head.y - self.blockSize)
        pointRight = Point(head.x + self.blockSize, head.y)
        pointRightDown = Point(head.x + self.blockSize, head.y + self.blockSize)
        pointDown = Point(head.x, head.y + self.blockSize)
        pointLeftDown = Point(head.x - self.blockSize, head.y + self.blockSize)
        
        if (INPUT_VERSION != "s" or "sn"):
            pointFarLeft = Point(head.x - 2*self.blockSize, head.y)
            pointFarLeftUp = Point(head.x - 2*self.blockSize, head.y - 2*self.blockSize)
            pointFarUp = Point(head.x, head.y - 2*self.blockSize)
            pointFarRightUp = Point(head.x + 2*self.blockSize, head.y - 2*self.blockSize)
            pointFarRight = Point(head.x + 2*self.blockSize, head.y)
            pointFarRightDown = Point(head.x + 2*self.blockSize, head.y + 2*self.blockSize)
            pointFarDown = Point(head.x, head.y + 2*self.blockSize)
            pointFarLeftDown = Point(head.x - 2*self.blockSize, head.y + 2*self.blockSize)
        # Check in which direction snake is going
        if (INPUT_VERSION != "fn" or "sn"):
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

            #Danger to the left forward
            (directionLeft and game.onCollision(pointLeftDown)) or
            (directionRight and game.onCollision(pointRightUp)) or
            (directionUp and game.onCollision(pointLeftUp)) or
            (directionDown and game.onCollision(pointRightDown)),

            #Danger to the right forward
            (directionLeft and game.onCollision(pointLeftUp)) or
            (directionRight and game.onCollision(pointRightDown)) or
            (directionUp and game.onCollision(pointRightUp)) or
            (directionDown and game.onCollision(pointLeftDown)),

            #Danger to the left backward
            (directionLeft and game.onCollision(pointRightDown)) or
            (directionRight and game.onCollision(pointLeftUp)) or
            (directionUp and game.onCollision(pointLeftDown)) or
            (directionDown and game.onCollision(pointRightUp)),

            #Danger to the right backward
            (directionLeft and game.onCollision(pointRightUp)) or
            (directionRight and game.onCollision(pointLeftDown)) or
            (directionUp and game.onCollision(pointRightDown)) or
            (directionDown and game.onCollision(pointLeftUp))
        ]

        #FAR VERSION
        far = [
            #Danger far ahead
            (directionLeft and game.onCollision(pointFarLeft)) or
            (directionRight and game.onCollision(pointFarRight)) or
            (directionUp and game.onCollision(pointFarUp)) or
            (directionDown and game.onCollision(pointFarDown)),

            #Danger far to the right
            (directionLeft and game.onCollision(pointFarUp)) or
            (directionRight and game.onCollision(pointFarDown)) or
            (directionUp and game.onCollision(pointFarRight)) or
            (directionDown and game.onCollision(pointFarLeft)),

            #Danger far to the left
            (directionLeft and game.onCollision(pointFarDown)) or
            (directionRight and game.onCollision(pointFarUp)) or
            (directionUp and game.onCollision(pointFarLeft)) or
            (directionDown and game.onCollision(pointFarRight)),

            #Danger far to the left forward
            (directionLeft and game.onCollision(pointFarLeftDown)) or
            (directionRight and game.onCollision(pointFarRightUp)) or
            (directionUp and game.onCollision(pointFarLeftUp)) or
            (directionDown and game.onCollision(pointFarRightDown)),

            #Danger far to the right forward
            (directionLeft and game.onCollision(pointFarLeftUp)) or
            (directionRight and game.onCollision(pointFarRightDown)) or
            (directionUp and game.onCollision(pointFarRightUp)) or
            (directionDown and game.onCollision(pointFarLeftDown)),

            #Danger far to the left backward
            (directionLeft and game.onCollision(pointFarRightDown)) or
            (directionRight and game.onCollision(pointFarLeftUp)) or
            (directionUp and game.onCollision(pointFarLeftDown)) or
            (directionDown and game.onCollision(pointFarRightUp)),

            #Danger far to the right backward
            (directionLeft and game.onCollision(pointFarRightUp)) or
            (directionRight and game.onCollision(pointFarLeftDown)) or
            (directionUp and game.onCollision(pointFarRightDown)) or
            (directionDown and game.onCollision(pointFarLeftUp)),
        ]
        if (INPUT_VERSION != "s" or "sn"):
            state.extend(far)

        #Move direction
        direction = [
            directionLeft,
            directionRight,
            directionUp,
            directionDown
        ]
        if (INPUT_VERSION != "s" or "sn"):
            state.extend(direction)

        #Snack detection
        snack = [
            game.snack.x < game.head.x, #Snack to the left
            game.snack.x > game.head.x, #Snack to the right
            game.snack.y > game.head.y, #Snack down
            game.snack.y < game.head.y  #Snack up
        ]
        state.extend(snack)
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

def save(folderPath):
    agent = Agent()
    print("Check agent",agent)
    agent.model.saveModel(folderPath) # Save the model
    agent.trainer.saveParameters( agent.plotScores, agent.plotMeanScores, agent.totalScore, agent.numberOfGames, folderPath) # Save parameters  

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
            save(CHECKPOINT_PATH)
            
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

