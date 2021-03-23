from math import gamma
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class LinearQNet(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super().__init__()

        self.linear1 = nn.Linear(inputSize, hiddenSize)
        self.linear2 = nn.Linear(hiddenSize, outputSize)

    def forward(self, input):
        input = F.relu(self.linear1(input))
        input = self.linear2(input)
        return input
    
    # Saving the model
    def saveModel(self, folder, fileName='model.pth'):
        modelFolderPath = folder # Model will be saved in the 'model' folder
        if not os.path.exists(modelFolderPath):
            os.makedirs(modelFolderPath) # Creating the folder
        fileName = os.path.join(modelFolderPath, fileName) # Setting filepath for the model
        torch.save(self.state_dict(), fileName) # Saving the model


class QTrainer:
    def __init__(self, model, learningRate, gamma):
        self.model = model # Defining model
        self.learningRate = learningRate # Defining learning rate
        self.gamma = gamma # Defining discount rate
        self.optimizer = optim.Adam(model.parameters(), self.learningRate) # Optimization using pyTorch
        self.criterion = nn.MSELoss() # Loss function

    def saveParameters(self, scores, meanScores, totalScore, numberOfGames, folder, fileName='checkpoint.pth'):
        parametersFolderPath = folder
        if not os.path.exists(parametersFolderPath):
            os.makedirs(parametersFolderPath) # Creating the folder
        fileName = os.path.join(parametersFolderPath, fileName) # Setting filepath for the model
        
        checkpoint = {
            'scores': scores,
            'meanScores': meanScores,
            'totalScore': totalScore,
            'numberOfGames': numberOfGames,
            'learningRate': self.learningRate,
            'gamma': self.gamma,
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(checkpoint, fileName)

    # def loadParameters(self, model, learningRate, gamma, optimizer):
    #     self.model = LinearQNet(11, 256, 3)
    #     self.learningRate = learningRate
    #     self.gamma = gamma
    #     self.optimizer = optimizer
    #     self.model.load_state_dict(model)
    #     self.model.eval()

    def trainStep(self, state, action, reward, nextState, gameOver):
        # Convert to tensors
        state = torch.tensor(state, dtype=torch.float) 
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        nextState = torch.tensor(nextState, dtype=torch.float)

        # Check if 1 dimensional
        if len(state.shape) == 1:
            # Append 1 dimension
            state = torch.unsqueeze(state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            nextState = torch.unsqueeze(nextState, 0)
            gameOver = (gameOver, )

        # 1: Predicted Q values with the current state
        prediction = self.model(state)
        # 2: newQ = R + y * max(Next predicted Q value) -> Only do this when game is not over
        # In order to do that we need to clone prediction in order to obtain the same format
        target = prediction.clone()
        for index in range(len(gameOver)):
            newQ = reward[index]
            if not gameOver[index]:
                # newQ = R + y * max
                newQ = reward[index] + self.gamma * torch.max(self.model(nextState[index]))
            # Assign new Q value to the cloned Q
            target[index][torch.argmax(action[index]).item()] = newQ
        
        #Emptying the gradients
        self.optimizer.zero_grad()
        # Loss calculated by mean square value -> loss = (newQ - Q)^2
        loss = self.criterion(target, prediction)
        #Back propagation
        loss.backward()
        #Optimizing after each step
        self.optimizer.step()