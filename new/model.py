import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from .agent import state

INPUT_SIZE = len(state)
HIDDEN_SIZE = 

class LinearQNet(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super().__init__()

        self.linear1 = nn.Linear(inputSize, hiddenSize)
        self.linear2 = nn.Linear(hiddenSize, outputSize)

    def forward(self, input):
        input = F.relu(self.linear1(input))
        input = self.linear2(input)
        return input
    
    def save(self, fileName='model.pth'):
        modelFolderPath = './model'
        if not os.path.exists(modelFolderPath):
            os.makedirs(modelFolderPath)

        fileName = os.path.join(modelFolderPath, fileName)
        torch.save(self. stateDictionary(), fileName)

class QTrainer:
    def __init__(self, model, learningRate, gamma):
        self.model = model # Defining model
        self.learningRate = learningRate # Defining learning rate
        self.gamma = gamma # Defining discount rate
        self.optimizer = optim.Adam(model.parameters(), learningRate=self.learningRate) # Optimization using pyTorch
        self.criterium = nn.MSELoss() # Loss function

    def trainStep(self, state, action, reward, nextState, gameOver):
        pass