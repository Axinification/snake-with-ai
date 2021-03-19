# snake-with-ai

Gra snake - 100%
Wyliczanie potrzebnych danych - 100%
Wizualizacja danych w grze - 100%
Napisanie sieci neuronowej - 0%
Machine Learning - 0%

Installed dependencies:
pip install pygame
pip intall torch torchvision
pip install matplotlib ipython

In this project Im going to use the DeepQ Learning which is an extension of Reinforced Learning

The game is a reconstructed version used in previous project (old folder), reconstruction is done to optimize the game for AI and is based on https://github.com/python-engineer/python-fun/tree/master/snake-pygame

Evironement for learning will be the game itself (PyGame)
    Environement will contain game logic in loop and give out appropriate points for actions

Model is going to be implemented in PyTorchs Linear_QNet (PyTorch)
    Model predicts state and on the basis of that is returning an action

Agent combines information returned from Environement and Model to train the neural network
    Training will consist of training loop with this logic:
    - Based on the game, next state is calculated
    - Based on the state an action is returned
    - The next move is then predicted (model.predict())
    - We will get next play step b prediction
    - Based on this information we will get 3 values
        reward, game_over state and score
    - With this information we calculate new state
    - We will remember this information and use it to train the model (model.train())

Reward system used:
    - eating snack: +10
    - game over: -10
    - else: 0

Actions: 
    (for Left Right version)
    - [1,0,0] straight
    - [0,1,0] right turn
    - [0,0,1] left turn

    (for WASD version)
    - [1,0,0,0] direction left
    - [0,1,0,0] direction right
    - [0,0,1,0] direction up
    - [0,0,0,1] direction down



State: (11 values)
    - Danger: [straight, left, right] (3)
    - Direction: [left, right, up, down] (4)
    - Snack: [left, right, up, down] (4)


    MODEL:
    (for Left Right version)
    - Input Neurons -> 11
    - Hidden Layer ->
    - Output -> 3

    (for WASD version)
    - Input Neurons -> 11
    - Hidden Layer ->
    - Output -> 4


DeepQ Learning Basics:
    Q Value: Quality of Action

Process:
Loop    0. Init Q Value = init model
    |-->1. Choose Action (model.predict(state)) // or random
    |   2. Perform Action
    |   3. Measure Reward
    |<--4. Update Q Value + train model

    The point of looping is to improve the Quality of model,
    starting from randomn moves to calculated ones

Loss Function: Bellman Equation
    s - State
    a - Action
    α - Learning Rate
    γ - Discount Rate
    NewQ(s,a) - New Q value for given state and action
    Q(s,a) - Current Q value
    R(s,a) - Reward for taking given action at given state
    maxQ'(s',a') - Maximum expected future reward given the new s' and all possible actions at that new state

    NewQ(s,a) = Q(s,a) + α*[R(s,a) + γ*maxQ'(s',a') - Q(s,a)] 

    Q = model.predict(first state)
    Qnew = R + γ*max(Q(predicted state))

    loss = (Qnew - Q)^2 - Mean square error

