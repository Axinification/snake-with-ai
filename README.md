# snake-with-ai

Installed dependencies:  
pip install pygame  
pip intall torch torchvision  
pip install matplotlib ipython  

In this project Im going to use the DeepQ Learning which is an extension of Reinforced Learning  

The game is a reconstructed version used in previous project,  
reconstruction is done to optimize the game for AI and is based on  
https://github.com/python-engineer/python-fun/tree/master/snake-pygame 

Most of the learning process can be changed via variables.py file.  

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

Reward system used: those values change over time (see variables.py)  
    - eating snack: +30  
    - game over: -20  
    - else: 0  

Actions:  
    (for WASD version)  
    - [1,0,0,0] direction left  
    - [0,1,0,0] direction right  
    - [0,0,1,0] direction up  
    - [0,0,0,1] direction down  

Actions are based on clockwork rotation of axis and is calculated by using modulo of 4 to get the directions.  
    Modulo used for iteration -> (2+1)%4 = 3 -> (3+1)%4 = 0  
For use case see environmentAI.py line 133  


State: (Amount of inputs is based on version selected)  

    Those 2 types of input are used by every version of the model as they comprise the core mechanics 
    for reward system and are hard to change
    - Direction: [left, right, up, down] (4)
    - Danger: [left-backward, left, left-forward, straight, 
                    right-forward, right, right-backward] (7)

    Versions that are using those inputs are: [f, fs]
    - Danger far: [left-backward-far, left-far, left-forward-far, straight-far, 
                    right-forward-far, right-far, right-backward-far] (7)
    
    Versions that are using those inputs are: [f, s]
    - Snack: [left, right, up, down] (4) 

    MODEL:
    - Input Neurons -> 11 / 15 / 18 / 22 (Amount of inputs is based on chosen version)
    - Hidden Layer -> 1 / 2 / 3 (Amount of hidden layers is set at the start)
    - Output -> 4 (Output is always 4 as it tells the snake in which direction to go next)


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

Changable values of the learning process:  
    - LEARNING_RATE # The rate of bias change, the less the slower and more precise the learning  
    - EPSILON_DELTA # 0 randomness after x games  
    - GAMMA # Has to be less than 1. Lower discount rate strives for quick rewards and higher for the long term ones.  
    - GAMMA_INCREMENT # Amount of gamma increment if dynamic change is enabled.  
    For more changeable values see variables.py  

UML Class Diagram  

![UML class](https://user-images.githubusercontent.com/73855075/118899013-2072b080-b90e-11eb-8111-3d9100f62257.png)
