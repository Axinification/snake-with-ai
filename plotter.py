import matplotlib.pyplot as plt

def plot(scores, meanScores):
    plt.ion() # Interactive mode
    plt.pause(0.0001)
    plt.cla() # Clear current lines
    plt.pause(0.0001)
    plt.xlabel('Number of Games')
    plt.pause(0.0001) 
    plt.ylabel('Score')
    plt.pause(0.0001) 
    plt.title('Training')
    plt.pause(0.0001) 
    plt.grid(True)
    plt.pause(0.0001)  
    plt.plot(scores) # Plot Scores
    plt.pause(0.0001)  
    plt.plot(meanScores) # Plot Mean Scores
    plt.pause(0.0001)    
    plt.gca() # Get current lines
    plt.pause(0.0001)  
    plt.draw() # Draw
    plt.pause(0.0001)  
    # plt.text(len(scores)-1, scores[-1], str(scores[-1])) # Set the display of scores
    # plt.text(len(meanScores)-1, meanScores[-1], str(meanScores[-1])) # Set the display of mean scores