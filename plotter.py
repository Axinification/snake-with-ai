import matplotlib.pyplot as plt
import numpy

def plotSetup():
    plt.ion()
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.title('Training')
    plt.xlim(xmin=0) # Set the minimum x value
    plt.ylim(ymin=0) # Set the minimum y value
    plt.grid(True)
    plt.pause(0.0001)  
    plt.plot([]) # Plot Scores
    plt.pause(0.0001)  
    plt.plot([])
    plt.draw()



def plotUpdate(scores, meanScores):
    plt.ion() # Interactive mode
    plt.cla() # Clear current lines
    plt.pause(0.0001)  
    plt.plot(scores) # Plot Scores
    plt.pause(0.0001)  
    plt.plot(meanScores) # Plot Mean Scores
    plt.pause(0.0001)    
    # plt.xlabel('Number of Games')
    # plt.ylabel('Score')
    # plt.title('Training')
    plt.grid(True) # Set grid
    plt.pause(0.0001) 
    # plt.xlim(xmin=0) # Set the minimum x value
    # plt.ylim(ymin=0) # Set the minimum y value
    # plt.gcf()
    plt.gca() # Get current lines
    plt.pause(0.0001)  
    plt.draw() # Draw
    plt.pause(0.0001)  
    # plt.savefig("test.png")
    # plt.text(len(scores)-1, scores[-1], str(scores[-1])) # Set the display of scores
    # plt.text(len(meanScores)-1, meanScores[-1], str(meanScores[-1])) # Set the display of mean scores


# import matplotlib.pyplot as plt

# x = [0,1,6,12,0,2,4,5,6,6,3,6,8,2]
# y = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]

# plt.ion()
# def plotSetup():
    # plt.ion() # Turn the interface on
    # plt.gcf()
    # plt.title('Training') # Set the plot title
    # plt.xlabel('Number of Games') # Set the x axis label
    # plt.ylabel('Score') # Set the y axis label
    # plt.ylim(ymin=0) # Set the minimum y value
    # plt.plot(x,y)
    # plt.tight_layout()
    # plt.show()


# plt.plot(x,y)
# plt.tight_layout()
# plt.show()

# def plot(scores, meanScores):
    # plt.clf() # Clear current figure
    # plt.title('Training') # Set the plot title
    # plt.xlabel('Number of Games') # Set the x axis label
    # plt.ylabel('Score') # Set the y axis label
    # plt.plot(scores[:]) # Plot scores
    # plt.plot(meanScores[:]) # Plot mean scores
    # plt.ylim(ymin=0) # Set the minimum y value
    # plt.text(len(scores)-1, scores[-1], str(scores[-1])) # Set the display of scores
    # plt.text(len(meanScores)-1, meanScores[-1], str(meanScores[-1])) # Set the display of mean scores
    # plt.show()