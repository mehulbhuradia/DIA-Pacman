import matplotlib.pyplot as plt
from pacman import *
import ghostAgents
import layout
import textDisplay
import graphicsDisplay
import copy

## set up the parameters to newGame


timeout = 30
layout = layout.getLayout("mediumClassic")
pacmanType = loadAgent('ApproximateQLearningAgent', True)
numGhosts = 1
ghosts = [ghostAgents.RandomGhost(i + 1) for i in range(numGhosts)]
catchExceptions = True

numtraining = 7600
numgames = 8000


def runGames(pacman, numGames, numTraining=0):
    rules = ClassicGameRules(timeout)
    games = []

    for i in range(numGames):
        beQuiet = True
        if beQuiet:
            # Suppress output and graphics
            gameDisplay = textDisplay.NullGraphics()
            rules.quiet = True
        else:
            timeInterval = 0.001
            textDisplay.SLEEP_TIME = timeInterval
            gameDisplay = graphicsDisplay.PacmanGraphics(1.0, timeInterval)
            rules.quiet = False

        game = rules.newGame(layout, pacman, ghosts, gameDisplay, beQuiet, catchExceptions)
        game.run()
        if not i < numTraining:
            games.append(game)
        scores = [(game.state.getScore()+1000) for game in games]
    if (numGames - numTraining) > 0:
        print('Average Score:', sum(scores) / float(len(scores)))
        print('Scores:       ', ', '.join([str(score) for score in scores]))
    return scores


def runRL():
    thePacman = pacmanType(numTraining=numtraining, extractor='FeatureExtractorExp2')
    scores = runGames(thePacman, numgames, numtraining)
    plt.plot(scores)
    plt.show()
    print("Maximum score",max(scores))


runRL()
