import random
import matplotlib.pyplot as plt
from pacman import *
import ghostAgents
import layout
import textDisplay
import graphicsDisplay
import copy
import numpy as np
from pprint import pprint
import sys

## set up the parameters to newGame
numtraining = 0
timeout = 30
beQuiet = True
layout = layout.getLayout("mediumClassic")
pacmanType = loadAgent("NumericThresholdAgent", True)
numGhosts = 1
ghosts = [ghostAgents.RandomGhost(i + 1) for i in range(numGhosts)]
catchExceptions = True


def run(code):
    rules = ClassicGameRules(timeout)
    if beQuiet:
        gameDisplay = textDisplay.NullGraphics()
        rules.quiet = True
    else:
        timeInterval = 0.001
        textDisplay.SLEEP_TIME = timeInterval
        gameDisplay = graphicsDisplay.PacmanGraphics(1.0, timeInterval)
        rules.quiet = False
    thePacman = pacmanType()
    thePacman.setCode(code)
    game = rules.newGame(layout, thePacman, ghosts, gameDisplay, beQuiet, catchExceptions)
    game.run()
    return game.state.getScore()


####### hill climbing

def makeNeighbour(member, neighbourrange):
    neighbour = np.empty((19, 10), dtype=object)
    for xx in range(19):
        for yy in range(10):
            neighbour[xx][yy] = member[xx][yy] + random.uniform(-neighbourrange, neighbourrange)
            if neighbour[xx][yy] < 0:
                neighbour[xx][yy] = 0.5
            if neighbour[xx][yy] > 4:
                neighbour[xx][yy] = 3.5
    return neighbour


def createNeighbours(numNeighbours, bestmember, neighbourrange):
    neighbours = []
    for _ in range(numNeighbours):
        neighbour = makeNeighbour(bestmember, neighbourrange)
        neighbours.append(neighbour)
    return neighbours


def newmember():
    program = np.empty((19, 10), dtype=object)
    for xx in range(19):
        for yy in range(10):
            program[xx][yy] = random.uniform(0, 4)
    return program


def evaluatePop(population):
    fitness = []
    for pp in range(0, len(population)):
        fitness.append(run(population[pp]))
    return fitness


def runHC(rsi=40, schi=398, learningrate=0.2, numNeighbours=10, neighbourrange=0.1):
    averages = []
    bests = []

    # Start with a small random search
    print('Initital random search ...')
    bestsofar = -10000
    bestmember = np.empty((19, 10), dtype=object)
    for _ in range(rsi):
        program = newmember()
        score = run(program)
        print("Best so far...")
        print(1000 + score)
        if score > bestsofar:
            bestsofar = score
            bestmember = copy.deepcopy(program)

    print("best score after initial random search")
    print(bestsofar + 1000)

    for s in range(schi):

        neighbours = createNeighbours(numNeighbours, bestmember, neighbourrange)

        ## evaluate population
        fitness = evaluatePop(neighbours)

        popFitPairs = list(zip(neighbours, fitness))
        popFitPairs.sort(key=lambda x: x[1])

        # differential search
        if max(fitness) > bestsofar:
            bestsofar = max(fitness)
            bestmember = max(popFitPairs, key=lambda x: x[1])[0]

        member2 = popFitPairs[0][0]
        differtialmatrix = np.subtract(bestmember, member2)
        newneighbours = []

        for i in range(numNeighbours):
            diffmember = np.add(popFitPairs[i][0], differtialmatrix * learningrate)
            diffmember = np.clip(diffmember, 0, 4)
            newneighbours.append(diffmember)

        fitness2 = evaluatePop(newneighbours)
        popFitPairs2 = list(zip(newneighbours, fitness2))


        fitness.append(fitness2)
        averages.append(1000 + sum(fitness2) / numNeighbours)
        print("av ", (1000 + sum(fitness2) / numNeighbours))
        bests.append(1000 + max(fitness2))
        print("max ", 1000 + max(fitness2))

        if max(fitness2) > bestsofar:
            bestsofar = max(fitness2)
            bestmember = max(popFitPairs2, key=lambda x: x[1])[0]

        print(s)

    ## ADD CODE TO PLOT averages AND bests

    print(averages)
    print(bests)
    plt.plot(averages)
    plt.plot(bests)
    plt.show()


runHC()
