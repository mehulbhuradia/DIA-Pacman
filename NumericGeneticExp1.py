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
pacmanType = loadAgent("NumericAgent", True)
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


####### genetic algorithm


def cross(parent1, parent2):
    crossoverchild = np.empty((19, 10), dtype=object)
    # chossing a random crossover point
    crossoverx = random.randrange(19)
    crossovery = random.randrange(10)
    crossoverchild[0:crossoverx, 0:crossovery] = parent1[0:crossoverx, 0:crossovery]
    crossoverchild[0:crossoverx, crossovery:10] = parent2[0:crossoverx, crossovery:10]
    crossoverchild[crossoverx:19, 0:crossovery] = parent2[crossoverx:19, 0:crossovery]
    crossoverchild[crossoverx:19, crossovery:10] = parent1[crossoverx:19, crossovery:10]
    return crossoverchild


def mutate(parentp, numberOfMutations):
    mchild = copy.deepcopy(parentp)
    for _ in range(numberOfMutations):
        xx = random.randrange(19)
        yy = random.randrange(10)
        mchild[xx][yy] = random.uniform(0, 4)
    return mchild


def newmember():
    program = np.empty((19, 10), dtype=object)
    for xx in range(19):
        for yy in range(10):
            program[xx][yy] = random.uniform(0, 4)
    return program


def newpop(popSiz):
    population = []
    for _ in range(popSiz):
        program = newmember()
        population.append(program)
    return population


def evaluatePop(population):
    fitness = []
    for pp in range(0, len(population)):
        fitness.append(run(population[pp]))
    return fitness


def runGA(popSiz=20, timescale=400, tournamentSize=7, numberOfMutations=5):
    averages = []
    bests = []

    ## create random initial population
    population = newpop(popSiz)

    print("Beginning Evolution")
    # start the generations
    for t in range(timescale):

        ## evaluate population
        fitness = evaluatePop(population)

        averages.append(1000 + sum(fitness) / popSiz)
        print("av ", 1000 + sum(fitness) / popSiz)
        bests.append(1000 + max(fitness))
        print("max ", 1000 + max(fitness))

        popFitPairs = list(zip(population, fitness))
        newPopulation = []

        halfPop = int(popSiz / 2)

        for i in range(halfPop):

            # select two parents using tournaments
            tournament1 = random.sample(popFitPairs, tournamentSize)
            parent1 = max(tournament1, key=lambda x: x[1])[0]

            tournament2 = random.sample(popFitPairs, tournamentSize)
            parent2 = max(tournament2, key=lambda x: x[1])[0]

            # crossover the selected parents
            crossover_child = cross(parent1, parent2)

            # mutate the child
            child = mutate(crossover_child, numberOfMutations)

            newPopulation.append(child)

            # add the best half of the population to the next generation
            newPopulation.append(sorted(popFitPairs, key=lambda x: x[1])[i][0])

        print(t)
        population = copy.deepcopy(newPopulation)

    ## ADD CODE TO PLOT averages AND bests

    print(averages)
    print(bests)
    plt.plot(averages)
    plt.plot(bests)
    plt.show()

runGA()
