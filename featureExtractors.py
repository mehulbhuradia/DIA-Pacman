# featureExtractors.py
# --------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"Feature extractors for Pacman game states"

from game import Directions, Actions
import util
import math
import numpy as np


# returns the distance from the closest food
def closestFood(pos, food, walls):
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a food at this location then exit
        if food[pos_x][pos_y]:
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist + 1))
    # no food found
    return None


# returns the distance from the closest ghost
def closestGhost(pos, gx, gy, walls):
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a ghost at this location then exit
        if pos_x == gx and pos_y == gy:
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist + 1))
    # no ghost found
    return None


class FeatureExtractor:
    """
    Returns simple features for a basic Pacman that eats food but doesnt avoid ghosts:
    - whether food will be eaten
    - how far away the next food is
    """

    def getFeatures(self, state, action):
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()
        features = util.Counter()

        features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        features["eats-food"] = 1.0

        dist = closestFood((next_x, next_y), food, walls)
        if dist is not None:
            # make the distance a number less than one otherwise the update will diverge wildly
            features["closest-food"] = float(dist) / (walls.width * walls.height)
        features.divideAll(10.0)
        return features


class FeatureExtractorExp2:
    """
    Returns simple features for a basic reflex Pacman:
    - whether food will be eaten
    - how far away the next food is
    - whether a ghost collision is imminent
    - whether a ghost is one step away
    - whether a ghost is scared
    - whether a scared ghost collision is imminent
    - whether a ghost scared ghost is one step away
    """

    def getFeatures(self, state, action):
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()

        features = util.Counter()

        features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        ghostState = state.getGhostState(1)
        isScared = ghostState.scaredTimer > 0

        # count the number of ghosts 1-step away
        features["#-of-ghosts-1-step-away"] = sum(
            (next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)

        # if there is no danger of ghosts and they arent scared then add the food feature
        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0
        else:
            features["eats-food"] = 0

        features["scared"] = 0
        features["#-of-scared-ghosts-1-step-away"] = 0
        features["closest-scared-ghost"] = 0

        # the distances are a number less than one so that the update doesn't diverge wildly
        fdist = closestFood((next_x, next_y), food, walls)
        if fdist is not None:
            features["closest-food"] = float(fdist) / (walls.width * walls.height)

        gx, gy = state.getGhostPosition(1)
        gdist = closestGhost((next_x, next_y), gx, gy, walls)

        if isScared:
            features["scared"] = 1
            features["#-of-scared-ghosts-1-step-away"] = features["#-of-ghosts-1-step-away"]
            features["#-of-ghosts-1-step-away"] = 0
            if gdist is not None:
                features["closest-scared-ghost"] = gdist
        features.divideAll(10.0)
        return features
