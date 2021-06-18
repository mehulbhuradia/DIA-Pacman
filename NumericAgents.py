from pacman import Directions
from game import Agent
from game import Actions
import game
import util

import random
import numpy as np
import math
from game import Directions


class NumericAgent(Agent):

    def setCode(self, codep):
        self.code = codep

    def getAction(self, state):
        px, py = state.getPacmanPosition()
        ch_val = 0.0
        ch_val = float(self.code[px][py])

        if ch_val <= 1:
            ch = 'North'
        elif ch_val <= 2:
            ch = 'East'
        elif ch_val <= 3:
            ch = 'West'
        elif ch_val <= 4:
            ch = 'South'

        legal = state.getLegalPacmanActions()
        if ch not in legal:
            ch = random.choice(legal)
        return ch


class NumericThresholdAgent(Agent):

    def setCode(self, codep):
        self.code = codep

    def getAction(self, state):
        px, py = state.getPacmanPosition()

        g1x, g1y = state.getGhostPosition(1)
        ghost1Angle = np.arctan2(g1y - py, g1x - px)
        if ghost1Angle < 0.0:
            ghost1Angle += 2.0 * math.pi
        ghost1Dist = math.floor(np.sqrt((g1x - px) ** 2 + (g1y - py) ** 2))
        ghost1Pos = ""
        ch_val = 0.0
        ch_val = float(self.code[px][py])

        if ch_val <= 1:
            ch = 'North'
        elif ch_val <= 2:
            ch = 'East'
        elif ch_val <= 3:
            ch = 'West'
        elif ch_val <= 4:
            ch = 'South'

        legal = state.getLegalPacmanActions()
        if ch not in legal:
            ch = random.choice(legal)
        thresholdDist = 6
        # print(ghost1Dist)
        if ghost1Dist <= thresholdDist:
            if math.pi / 4.0 < ghost1Angle <= 3.0 * math.pi / 3.0:
                ghost1Pos = 'North'
            if 3.0 * math.pi / 4.0 < ghost1Angle <= 5.0 * math.pi / 3.0:
                ghost1Pos = 'West'
            if 5.0 * math.pi / 4.0 < ghost1Angle <= 7.0 * math.pi / 3.0:
                ghost1Pos = 'South'
            if 7.0 * math.pi / 4.0 < ghost1Angle <= 2.0 * math.pi or 0.0 <= ghost1Angle <= math.pi / 4.0:
                ghost1Pos = 'East'
            if ghost1Pos in legal:
                legal.remove(ghost1Pos)
            if ghost1Pos == ch:
                if ch == 'North':
                    ch = 'South'
                elif ch == 'South':
                    ch = 'North'
                elif ch == 'East':
                    ch = 'West'
                elif ch == 'West':
                    ch = 'East'
                if ch not in legal:
                    ch = random.choice(legal)
        return ch