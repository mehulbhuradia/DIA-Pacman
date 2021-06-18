from game import *
from featureExtractors import *
import random, util, time


class ApproximateQLearningAgent(Agent):
    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, extractor='SimpleExtractor'):
        self.index = 0  # This is always Pacman
        self.numTraining = int(numTraining)
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)
        self.discount = float(gamma)
        self.accumTrainRewards = 0.0
        self.accumTestRewards = 0.0
        self.episodesSoFar = 0
        self.episodeStartTime = time.time()
        self.lastWindowAccumRewards = 0.0
        self.featExtractor = util.lookup(extractor, globals())()
        self.qValues = util.Counter()
        self.weights = util.Counter()

    def final(self, state):
        deltaReward = state.getScore() - self.lastState.getScore()
        self.observeTransition(self.lastState, self.lastAction, state, deltaReward)
        if self.episodesSoFar < self.numTraining:
            self.accumTrainRewards += self.episodeRewards
        else:
            self.accumTestRewards += self.episodeRewards
        self.episodesSoFar += 1
        if self.episodesSoFar >= self.numTraining:
            # Take off the training wheels
            self.epsilon = 0.0  # no exploration
            self.alpha = 0.0  # no learning

        self.lastWindowAccumRewards += state.getScore()

        NUM_EPS_UPDATE = 100
        if self.episodesSoFar % NUM_EPS_UPDATE == 0:
            print('Reinforcement Learning Status:')
            windowAvg = self.lastWindowAccumRewards / float(NUM_EPS_UPDATE)
            if self.episodesSoFar <= self.numTraining:
                trainAvg = self.accumTrainRewards / float(self.episodesSoFar)
                print('\tCompleted %d out of %d training episodes' % (
                    self.episodesSoFar, self.numTraining))
                print('\tAverage Rewards over all training: %.2f' % (
                    trainAvg))
            else:
                testAvg = float(self.accumTestRewards) / (self.episodesSoFar - self.numTraining)
                print('\tCompleted %d test episodes' % (self.episodesSoFar - self.numTraining))
                print('\tAverage Rewards over testing: %.2f' % testAvg)
            print('\tAverage Rewards for last %d episodes: %.2f' % (
                NUM_EPS_UPDATE, windowAvg))
            print('\tEpisode took %.2f seconds' % (time.time() - self.episodeStartTime))
            self.lastWindowAccumRewards = 0.0
            self.episodeStartTime = time.time()

        if self.episodesSoFar == self.numTraining:
            msg = 'Training Done (turning off epsilon and alpha)'
            print('%s\n%s' % (msg, '-' * len(msg)))

    def getAction(self, state):

        # Pick Action
        possibleActions = state.getLegalActions()
        action = None
        if possibleActions:
            if util.flipCoin(self.epsilon) == True:
                action = random.choice(possibleActions)
            else:
                action = self.getPolicy(state)
        self.doAction(state, action)
        return action

    def getPolicy(self, state):
        possibleActions = state.getLegalActions()
        if possibleActions:
            maxv = float("-inf")
            bestAction = None
            for action in possibleActions:
                q = self.getQValue(state, action)
                if q >= maxv:
                    maxv = q
                    bestAction = action
            return bestAction
        return None

    def getValue(self, state):
        possibleActions = state.getLegalActions()
        if possibleActions:
            maxv = float("-inf")
            for action in possibleActions:
                q = self.getQValue(state, action)
                if q >= maxv:
                    maxv = q
            return maxv
        return 0.0

    def doAction(self, state, action):
        self.lastState = state
        self.lastAction = action

    def observationFunction(self, state):
        if not self.lastState is None:
            reward = state.getScore() - self.lastState.getScore()
            self.observeTransition(self.lastState, self.lastAction, state, reward)
        return state

    def registerInitialState(self, state):
        self.lastState = None
        self.lastAction = None
        self.episodeRewards = 0.0
        if self.episodesSoFar == 0:
            print('Beginning %d episodes of Training' % (self.numTraining))

    def observeTransition(self, state, action, nextState, deltaReward):
        self.episodeRewards += deltaReward
        f = self.featExtractor.getFeatures(state, action)
        alphadiff = self.alpha * (
                (deltaReward + self.discount * self.getValue(nextState)) - self.getQValue(state, action))
        for feature in f.keys():
            self.weights[feature] = self.weights[feature] + alphadiff * f[feature]

    def getQValue(self, state, action):
        qValue = 0.0
        features = self.featExtractor.getFeatures(state, action)
        for key in features.keys():
            qValue += (self.weights[key] * features[key])
        return qValue

    def update(self, state, action, nextState, reward):
        features = self.featExtractor.getFeatures(state, action)
        possibleStateQValues = []
        for act in self.getLegalActions(state):
            possibleStateQValues.append(self.getQValue(state, act))
        for key in features.keys():
            self.weights[key] += self.alpha * (reward + self.discount * (
                    (1 - self.epsilon) * self.getValue(nextState) + (self.epsilon / len(possibleStateQValues)) * (
                sum(possibleStateQValues))) - self.getQValue(state, action)) * features[key]
