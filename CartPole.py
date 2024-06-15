#####################
# I used Tensorflow version 2.9.2
# gym 0.17.0
# python 2
#####################

####################
# I used Epsilon greedy approach and SARSA leaning algorithm for this environment
####################

####################
# The problem with SARSA algorithm is we can't store action values for all states in this environment
# Because there are infinity possiable states
# we need to set the upper and lower limits for the environment and discretize it.
# So, the best option for such problems is using neural networks
####################

####################
# the algorithm we used to train the data
# S -- state
# A -- Action
# alpha = step size
# gamma = discount rate
# S_next -- next state
# Q(S,A) -- action value function
# Q_max(S) -- maximum value in action value function of state S

# Q(S,A) = Q(S,A) + alpha*(R + gamma*Q_max(S_next) - Q(S,A))
# The above one is used when S_next is not terminal step
#
# Q(S,A) = Q(S,A) + alpha*(R  - Q(S,A))
# if S_next is terminal step
#####################

####################
#For the first few itreations while algorithm is learning the game
# we select random action at a state S
#
# After some itreations
# we select the action if the action value is greater than epsilon
# if not we will select random action
#
# For the next few Iterations
# we keep decresing the value of epsilon for every iteration
#
# By this algorithm learns how to balance the stick
#
####################
import gym
import numpy as np
import time


class Q_Learning:

    def _init_(self, env, alpha, gamma, epsilon, numberEpisodes, numberOfBins, lowerBounds, upperBounds):
        self.env = env
        # alpha is step size
        self.alpha = alpha
        # gamma is discount rate
        self.gamma = gamma
        #epsilon -- used in epsilon greddy apporach
        self.epsilon = epsilon

        self.actionNumber = env.action_space.n
        self.numberEpisodes = numberEpisodes
        # number of discrete points in position, velocity, angle, angular velocity
        self.numberOfBins = numberOfBins
        # upper and lower limits of the system/cartpole
        self.lowerBounds = lowerBounds
        self.upperBounds = upperBounds

        # this list stores sum of rewards in every learning episode
        self.sumRewardsEpisode = []

        # this matrix is the action value function matrix
        self.Qmatrix = np.random.uniform(low=0, high=1, size=(
        numberOfBins[0], numberOfBins[1], numberOfBins[2], numberOfBins[3], self.actionNumber))

    ###
    #This function returns the discreted state that when we pass a arbitary state
    ###
    def returnIndexState(self, state):
        position = state[0]
        velocity = state[1]
        angle = state[2]
        angularVelocity = state[3]

        cartPositionBin = np.linspace(self.lowerBounds[0], self.upperBounds[0], self.numberOfBins[0])
        cartVelocityBin = np.linspace(self.lowerBounds[1], self.upperBounds[1], self.numberOfBins[1])
        poleAngleBin = np.linspace(self.lowerBounds[2], self.upperBounds[2], self.numberOfBins[2])
        poleAngleVelocityBin = np.linspace(self.lowerBounds[3], self.upperBounds[3], self.numberOfBins[3])

        indexPosition = np.maximum(np.digitize(state[0], cartPositionBin) - 1, 0)
        indexVelocity = np.maximum(np.digitize(state[1], cartVelocityBin) - 1, 0)
        indexAngle = np.maximum(np.digitize(state[2], poleAngleBin) - 1, 0)
        indexAngularVelocity = np.maximum(np.digitize(state[3], poleAngleVelocityBin) - 1, 0)

        return tuple([indexPosition, indexVelocity, indexAngle, indexAngularVelocity])

    ###
    # This function return the action that is need to be taken in a particular state
    ###
    def selectAction(self, state, index):

        # first 500 episodes we select completely random actions to have enough exploration
        if index < 500:
            return np.random.choice(self.actionNumber)

            # Returns a random real number in the half-open interval [0.0, 1.0)
        # this number is used for the epsilon greedy approach
        randomNumber = np.random.random()

        # after 7000 episodes, we slowly start to decrease the epsilon parameter
        if index > 7000:
            self.epsilon = 0.999 * self.epsilon

        # if this condition is satisfied, we are exploring, that is, we select random actions
        if randomNumber < self.epsilon:
            # returns a random action selected from: 0,1,...,actionNumber-1
            return np.random.choice(self.actionNumber)

            # otherwise, we are selecting greedy actions
        else:
            return np.random.choice(np.where(
                self.Qmatrix[self.returnIndexState(state)] == np.max(self.Qmatrix[self.returnIndexState(state)]))[0])

    ###
    # This function simulates environment. Here the system/algorithm learns how to play by constantly updation action values in Q matrix
    def simulateEpisodes(self):
        # here we loop through the episodes
        for indexEpisode in range(self.numberEpisodes):

            # list that stores rewards per episode
            rewardsEpisode = []

            # reset the environment at the beginning of every episode
            stateS = self.env.reset()
            stateS = list(stateS)

            print("Simulating episode {}".format(indexEpisode))

            # here we step from one state to another
            # this will loop until a terminal state is reached
            terminalState = False
            while not terminalState:
                # return a discretized index of the state
                stateSIndex = self.returnIndexState(stateS)

                # select an action on the basis of the current state, denoted by stateS
                actionA = self.selectAction(stateS, indexEpisode)

                # here we step and return the state, reward, and boolean denoting if the state is a terminal state
                # prime means that it is the next state
                stateSprime, reward, terminalState, _ = self.env.step(actionA)

                rewardsEpisode.append(reward)
                stateSprime = list(stateSprime)
                stateSprimeIndex = self.returnIndexState(stateSprime)

                # return the max value, we do not need actionAprime...
                QmaxPrime = np.max(self.Qmatrix[stateSprimeIndex])

                if not terminalState:
                    # stateS+(actionA,) - we use this notation to append the tuples
                    # for example, for stateS=(0,0,0,1) and actionA=(1,0)
                    # we have stateS+(actionA,)=(0,0,0,1,0)
                    error = reward + self.gamma * QmaxPrime - self.Qmatrix[stateSIndex + (actionA,)]
                    self.Qmatrix[stateSIndex + (actionA,)] += self.alpha * error
                else:
                    # in the terminal state, we have Qmatrix[stateSprime,actionAprime]=0
                    error = reward - self.Qmatrix[stateSIndex + (actionA,)]
                    self.Qmatrix[stateSIndex + (actionA,)] += self.alpha * error

                # set the current state to the next state
                stateS = stateSprime

            print("Sum of rewards {}".format(np.sum(rewardsEpisode)))
            self.sumRewardsEpisode.append(np.sum(rewardsEpisode))

    ###
    # This function simulate the learned strategy
    ###
    def simulateLearnedStrategy(self):
        env1 = gym.make('CartPole-v1')
        currentState = env1.reset()
        env1.render()
        timeSteps = 1000
        obtainedRewards = []

        for timeIndex in range(timeSteps):
            actionInStateS = np.argmax(self.Qmatrix[self.returnIndexState(currentState)])
            currentState, reward, terminated, _ = env1.step(actionInStateS)
            obtainedRewards.append(reward)
            env1.render()
            if terminated:
                break

        return obtainedRewards, env1

# making cartpole envirnoment
env = gym.make('CartPole-v1')
state = env.reset()

# here define the parameters for state discretization
upperBounds = env.observation_space.high
lowerBounds = env.observation_space.low

# keeping max and min values for cart velocity
# we do this because, by deafult max and min velocities are (- infinity, infinity) , but we cant divide a interval of size infinity into intervals
# so we keep max and min values for velocities
cartVelocityMin = -6
cartVelocityMax = 6

# same case for angular velocity
poleAngleVelocityMin = -10
poleAngleVelocityMax = 10

# changing upper and lower bounds of cart velocity and angular velocity
upperBounds[1] = cartVelocityMax
upperBounds[3] = poleAngleVelocityMax
lowerBounds[1] = cartVelocityMin
lowerBounds[3] = poleAngleVelocityMin

# selecting how many intervals to divide x, velocity, angle, angular velocity
numberOfBinsPosition = 30
numberOfBinsVelocity = 30
numberOfBinsAngle = 30
numberOfBinsAngleVelocity = 30
numberOfBins = [numberOfBinsPosition, numberOfBinsVelocity, numberOfBinsAngle, numberOfBinsAngleVelocity]

# define the parameters
alpha = 0.1
gamma = 1
epsilon = 0.2
numberEpisodes = 10000

# create an object
Q1 = Q_Learning(env, alpha, gamma, epsilon, numberEpisodes, numberOfBins, lowerBounds, upperBounds)
# run the Q-Learning algorithm
Q1.simulateEpisodes()
# simulate the learned strategy
for i in range(100):
    (obtainedRewardsOptimal, env1) = Q1.simulateLearnedStrategy()

# close the environment
env1.close()
