# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)

        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        "*** YOUR CODE HERE ***"
        Food_List = newFood.asList()
        minFoodDistance = float("inf")
        minghostDistance = float("inf")

        for food in Food_List:
            dist = util.manhattanDistance(newPos, food)
            if dist < minFoodDistance:
                minFoodDistance = dist

        for ghosts in newGhostStates:
            ghostPosition = ghosts.getPosition()
            dist = util.manhattanDistance(newPos, ghostPosition)
            if dist < minghostDistance:
                minghostDistance = dist

        if (ghosts.getPosition() == newPos or ghostPosition[0] == newPos[0] or ghostPosition[1] == newPos[1]):
            minghostDistance = -float("inf")

        score = minghostDistance / minFoodDistance + successorGameState.getScore()
        return score


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"

        v = maxValue1(self, gameState, 0, self.depth)

        return v[1]


def maxValue1(self, gameState, index, curr_depth):
    if curr_depth == 0 or gameState.isWin() or gameState.isLose():
        return " ",self.evaluationFunction(gameState)

    val = [" ", -float("inf")]

    actions = gameState.getLegalActions(index)
    scores = [minValue1(self, gameState.generateSuccessor(index, action), index+1, curr_depth) for action in actions]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]



    return bestScore, actions[bestIndices[0]]


def minValue1(self, gameState, index, curr_depth):
    if curr_depth == 0 or gameState.isWin() or gameState.isLose():
        return " ", self.evaluationFunction(gameState)

    val = " ", float("inf")

    actions = gameState.getLegalActions(index)
    if index != gameState.getNumAgents()-1:
        scores = [minValue1(self, gameState.generateSuccessor(index, action), index + 1, curr_depth) for action in
                  actions]
    else:
        scores = [maxValue1(self, gameState.generateSuccessor(index, action), 0, curr_depth - 1) for action in actions]
    bestScore = min(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]


    return bestScore,actions[bestIndices[0]]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        index=0
        v = maxValue(self, gameState, -float("inf"), float("inf"), self.depth,index)
        return v[0]

        util.raiseNotDefined()

def maxValue(self, State, a, b, cur_depth,index):
    if cur_depth==0 or State.isWin() or State.isLose():
        return " ", self.evaluationFunction(State)

    val = [" ", -float("inf")]
    move = None
    for action in State.getLegalActions(0):

        v = minValue(self, State.generateSuccessor(0, action), a, b, cur_depth,index+1)

        if v[1] > val[1]:
            val[1] = v[1]
            move = action
        if val[1] > b:
            return move, val[1]
        a = max(a, val[1])
    return move, val[1]

def minValue(self, state, a, b, cur_depth,index):
    if cur_depth==0 or state.isWin() or state.isLose():
        return " ", self.evaluationFunction(state)

    val = [" ", float("inf")]
    move = None
    for action in state.getLegalActions(index):
        if index == state.getNumAgents()-1:
            v = maxValue(self, state.generateSuccessor(index, action), a, b,
                         cur_depth -1,0)

        else:
            v = minValue(self, state.generateSuccessor(index, action), a, b,
                         cur_depth,index+1)
        if v[1] < val[1]:
            move = action
            val[1] = v[1]

        if val[1] < a:
            return move, val[1]
        b = min(b, val[1])

    return move, val[1]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        index = 0
        v = maxVal(self, gameState, self.depth, index)
        return v[0]

        util.raiseNotDefined()

def maxVal(self, State, cur_depth, index):
    if cur_depth == 0 or State.isWin() or State.isLose():
        return " ", self.evaluationFunction(State)

    val = [" ", -float("inf")]
    move = None
    for action in State.getLegalActions(0):

        v = minVal(self, State.generateSuccessor(0, action), cur_depth, index + 1)

        if v[1] > val[1]:
            val[1] = v[1]
            move = action

    return move, val[1]

def minVal(self, state, cur_depth, index):
    if cur_depth == 0 or state.isWin() or state.isLose():
        return " ", self.evaluationFunction(state)

    val = [" ", float("inf")]
    vp=0
    move = None
    actions=state.getLegalActions(index)
    p = 1.0 / float(len(actions))
    for action in actions:
        if index == state.getNumAgents() - 1:
            v = maxVal(self, state.generateSuccessor(index, action), cur_depth - 1, 0)
            vp += p*v[1]
        else:
            v = minVal(self, state.generateSuccessor(index, action), cur_depth, index + 1)
            vp += p * v[1]

        if v[1] < val[1]:
            move = action
            val[1] = v[1]


    return move, vp


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"


    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    capsules=currentGameState.getCapsules()
    "*** YOUR CODE HERE ***"
    if currentGameState.isLose():
        return -10000
    if currentGameState.isWin():
        return 10000
    Food_List = newFood.asList()
    mincapDistance=10000
    minFoodDistance = 10000
    minghostDistance = 10000

    for food in Food_List:
        dist = util.manhattanDistance(newPos, food)
        if dist < minFoodDistance:
            minFoodDistance = dist

    for ghosts in newGhostStates:
        ghostPosition = ghosts.getPosition()
        dist = util.manhattanDistance(newPos, ghostPosition)
        if dist < minghostDistance:
            minghostDistance = dist

    for capsule in capsules:

        dist =util.manhattanDistance(newPos,capsule)
        if(dist<mincapDistance):
            mincapDistance=dist

    if (ghosts.getPosition() == newPos):
        minghostDistance = -10000

    score = (minghostDistance / (minFoodDistance*50*mincapDistance)) + currentGameState.getScore()
    return score

# Abbreviation
better = betterEvaluationFunction

