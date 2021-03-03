#   Look for #IMPLEMENT tags in this file. These tags indicate what has
#   to be implemented to complete the warehouse domain.

#   You may add only standard python imports---i.e., ones that are automatically
#   available on TEACH.CS
#   You may not remove any imports.
#   You may not import or otherwise source any of your own files

import os
from search import * #for search engines
from snowman import SnowmanState, Direction, snowman_goal_state #for snowball specific classes
from test_problems import PROBLEMS #20 test problems

import math
from timeit import default_timer as timer
import sys

def heur_manhattan_distance(state):
#IMPLEMENT
    '''admissible sokoban puzzle heuristic: manhattan distance'''
    '''INPUT: a snowman state'''
    '''OUTPUT: a numeric value that serves as an estimate of the distance of the state to the goal.'''
    #We want an admissible heuristic, which is an optimistic heuristic.
    #It must never overestimate the cost to get from the current state to the goal.
    #The sum of the Manhattan distances between each snowball that has yet to be stored and the storage point is such a heuristic.
    #When calculating distances, assume there are no obstacles on the grid.
    #You should implement this heuristic function exactly, even if it is tempting to improve it.
    #Your function should return a numeric value; this is the estimate of the distance to the goal.

    #Manhattan Distance = |x1-x0| + |y1-y0|
    distance = 0
    destination_x=state.destination[0]
    destination_y=state.destination[1]

    for snow in state.snowballs:
        distance += abs(snow[0] - destination_x) + abs(snow[1] - destination_y)

    return distance

#HEURISTICS
def trivial_heuristic(state):
    '''trivial admissible snowball heuristic'''
    '''INPUT: a snowball state'''
    '''OUTPUT: a numeric value that serves as an estimate of the distance of the state (# of moves required to get) to the goal.'''   
    return len(state.snowballs)


def heur_alternate(state):
#IMPLEMENT
    '''a better heuristic'''
    '''INPUT: a sokoban state'''
    '''OUTPUT: a numeric value that serves as an estimate of the distance of the state to the goal.'''
    #heur_manhattan_distance has flaws.
    #Write a heuristic function that improves upon heur_manhattan_distance to estimate distance between the current state and the goal.
    #Your function should return a numeric value for the estimate of the distance to the goal.

    
    cornor=((0, state.height - 1), (state.width - 1, state.height - 1), (state.width - 1, 0), (0,0))
    #initialize variables
    distance = 0

    destination_x=state.destination[0]
    destination_y=state.destination[1]
    snow = state.snowballs
    robot_x = state.robot[0]
    robot_y = state.robot[1]
    prev=0
    min_rot=float("inf")
    for snowball in state.snowballs:
      
        if snowball != state.destination:
            #when the snowball at the four corner of the board
            if snowball in cornor:
                return float("inf")
            
            #when the snowball move one step will reach the wall, but the destination is not on that wall
            if((snowball[0] == state.width - 1 or snowball[0] == 0) and snowball[0]-destination_x !=0):
                return float("inf")
              
            elif((snowball[1] == state.height - 1 or snowball[1] == 0) and snowball[1]-destination_y !=0):
                return float("inf")
            
            #when the snowball 
            if((((snowball[1] == 0) or ((snowball[0], snowball[1] + 1) in state.obstacles)) 
                or ((snowball[1] == state.height - 1) or ((snowball[0], snowball[1] - 1) in state.obstacles))) 
               and (((snowball[0] == 0) or ((snowball[0] - 1, snowball[1]) in state.obstacles)) or 
                    ((snowball[0] == state.width - 1) or ((snowball[0] + 1, snowball[1]) in state.obstacles)))):
              
                return float("inf")
                 
            
         
            #for single snowball, the big snowball has the higher priority
            if(snow[snowball] ==0):
                distance += (abs(snowball[0] - destination_x) + abs(snowball[1] - destination_y))/2            
            
            if(snow[snowball] ==2 or snow[snowball] ==1):
                distance += (abs(snowball[0] - destination_x) + abs(snowball[1] - destination_y))
            #if two snowball stack together, two ways needed to move them to the destination, so x2
            elif(snow[snowball]==4 or snow[snowball]==5 or snow[snowball]==3):
                distance += 2*(abs(snowball[0] - destination_x) + abs(snowball[1] - destination_y))
            #if three snowball stack together, two ways needed to move them to the destination, so distance x3  
            elif(snow[snowball]==6):
                distance += 3*(abs(snowball[0] - destination_x) + abs(snowball[1] - destination_y))     
            
        elif (snowball == state.destination):
            #if the snowball already build on destination, no need to calculate distance
            if snow[snowball] == 6:
                return 0
    
            elif(snow[snowball] == 0 or snow[snowball] == 3):
                distance += 0  
            #if the snowball are on the destination but do not have the bms order 
            else:
                distance +=(math.sqrt((destination_x - robot_x) **2 + (destination_x - robot_y) **2))
        
        cur_rot=abs(robot_x - snowball[0]) + abs(robot_y - snowball[1])
        if min_rot>cur_rot:
            min_rot=cur_rot
    #the min distance between robot and each snowball
    robot_dis = min_rot           
    total_dis = distance+robot_dis

    return total_dis
  
def heur_zero(state):
    '''Zero Heuristic can be used to make A* search perform uniform cost search'''
    return 0

def fval_function(sN, weight):
#IMPLEMENT
    """
    Provide a custom formula for f-value computation for Anytime Weighted A star.
    Returns the fval of the state contained in the sNode.

    @param sNode sN: A search node (containing a SokobanState)
    @param float weight: Weight given by Anytime Weighted A star
    @rtype: float
    """
  
    #Many searches will explore nodes (or states) that are ordered by their f-value.
    #For UCS, the fvalue is the same as the gval of the state. For best-first search, the fvalue is the hval of the state.
    #You can use this function to create an alternate f-value for states; this must be a function of the state and the weight.
    #The function must return a numeric f-value.
    #The value will determine your state's position on the Frontier list during a 'custom' search.
    #You must initialize your search engine object as a 'custom' search engine if you supply a custom fval function.
    
    return sN.gval + (weight * sN.hval)

def anytime_weighted_astar(initial_state, heur_fn, weight=1.0, timebound = 5):
    #IMPLEMENT
    '''Provides an implementation of anytime weighted a-star, as described in the HW1 handout'''
    '''INPUT: a sokoban state that represents the start state and a timebound (number of seconds)'''
    '''OUTPUT: A goal state (if a goal is found), else False'''
    '''implementation of weighted astar algorithm'''

    
    time = os.times()[0]
    time_diff = 0

    fval_fn = (lambda sN: fval_function(sN, weight))
    search_engine = SearchEngine(strategy='custom', cc_level= 'default')
    search_engine.init_search(initial_state, snowman_goal_state, heur_fn, fval_fn)

    # Initialize cost bounds and variables
    cost_bound = (float("inf"), float("inf"), float("inf"))
    best = False
    result = search_engine.search(timebound)


    
    while time_diff < timebound -2.5:
        if result == False: 
            return best
  
        end_time = os.times()[0]
        time_diff = end_time - time
  
        if (result.gval <= cost_bound[0]):
            cost_bound = (result.gval, float('inf'), float('inf'))
            best = result
  
        result = search_engine.search(timebound - time_diff, cost_bound)

    return best



def anytime_gbfs(initial_state, heur_fn, timebound = 5):
    #IMPLEMENT
    '''Provides an implementation of anytime greedy best-first search, as described in the HW1 handout'''
    '''INPUT: a sokoban state that represents the start state and a timebound (number of seconds)'''
    '''OUTPUT: A goal state (if a goal is found), else False'''
    '''implementation of weighted astar algorithm'''

    time = os.times()[0]
    time_diff = 0

    search_engine = SearchEngine(strategy='best_first', cc_level='default')
    search_engine.init_search(initial_state, snowman_goal_state, heur_fn)

    cost_bound = (float("inf"), float("inf"), float("inf"))
    best = False
    result = search_engine.search(timebound)
    

    while time_diff  < timebound - 2.5:
        if result == False:         
            return best
  
        end_time = os.times()[0]
        time_diff = end_time - time
  
        if (result.gval <= cost_bound[0]):
            cost_bound = (result.gval, float('inf'), float('inf'))
            best = result
        result = search_engine.search(timebound - time_diff, cost_bound)
    return best

