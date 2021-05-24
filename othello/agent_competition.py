"""
An AI player for Othello. 
"""
#description of my heuristic
#1. Stable position (Corners & X and C squares & Edges)
#The four corners are the most important position that need to occupy, so increase these positions' weight
#The opponent can take the corner when one of three position that adjacent to the corner is occupied, so lower these positions' weight
#The opponent cannot surround edge positions, so increase these positions' weight

#2. Mobilitiy
#Count the number of moves that the player can make for the current board.

#3. In the opening, mid-game, and end-game, use different weight strategy.


import random
import sys
import time
cache_board = {}
# You can use the functions in othello_shared to write your AI
from othello_shared import find_lines, get_possible_moves, get_score, play_move


def eprint(*args, **kwargs): #you can use this for debugging, as it will print to sterr and not stdout
    print(*args, file=sys.stderr, **kwargs)
    
# Method to compute utility value of terminal state
def compute_utility(board, color):
    #IMPLEMENT

    #The function get_score(board) returns a tuple (number of dark disks, number of light disks).
    dark_disks, light_disks = get_score(board)
   
    # The utility should be calculated as the number of disks of the player's colour minus the number of disks of the opponent.
    
    #if AI plays color 1
    if color == 1:
        utility = dark_disks - light_disks
    else: 
        utility = light_disks - dark_disks 
        
    return utility


#count the possible moves for the player
def compute_mobility(board, color):
    return len(get_possible_moves(board, color))

#count for corner position
def compute_corners(board, color):
    count=0
    corners = [board[0][0], board[0][-1], board[-1][-1], board[-1][0]]
    for i in corners:
        if i==color:
            count+=1
    
    return count

    
#count for x and c squares
def compute_xc_squares(board, color):
    
    #check if one of three position that adjacent to the corner is occupied
    count=0
    #left top corner
    if board[1][0] == color or board[1][1] == color or board[0][1] == color:
        count += 1
    #left bottom corner
    if board[-2][-1] == color or board[-2][-2] == color or board[-1][-2] == color:
        count += 1    
    #right bottom corner
    if board[0][-2] == color or board[1][-2] == color or board[1][-1] == color:
        count += 1    
    #rigth top corner 
    if board[-2][0] == color or board[-2][1] == color or board[-1][1] == color:
        count += 1
    
    return count


#count for edge position
def compute_edge(board, color):
    edge_count=0
    frontier_count=0
    for i in range(2, len(board)-2):
        for j in range(2, len(board)-2):
            if board[0][i]==color or board[i][0]==color or board[-1][i]==color or board[i][-1]==color:
                edge_count+=1
            else:
                if (board[i-1][j]==color or board[i+1][j] == color 
                    or board[i][j+1] == color or board[i][j-1] == color):
                    frontier_count -= 1
                    
    return edge_count, frontier_count

# Better heuristic value of board 
def compute_heuristic(board, color): 
    #IMPLEMENT    
    
    #1. Stable position (Corners & X and C squares & Edges)
    #The four corners are the most important position that need to occupy, so increase these positions' weight
    #The opponent can take the corner when one of three position that adjacent to the corner is occupied, so lower these positions' weight
    #The opponent cannot surround edge positions, so increase these positions' weight
    
    #2. Mobilitiy
    #Count the number of moves that the player can make for the current board.
    
    #3. In the opening, mid-game, and end-game, use different weight strategy. 
    
    corners=compute_corners(board, color)

    xc_squares=-compute_xc_squares(board, color)
    edge_count, frontier_count=compute_edge(board, color)
    
    utility = compute_utility(board, color)*2
    mobilitiy = compute_mobility(board,color)
    
    game=get_score(board)[0]+get_score(board)[1]
    open_game = len(board) // 3
    mid_game =  len(board) // 3*2
    end_game =  len(board)
    
    
    if game < open_game:
        corners=corners*200
        xc_squares=xc_squares*100
        edge_count=edge_count*50
        frontier_count=frontier_count
    elif game > open_game and game <= mid_game:
        corners=corners*210
        xc_squares=xc_squares*90
        edge_count=edge_count*55
        frontier_count=frontier_count
        
    else:
        corners=corners*220
        xc_squares=xc_squares*80
        edge_count=edge_count*60
        frontier_count=frontier_count
    
   
    return corners+xc_squares+edge_count+utility+mobilitiy



############ MINIMAX ###############################
def minimax_min_node(board, color, limit, caching=0):
    # IMPLEMENT
    
    if caching==1 and (board, color) in cache_board:
        return cache_board[(board, color)]
    
    opponent_color = 1
    if color == 1:
        opponent_color = 2    

    actions = get_possible_moves(board, opponent_color)
    if not actions or limit == 0:
        if caching:
            cache_board[(board, color)] = (None, compute_utility(board, color))
        return (None, compute_utility(board, color))
    
    best_move = None
    value = float('inf')
    
    for move in actions:
        nxt_board = play_move(board, opponent_color, move[0], move[1])
        nxt_move, nxt_val = minimax_max_node(nxt_board, color, limit-1)
        if value > nxt_val:
            value, best_move = nxt_val, move
            
    if caching==1:
        cache_board[(board, color)] = (best_move, value)

    return (best_move, value)


def minimax_max_node(board, color, limit, caching=0):  # returns highest possible utility
    # IMPLEMENT
    if caching==1 and (board, color) in cache_board:
        return cache_board[(board, color)]

    actions = get_possible_moves(board, color)
    if len(actions)==0 or limit == 0:
        utility = compute_utility(board, color)
        if caching:
            cache_board[(board, color)] = (None, utility)
        return (None, utility)
    
    best_move = None
    value = float('-inf')
    for move in actions:
        new_board = play_move(board, color, move[0], move[1])
        nxt_move, nxt_val = minimax_min_node(new_board, color, limit-1)

        if value < nxt_val:
            value, best_move = nxt_val, move
            
    if caching==1:
        cache_board[(board, color)] = (best_move, value)

    return (best_move, value)

def select_move_minimax(board, color, limit, caching = 0):
    """
    Given a board and a player color, decide on a move. 
    The return value is a tuple of integers (i,j), where
    i is the column and j is the row on the board.  

    Note that other parameters are accepted by this function:
    If limit is a positive integer, your code should enfoce a depth limit that is equal to the value of the parameter.
    Search only to nodes at a depth-limit equal to the limit.  If nodes at this level are non-terminal return a heuristic 
    value (see compute_utility)
    If caching is ON (i.e. 1), use state caching to reduce the number of state evaluations.
    If caching is OFF (i.e. 0), do NOT use state caching to reduce the number of state evaluations.    
    """
    #IMPLEMENT 
    move = minimax_max_node(board, color, limit, caching)
    return move[0]


############ ALPHA-BETA PRUNING #####################
def alphabeta_min_node(board, color, alpha, beta, limit, caching = 0, ordering = 0):
    #IMPLEMENT
    if caching==1 and (board, color) in cache_board:
        return cache_board[(board, color)]
    
    opponent_color = 1
    if color == 1:
        opponent_color = 2    
    
    actions = get_possible_moves(board, opponent_color)
    
    
    if len(actions)==0 or limit == 0:
        return (None, compute_utility(board, color))
    
    best_move = None
    value = float('inf')
    sorted_actions = []
    
    
    if ordering==1:
        for mov in actions:
            new_graph = play_move(board, opponent_color, mov[0], mov[1])
            sorted_actions.append((mov, compute_utility(new_graph, color), new_graph))        
        sorted_actions.sort(key=lambda x: x[1])
    
        new_action=[]
        for utility_move in sorted_actions:
            new_action.append(utility_move[0])
                
        for move in new_action:
            nxt_board = play_move(board, opponent_color, move[0], move[1])
            nxt_move, nxt_val = alphabeta_max_node(nxt_board, color, alpha, beta, limit-1, caching, ordering)
            
            if value > nxt_val:
                value, best_move = nxt_val, move
            
            beta=min(beta, value)       
            if beta <= alpha:
                break;
    else:
        for move in actions:
            nxt_board = play_move(board, opponent_color, move[0], move[1])
            nxt_move, nxt_val = alphabeta_max_node(nxt_board, color, alpha, beta, limit-1, caching, ordering)
            
            if value > nxt_val:
                value, best_move = nxt_val, move
            
            beta=min(beta, value)       
            if beta <= alpha:
                break;        
        
    if caching==1:
        cache_board[(board, color)] = (best_move, value)

    return (best_move, value)    

def alphabeta_max_node(board, color, alpha, beta, limit, caching = 0, ordering = 0):
    #IMPLEMENT
    if caching==1 and (board, color) in cache_board:
        return cache_board[(board, color)]
    
    actions = get_possible_moves(board, color)
    
    if len(actions)==0 or limit == 0:
        return (None, compute_utility(board, color))

    best_move = None
    value = float('-inf')
    sorted_actions = []
    
    if ordering==1:   
        for mov in actions:
            new_graph = play_move(board, color, mov[0], mov[1])
            sorted_actions.append((mov, compute_utility(new_graph, color), new_graph))
        
        sorted_actions.sort(key=lambda tup: tup[1], reverse=True)  
        
        new_action=[]
        for utility_move in sorted_actions:
            new_action.append(utility_move[0])
        
        for move in new_action:
            nxt_board = play_move(board, color, move[0], move[1])
            nxt_move, nxt_val = alphabeta_min_node(nxt_board, color, alpha, beta, limit-1, caching, ordering)
            if nxt_val >  value:
                value, best_move = nxt_val, move
    
            alpha = max(alpha, value)
            if alpha >= beta:
                break        

    else:
        for move in actions:
            nxt_board = play_move(board, color, move[0], move[1])
            nxt_move, nxt_val = alphabeta_min_node(nxt_board, color, alpha, beta, limit-1, caching, ordering)
            if nxt_val >  value:
                value, best_move = nxt_val, move
    
            alpha = max(alpha, value)
            if alpha >= beta:
                break  
        
    if caching:
        cache_board[(board, color)] = (best_move, value)

    return (best_move, value)

def select_move_alphabeta(board, color, limit, caching = 0, ordering = 0):
    """
    Given a board and a player color, decide on a move. 
    The return value is a tuple of integers (i,j), where
    i is the column and j is the row on the board.  

    Note that other parameters are accepted by this function:
    If limit is a positive integer, your code should enfoce a depth limit that is equal to the value of the parameter.
    Search only to nodes at a depth-limit equal to the limit.  If nodes at this level are non-terminal return a heuristic 
    value (see compute_utility)
    If caching is ON (i.e. 1), use state caching to reduce the number of state evaluations.
    If caching is OFF (i.e. 0), do NOT use state caching to reduce the number of state evaluations.    
    If ordering is ON (i.e. 1), use node ordering to expedite pruning and reduce the number of state evaluations. 
    If ordering is OFF (i.e. 0), do NOT use node ordering to expedite pruning and reduce the number of state evaluations. 
    """
    #IMPLEMENT (and replace the line below)
    alpha = float('-inf')
    beta = float('inf')
    best_move = alphabeta_max_node(board, color, alpha, beta, limit, caching, ordering)    
    return best_move[0] #change this!

####################################################
def run_ai():
    """
    This function establishes communication with the game manager.
    It first introduces itself and receives its color.
    Then it repeatedly receives the current score and current board state
    until the game is over.
    """
    print("Othello AI") # First line is the name of this AI
    arguments = input().split(",")
    
    color = int(arguments[0]) #Player color: 1 for dark (goes first), 2 for light. 
    limit = int(arguments[1]) #Depth limit
    minimax = int(arguments[2]) #Minimax or alpha beta
    caching = int(arguments[3]) #Caching 
    ordering = int(arguments[4]) #Node-ordering (for alpha-beta only)

    if (minimax == 1): eprint("Running MINIMAX")
    else: eprint("Running ALPHA-BETA")

    if (caching == 1): eprint("State Caching is ON")
    else: eprint("State Caching is OFF")

    if (ordering == 1): eprint("Node Ordering is ON")
    else: eprint("Node Ordering is OFF")

    if (limit == -1): eprint("Depth Limit is OFF")
    else: eprint("Depth Limit is ", limit)

    if (minimax == 1 and ordering == 1): eprint("Node Ordering should have no impact on Minimax")

    while True: # This is the main loop
        # Read in the current game status, for example:
        # "SCORE 2 2" or "FINAL 33 31" if the game is over.
        # The first number is the score for player 1 (dark), the second for player 2 (light)
        next_input = input()
        status, dark_disks_s, light_disks_s = next_input.strip().split()
        dark_disks = int(dark_disks_s)
        light_disks = int(light_disks_s)

        if status == "FINAL": # Game is over.
            print
        else:
            board = eval(input()) # Read in the input and turn it into a Python
                                  # object. The format is a list of rows. The
                                  # squares in each row are represented by
                                  # 0 : empty square
                                  # 1 : dark disk (player 1)
                                  # 2 : light disk (player 2)

            # Select the move and send it to the manager
            if (minimax == 1): #run this if the minimax flag is given
                movei, movej = select_move_minimax(board, color, limit, caching)
            else: #else run alphabeta
                movei, movej = select_move_alphabeta(board, color, limit, caching, ordering)
            
            print("{} {}".format(movei, movej))

if __name__ == "__main__":
    run_ai()
