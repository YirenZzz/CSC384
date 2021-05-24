#Look for #IMPLEMENT tags in this file. These tags indicate what has
#to be implemented to complete the warehouse domain.  

'''
Construct and return Tenner Grid CSP models.
'''

from cspbase import *
import itertools


def tenner_csp_model_1(initial_tenner_board):
    '''Return a CSP object representing a Tenner Grid CSP problem along 
       with an array of variables for the problem. That is return

       tenner_csp, variable_array

       where tenner_csp is a csp representing tenner grid using model_1
       and variable_array is a list of lists

       [ [  ]
         [  ]
         .
         .
         .
         [  ] ]

       such that variable_array[i][j] is the Variable (object) that
       you built to represent the value to be placed in cell i,j of
       the Tenner Grid (only including the first n rows, indexed from 
       (0,0) to (n,9)) where n can be 3 to 7.
       
       
       The input board is specified as a pair (n_grid, last_row). 
       The first element in the pair is a list of n length-10 lists.
       Each of the n lists represents a row of the grid. 
       If a -1 is in the list it represents an empty cell. 
       Otherwise if a number between 0--9 is in the list then this represents a 
       pre-set board position. E.g., the board
    
       ---------------------  
       |6| |1|5|7| | | |3| |
       | |9|7| | |2|1| | | |
       | | | | | |0| | | |1|
       | |9| |0|7| |3|5|4| |
       |6| | |5| |0| | | | |
       ---------------------
       would be represented by the list of lists
       
       [[6, -1, 1, 5, 7, -1, -1, -1, 3, -1],
        [-1, 9, 7, -1, -1, 2, 1, -1, -1, -1],
        [-1, -1, -1, -1, -1, 0, -1, -1, -1, 1],
        [-1, 9, -1, 0, 7, -1, 3, 5, 4, -1],
        [6, -1, -1, 5, -1, 0, -1, -1, -1,-1]]
       
       
       This routine returns model_1 which consists of a variable for
       each cell of the board, with domain equal to {0-9} if the board
       has a 0 at that position, and domain equal {i} if the board has
       a fixed number i at that cell.
       
       model_1 contains BINARY CONSTRAINTS OF NOT-EQUAL between
       all relevant variables (e.g., all pairs of variables in the
       same row, etc.).
       model_1 also constains n-nary constraints of sum constraints for each 
       column.
    '''
    
    variable_array = []

    n_grid = initial_tenner_board[0]
    last_row = initial_tenner_board[1]
    
    var_array1=[]
    for i in range(len(n_grid)):
        variable_array += [[]]
        for j in range(len(n_grid[i])):
            if n_grid[i][j] == -1:
                var = Variable('v'+str(i)+str(j), list(range(10)))
            else:
                var = Variable('v'+str(i)+str(j), [n_grid[i][j]])
                
            variable_array[i].append(var)
            var_array1.append(var)
    
    csp = CSP('Model_1',var_array1)
    
    #check diagonals
    for i in range(len(variable_array)):
        
        for j in range(len(variable_array[i])):
            con_array=[]
             
            if (i != len(variable_array) - 1):
                con_array.append(variable_array[i + 1][j])
                if (j != len(variable_array[i]) - 1) :
                    con_array.append(variable_array[i + 1][j+1])     
                if (j !=0):
                    con_array.append(variable_array[i + 1][j-1])  
            for m in range(j+1,len(variable_array[i])):
                con_array.append(variable_array[i][m]) 
            
            for idx in con_array:
                binary = [variable_array[i][j], idx]
                c = Constraint('c', binary)
                var_tup=[]
                for v1 in variable_array[i][j].domain():
                    for v2 in idx.domain():
                        if v1!=v2:
                            var_tup.append((v1,v2))
                
                c.add_satisfying_tuples(var_tup)
                csp.add_constraint(c)                
    #column sum
    for i in range(len(variable_array[i])):
        col_sum=[]
        for j in range(len(variable_array)): 
            col_sum.append(variable_array[j][i])
            
        c = Constraint('cs' + str(i), col_sum)
        
        list_perm=[]
        for v in col_sum:
            list_perm.append(v.domain())
        
        result=[]
        for tup in itertools.product(*list_perm):
            if sum(tup) == last_row[i]:
                result.append(tup)
             
        c.add_satisfying_tuples(result)
        csp.add_constraint(c)  
   
    return csp, variable_array     
    

def tenner_csp_model_2(initial_tenner_board):
    """Return a CSP object representing a Tenner Grid CSP problem along
       with an array of variables for the problem. That is return
       tenner_csp, variable_array
       where tenner_csp is a csp representing tenner using model_1
       and variable_array is a list of lists
       [ [  ]
         [  ]
         .
         .
         .
         [  ] ]
       such that variable_array[i][j] is the Variable (object) that
       you built to represent the value to be placed in cell i,j of
       the Tenner Grid (only including the first n rows, indexed from
       (0,0) to (n,9)) where n can be 3 to 8.
       The input board takes the same input format (a list of n length-10 lists
       specifying the board as tenner_csp_model_1.
       The variables of model_2 are the same as for model_1: a variable
       for each cell of the board, with domain equal to {0-9} if the
       board has a -1 at that position, and domain equal {i} if the board
       has a fixed number i at that cell.
       However, model_2 has different constraints. In particular,
       model_2 has a combination of n-nary
       all-different constraints and binary not-equal constraints: all-different
       constraints for the variables in each row, binary constraints for
       contiguous cells (including diagonally contiguous cells), and n-nary sum
       constraints for each column.
       Each n-ary all-different constraint has more than two variables (some of
       these variables will have a single value in their domain).
       model_2 should create these all-different constraints between the relevant
       variables.
    """
    
    variable_array = []

    n_grid = initial_tenner_board[0]
    last_row = initial_tenner_board[1]
    
    var_array1=[]
    for i in range(len(n_grid)):
        variable_array += [[]]
        for j in range(len(n_grid[i])):
            if n_grid[i][j] == -1:
                var = Variable('v'+str(i)+str(j), list(range(10)))
            else:
                var = Variable('v'+str(i)+str(j), [n_grid[i][j]])
                
            variable_array[i].append(var)
            var_array1.append(var)
    
    csp = CSP('Model_2',var_array1)
    
    for i in range(len(variable_array)):
        
        for j in range(len(variable_array[i])):
            con_array=[]
            for m in range(j+1,len(variable_array[i])):
                con_array.append(variable_array[i][m])             
            if (i != len(variable_array) - 1):
                con_array.append(variable_array[i + 1][j])
                if (j != len(variable_array[i]) - 1) :
                    con_array.append(variable_array[i + 1][j+1])     
                if (j !=0):
                    con_array.append(variable_array[i + 1][j-1])  
            
            for idx in con_array:
                binary = [variable_array[i][j], idx]
                c = Constraint('c', binary)
                var_tup=[]
                for v1 in variable_array[i][j].domain():
                    for v2 in idx.domain():
                        if v1!=v2:
                            var_tup.append((v1,v2))
                
                c.add_satisfying_tuples(var_tup)
                csp.add_constraint(c)   
    
    #all-different constraints
    for i in range(len(variable_array)):
        scope = []
        varDoms=[]
        idx_array=[]
        for j in range(len(variable_array[i])): 
            if n_grid[i][j] == -1:
                scope.append(variable_array[i][j])
            else:
                idx_array.append(j)
        
        for idx in idx_array:
            if n_grid[i][idx] in list(range(10)):
                varDoms.append(n_grid[i][idx])
        
        varNew=[]
        for k in range(10):
            if k not in varDoms:
                varNew.append(k)
            
        sat_tuple=[]
        for t in itertools.permutations(varNew):
            sat_tuple.append(t)
            
        c = Constraint('cr'+str(i), scope)
        c.add_satisfying_tuples(sat_tuple)
        csp.add_constraint(c)   
    
    #column sum
    for i in range(len(variable_array[i])):
        col_sum=[]
        for j in range(len(variable_array)): 
            col_sum.append(variable_array[j][i])
            
        c = Constraint('cs' + str(i), col_sum)
        
        list_perm=[]
        for v in col_sum:
            list_perm.append(v.domain())
        
        result=[]
        for tup in itertools.product(*list_perm):
            if sum(tup) == last_row[i]:
                result.append(tup)
             
        c.add_satisfying_tuples(result)
        csp.add_constraint(c)  
   
    return csp, variable_array
