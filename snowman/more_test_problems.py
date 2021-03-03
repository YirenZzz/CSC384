from snowman import SnowmanState

# All tests were run with a 5 second timeout using both the manhattan distance heuristic and an alternate.  
# Benchmark results are below.

# TEST ONE
# Path lengths when testing regular best first search using manhattan distance as the heuristic:
# [32, -99, 33, 106, 53, -99, -99, 120, 98, -99]

# Path lengths when testing regular best first search using an alternate heuristic:
# [46, 237, 33, 67, 69, -99, -99, 113, -99, -99]

# TEST TWO
# Path lengths when testing Anytime GBFS using manhattan distance as the heuristic:
# [30, -99, 33, 68, 53, -99, -99, 110, 88, -99]

# Path lengths when testing Anytime GBFS using an alternate heuristic:
# [30, 110, 33, 57, 45, -99, -99, 83, -99, -99]

# TEST THREE
# Path lengths when testing Anytime Weighted A Star using manhattan distance as the heuristic: 
# [30, -99, 33, 52, 39, -99, -99, 73, 86, -99]

# Path lengths when testing Anytime Weighted A Star using an alternate heuristic:
# [30, -99, 33, 50, 45, -99, -99, 73, -99, -99]

ADDITIONAL_PROBLEMS = (
SnowmanState("START", 0, None, 6, 6, (2, 2), {(2, 1): 0, (4, 3): 1, (3, 3): 2}, frozenset(((3, 0), (5, 1), (1, 2), (1, 3), (2, 3), (5, 3))), (4, 1)), 
SnowmanState("START", 0, None, 10, 4, (8, 3), {(1, 2): 0, (8, 1): 1, (5, 1): 2}, frozenset(((5, 0), (0, 1), (4, 2), (5, 2), (6, 2), (4, 3), (5, 3))), (4, 1)), 
SnowmanState("START", 0, None, 8, 4, (0, 1), {(2, 2): 0, (4, 1): 1, (2, 1): 2}, frozenset(((0, 0), (3, 0), (6, 1), (7, 1), (3, 2), (6, 2), (7, 2), (6, 3), (7, 3))), (3, 1)), 
SnowmanState("START", 0, None, 9, 5, (7, 0), {(3, 2): 0, (7, 2): 1, (5, 2): 2}, frozenset(((4, 0), (0, 1), (7, 1), (0, 2), (0, 3), (2, 3), (4, 3), (5, 4), (6, 4))), (6, 1)), 
SnowmanState("START", 0, None, 9, 6, (4, 5), {(4, 2): 0, (6, 1): 1, (1, 4): 2}, frozenset(((4, 0), (5, 0), (1, 1), (1, 2), (8, 2), (1, 3), (2, 3), (6, 3), (7, 3), (8, 3), (6, 4), (7, 4), (8, 4), (0, 5), (5, 5), (6, 5), (7, 5), (8, 5))), (5, 4)), 
SnowmanState("START", 0, None, 12, 5, (6, 2), {(1, 2): 0, (4, 3): 1, (8, 2): 2}, frozenset(((0, 0), (1, 0), (3, 0), (4, 0), (6, 0), (7, 0), (8, 0), (9, 0), (0, 1), (3, 2), (4, 2), (7, 2), (10, 2), (11, 2), (7, 3), (10, 3), (11, 3), (0, 4), (5, 4), (9, 4), (10, 4), (11, 4))), (2, 2)), 
SnowmanState("START", 0, None, 8, 8, (6, 2), {(5, 2): 0, (7, 4): 1, (3, 4): 2}, frozenset(((3, 0), (4, 0), (5, 0), (6, 0), (1, 1), (3, 1), (4, 1), (5, 1), (6, 1), (0, 3), (4, 3), (6, 4), (5, 5), (3, 6), (4, 6), (7, 6))), (5, 4)), 
SnowmanState("START", 0, None, 7, 8, (2, 3), {(2, 1): 0, (5, 3): 1, (4, 5): 2}, frozenset(((6, 0), (6, 1), (0, 3), (1, 3), (4, 3), (0, 4), (1, 4), (2, 4), (1, 5), (2, 5), (2, 6), (3, 6), (4, 6))), (4, 4)), 
SnowmanState("START", 0, None, 6, 11, (4, 4), {(2, 1): 0, (1, 8): 1, (1, 9): 2}, frozenset(((3, 0), (0, 1), (3, 3), (0, 5), (1, 5), (3, 5), (4, 5), (5, 5), (0, 6), (1, 6), (4, 6), (5, 6), (0, 9))), (0, 4)), 
SnowmanState("START", 0, None, 10, 11, (3, 7), {(8, 4): 0, (8, 2): 1, (3, 6): 2}, frozenset(((0, 0), (2, 0), (6, 0), (7, 0), (8, 0), (9, 0), (0, 1), (4, 2), (4, 3), (7, 3), (9, 3), (0, 4), (1, 4), (2, 4), (5, 4), (6, 4), (1, 5), (6, 5), (1, 6), (2, 6), (1, 7), (4, 7), (5, 7), (1, 8), (6, 8), (3, 9), (2, 10), (3, 10), (5, 10), (6, 10))), (0, 5)), 
)

if __name__ == '__main__':
    for i, p in enumerate(ADDITIONAL_PROBLEMS):
        print(f'Test No: {(i + 1)}: ----------------------------------------')
        p.print_state()
        print()
