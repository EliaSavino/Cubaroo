'''
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

'''

import numpy as np
import pandas as pd
from src.cube import Cube

moves = [('R', True), ('R', False), ('L', True), ('L', False), ('U', True), ('U', False), ('D', False), ('D', True),
         ('F', True), ('F', False), ('B', False), ('B', True)]

cube = Cube()
# cube._test_move('L')
# exit()
for move in moves:
    cube.plot()
    cube.rotate(face=move[0], clockwise=move[1])

cube.plot()