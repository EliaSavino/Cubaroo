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
# print(cube._convert_cubelet_facelet())
# for move in moves:
#     cube.rotate(face=move[0], clockwise=move[1])
#     print(move)
#     print(cube._convert_cubelet_facelet())
#
c = Cube()
c.assert_roundtrip()              # should pass on solved
for m in "R":
    print(f"Testing move {m}")
    print('Before move:')
    c.plot_net()
    c.rotate(m, True)
    print('After move:')
    c.plot_net()
    c.rotate(m, False)
    print('After inverse move:')
    c.plot_net()