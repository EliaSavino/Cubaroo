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

c = Cube()
c.rotate('R', True)
c.plot_3d()      # 3D
c.plot_net()