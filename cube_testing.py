"""
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

"""

import numpy as np
import pandas as pd
from src.cube import Cube

moves = [
    ("R", True),
    ("R", False),
    ("L", True),
    ("L", False),
    ("U", True),
    ("U", False),
    ("D", False),
    ("D", True),
    ("F", True),
    ("F", False),
    ("B", False),
    ("B", True),
]

c = Cube()
c.scramble()
c.print_net(use_color=True)
for m in "RRLLUUDDFFBB":
    # print([str(corn) for corn in c.corners])
    # print(c.to_facelets())
    print(f"Move: {m}\n")
    c.rotate(m, clockwise=True)
    # c.print_net(use_color=True)
    # c.plot_3d()
    # c.rotate(m, clockwise=False)
    print("\n")
    # print([str(corn) for corn in c.corners])
    # print(c.to_facelets())
    # c.test_move(m)

print(c.get_history())
# c.plot_3d()
