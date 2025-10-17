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
for m in "RRLLUUDDFFBB":
    # print([str(corn) for corn in c.corners])
    # print(c.to_facelets())
    print(f"Move: {m}\n")
    c.rotate(m, clockwise=True)
    c.print_net(use_color=True)
    # c.plot_3d()
    # c.rotate(m, clockwise=False)
    print("\n")
    # print([str(corn) for corn in c.corners])
    # print(c.to_facelets())
    # c.test_move(m)
c.plot_3d()
c.print_net(use_color=True)
# for m in "BBFFLLRRDDUU":
#     # print([str(corn) for corn in c.corners])
#     # print(c.to_facelets())
#     print(f"Move: {m}\n")
#     c.rotate(m, clockwise=False)
#     c.print_net(use_color=True)
#     # c.rotate(m, clockwise=False)
#     print("\n")
#     # print([str(corn) for corn in c.corners])
#     # print(c.to_facelets())
#     # c.test_move(m)
# c.print_net(use_color=True)
# c.plot_net()

# slight improvement, but still not there. U and D work ok. R, L and F seem to work ok, except R, L flip the last face (i = 5) meaning if i do R on face 5 the left column rotates. and if i do L the right one rotates in the last face. F
