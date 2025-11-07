"""
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

"""
import sys

import numpy as np
import pandas as pd
from src.cube import Cube
from src.scorer import Scorer, ScoringOption
from tests.test_functions import make_f2l,make_first_layer,make_top_cross_permuted,make_top_cross_in_place,make_bottom_cross_in_place

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
scorers = {option: Scorer(option=option) for option in ScoringOption}
functions = {"Solved":lambda: Cube(), "TopCross": make_top_cross_in_place, "TopCrossScramble": make_top_cross_permuted,
             'F2L': make_f2l, "first_layer": make_first_layer, "bottom_cross": make_bottom_cross_in_place}



# c.scramble()
c.print_net(use_color=True)
for name, func in functions.items():
    print(name)
    c = func()
    c.print_net(use_color=True)
    print({option: s(c) for option, s in scorers.items()})

sys.exit()
while True:
    value = input("move:")
    if value == "q":
        break

    valid = ["U", "U'", "D","D'", "F", "F'", "B","B'", "R", "R'", "L", "L'" ]
    if value not in valid:
        print("Invalid move")
        continue

    lab = value[0].upper()
    rot = not(len(value) > 1 and value[1] == "'")
    print(f"lab: {lab}, rot: {rot}")
    c.rotate(lab, rot)
    c.print_net(use_color=True)
    print(scorer(c))

print(f"Final score: {scorer(c)} ")
c.plot_3d()
# print(c.score())
# for m in "RRLLUUDDFFBB":
#     # print([str(corn) for corn in c.corners])
#     # print(c.to_facelets())
#     print(f"Move: {m}\n")
#     c.rotate(m, clockwise=True)
#     # c.print_net(use_color=True)
#     # c.plot_3d()
#     # c.rotate(m, clockwise=False)
#     print("\n")
#     # print([str(corn) for corn in c.corners])
#     # print(c.to_facelets())
#     # c.test_move(m)
#
# print(c.get_history())
# print(c.score())
# # c.plot_3d()
