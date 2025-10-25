'''
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

'''

import numpy as np
import pandas as pd
from src.models.tiny_transformer import TransformerQNet
from src.models.actor import build_actor
import torch
from src.cube import Cube
from src.solvers.encoders import IndexCubieEncoder
from src.solvers.cube_gym import MOVES, apply_move



if __name__ == "__main__":
    model_path = "runs/transformer_dqn_weekend/final.pt"
    torch.load(model_path)

    model = TransformerQNet()
    model.load_state_dict(torch.load(model_path))
    actor = build_actor(model, "cpu", 12)
    cube = Cube()
    print("Initial Cube:")
    cube.print_net(use_color=True)
    print(cube.is_solved())
    apply_move(cube, "U")
    print(cube.is_solved())
    print("Scramble:")
    cube.scramble()
    cube.print_net(use_color=True)

    encoders = IndexCubieEncoder()
    max_moves = 200
    move_counter = 0
    while cube.moves_since_scramble() < max_moves:
        break

        if cube.is_solved():
            print("Solved!")
            break
        encoded_cube = torch.tensor(encoders.encode(cube), dtype=torch.long).unsqueeze(0)
        step = actor.act(encoded_cube)
        move = MOVES[step]

        apply_move(cube, move)
        move_counter += 1
        print(f"Move: {move} | Score: {cube.score()}")
        cube.print_net(use_color=True)
        print(
            f"Moves since scramble: {cube.moves_since_scramble()} | "
            f"Steps taken: {move_counter} | "
            f"Steps remaining: {max_moves - move_counter}"
        )






