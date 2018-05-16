from src.rl import RL
from src.enviroment import Env
import sys
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-l',
                        default='4',
                        dest='LENGTH',
                        help='input the length of the grid')

    parser.add_argument('-i',
                        default='20',
                        dest='ITERATION',
                        help='input the iteration of training')

    parser.add_argument('-d',
                        default='.1',
                        dest='DELAY',
                        help='input delay')

    args = parser.parse_args()
    
    try:
        length    = int(args.LENGTH)
        iteration = int(args.ITERATION)
    except ValueError:
        print('error: length or iteration must be an integer')
        sys.exit()

    try:
        delay = float(args.DELAY)
    except ValueError:
        print('error: delay must be an float')
        sys.exit()

    game = Env(length)
    rl   = RL(game.get_actions())

    while game.episode < iteration:
        s, done = game.restart()
        a = rl.actor(s)
        while not done:
            ns, r, done = game.go(a)
            na = rl.actor(ns)
            rl.learn(s, a, r, ns, na)
            s, a = ns, na
        print(rl.q_table)

