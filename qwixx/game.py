from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from random import randint

from tqdm import tqdm

from .player import Player


class Color(Enum):
    RED = 'red'
    YELLOW = 'yellow'
    GREEN = 'green'
    BLUE = 'blue'


@dataclass
class Action:
    color: Color
    n: int


def dice_throw() -> int:
    return randint(1, 6)


class Scoreboard:
    def __init__(self):
        self.red = []
        self.yellow = []
        self.green = []
        self.blue = []
        self.mapping = dict(zip(Color, [self.red, self.yellow, self.green, self.blue]))
        self.crossed = {color: False for color in Color}
        self.crosses = 0

    def __getitem__(self, color: Color):
        return self.mapping[color]

    @property
    def score(self) -> int:
        s = 0
        for color in Color:
            s += sum(range(1, len(self.mapping[color]) + 1 + self.crossed[color]))
        s -= 5 * self.crosses
        return s

    def is_legal(self, action: Action) -> bool:
        if action.color == Color.RED:
            return action.n > max(self.red, default=1)
        if action.color == Color.YELLOW:
            return action.n > max(self.yellow, default=1)
        if action.color == Color.GREEN:
            return action.n < min(self.green, default=13)
        if action.color == Color.BLUE:
            return action.n < min(self.blue, default=13)
        return False

    def move(self, action: Action):
        self.mapping[action.color].append(action.n)


class Qwixx:
    def __init__(self, players: list[Player]):
        self.n_players = len(players)
        self.scoreboards = [Scoreboard() for _ in range(self.n_players)]
        self.players = players
        self.crossed: set[Color] = set()
        self.turn = 0

    def play(self) -> list[int]:
        while not self.is_finished():
            self.move()
        return [scoreboard.score for scoreboard in self.scoreboards]

    def is_finished(self):
        return len(self.crossed) >= 2 or any(scoreboard.crosses == 4 for scoreboard in self.scoreboards)

    def filter(self, actions: list[Action], scoreboard: Scoreboard):
        if len(actions) == 0:
            return True
        if len(actions) == 1:
            return scoreboard.is_legal(actions[0])
        if len(actions) == 2:
            if actions[0].color == actions[1].color:
                if actions[0].color == Color.RED and actions[0].n >= actions[1].n:
                    return False
                if actions[0].color == Color.YELLOW and actions[0].n >= actions[1].n:
                    return False
                if actions[0].color == Color.GREEN and actions[0].n <= actions[1].n:
                    return False
                if actions[0].color == Color.BLUE and actions[0].n <= actions[1].n:
                    return False
            return all(scoreboard.is_legal(action) for action in actions)

    def update_crossed(self):
        for color in Color:
            if color in self.crossed:
                continue
            for scoreboard in self.scoreboards:
                if len(scoreboard[color]) >= 6 and scoreboard[color][-1] in [2, 12]:
                    scoreboard.crossed[color] = True
                    self.crossed.add(color)

    def move(self):
        colorless_throws = [dice_throw(), dice_throw()]
        colored_throws = {color: dice_throw() for color in Color}

        actions_phase_1 = [Action(color, sum(colorless_throws)) for color in Color if color not in self.crossed]

        actions_phase_2 = []
        for colorless_throw in set(colorless_throws):
            for color in Color:
                if color in self.crossed:
                    continue
                actions_phase_2.append(Action(color, colorless_throw + colored_throws[color]))

        main_actions = [[action] for action in actions_phase_2]
        for action_1 in actions_phase_1:
            for action_2 in actions_phase_2:
                main_actions.append([action_1, action_2])
            main_actions.append([action_1])
        main_actions.append([])

        side_actions = [[action] for action in actions_phase_1]
        side_actions.append([])

        sb_copy = deepcopy(self.scoreboards)
        for i, (player, scoreboard) in enumerate(zip(self.players, self.scoreboards)):
            scoreboards = deepcopy(sb_copy[:i] + sb_copy[i+1:])
            sbs = (deepcopy(scoreboard), deepcopy(scoreboards))
            if i == self.turn:
                player_main_actions = [actions for actions in main_actions if self.filter(actions, scoreboard)]
                actions = player.move(deepcopy(player_main_actions), True, *sbs)
                if actions not in player_main_actions:
                    actions = []
                if len(actions) == 0:
                    scoreboard.crosses += 1

                for action in actions:
                    scoreboard.move(action)

            else:
                player_side_actions = [actions for actions in side_actions if self.filter(actions, scoreboard)]
                actions = player.move(deepcopy(player_side_actions), False, *sbs)
                if actions not in player_side_actions:
                    actions = []
                for action in actions:
                    scoreboard.move(action)

        self.turn += 1
        self.turn %= len(self.players)
        self.update_crossed()


def compete(agent1, agent2, n_games: int = 1000):
    wins = draws = losses = 0
    # for i in tqdm(range(n_games), desc='Actively competing!'):
    for i in range(n_games):
        if i % 2 == 0:
            score = Qwixx([agent1, agent2]).play()
            if score[0] > score[1]:
                wins += 1
            elif score[1] > score[0]:
                losses += 1
            else:
                draws += 1
        else:
            score = Qwixx([agent2, agent1]).play()
            if score[1] > score[0]:
                wins += 1
            elif score[0] > score[1]:
                losses += 1
            else:
                draws += 1

    return wins, draws, losses
