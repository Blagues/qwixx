from collections import defaultdict
from copy import deepcopy
from random import random, sample

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from main import RandomPlayer
from qwixx.game import Color, compete
from qwixx.player import Player


class MLPPlayer(Player):
    def __init__(self, saved_model=None, input_size=106, hidden_size=64, output_size=1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        if saved_model:
            self.model.load_state_dict(torch.load(saved_model, map_location=torch.device('cuda'), weights_only=True))

    def move(self, actions, is_main, scoreboard, scoreboards):
        if not actions:
            return []

        input_tensors = torch.stack(
            [torch.tensor(self.extract_features(scoreboard, scoreboards, action, is_main), dtype=torch.float32) for
             action in actions])
        with torch.no_grad():
            values = self.model(input_tensors).squeeze()

        best_action_index = values.argmax().item()
        return actions[best_action_index]

    def extract_features(self, scoreboard, scoreboards, action, is_main):
        features = []

        # Player's scoreboard features
        for color in Color:
            features.extend(self.encode_color(scoreboard[color]))
            features.append(scoreboard.crossed[color])
            features.append(len(scoreboard[color]) / 13)  # Completion percentage
            features.append(self.calculate_color_potential(scoreboard[color], color))

        # Action encoding
        action_encoding = [0] * 4
        if action:
            action_encoding[list(Color).index(action[0].color)] = 1
            features.extend(action_encoding)
            features.append(action[0].n / 13)
            features.append(self.calculate_action_value(scoreboard, action[0]))
        else:
            features.extend(action_encoding)
            features.append(0)
            features.append(0)

        # Opponents' scoreboard features
        for other_scoreboard in scoreboards:
            for color in Color:
                features.append(len(other_scoreboard[color]) / 13)
                features.append(other_scoreboard.crossed[color])
                features.append(self.calculate_color_potential(other_scoreboard[color], color))

        # Game state features
        features.append(is_main)
        features.append(scoreboard.crosses / 4)
        features.append(sum(scoreboard.crossed.values()) / 4)  # Proportion of colors crossed
        features.append(self.calculate_score_difference(scoreboard, scoreboards))

        # Add some polynomial features
        features.extend(self.create_polynomial_features(features[:20], degree=2))

        return features

    @staticmethod
    def encode_color(color_scores):
        encoding = [0] * 13
        for score in color_scores:
            encoding[score - 2] = 1
        return encoding

    @staticmethod
    def calculate_color_potential(color_scores, color):
        if color in [Color.RED, Color.YELLOW]:
            return sum(1 for i in range(2, 13) if i not in color_scores and i > max(color_scores, default=1))
        else:
            return sum(1 for i in range(2, 13) if i not in color_scores and i < min(color_scores, default=13))

    @staticmethod
    def calculate_action_value(scoreboard, action):
        color_scores = scoreboard[action.color]
        if action.color in [Color.RED, Color.YELLOW]:
            return (action.n - max(color_scores, default=1)) / 13
        else:
            return (min(color_scores, default=13) - action.n) / 13

    @staticmethod
    def calculate_score_difference(scoreboard, scoreboards):
        player_score = scoreboard.score
        avg_opponent_score = sum(sb.score for sb in scoreboards) / len(scoreboards)
        return (player_score - avg_opponent_score) / 100  # Normalize the difference

    @staticmethod
    def create_polynomial_features(features, degree=2):
        poly_features = []
        for i in range(len(features)):
            for j in range(i, len(features)):
                poly_features.append(features[i] * features[j])
        return poly_features[:20]  # Limit to 20 polynomial features to control input size


class GeneticAlgorithm:
    def __init__(self, population_size=120, mutation_rate=0.5, crossover_rate=0.6, n_generations=100):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.n_generations = n_generations
        self.population = [MLPPlayer() for _ in range(population_size)]

        self.elite_size = int(0.1 * population_size)
        self.stats = defaultdict(list)
        self.switch_to_top_players = True

        self.n_gen = 0

        plt.ion()
        self.setup_plots()

    def setup_plots(self):
        self.fig, (self.ax1, self.ax2, self.ax3, self.ax4) = plt.subplots(4, 1, figsize=(12, 24))
        self.line_max, = self.ax1.plot([], [], label='Max Fitness')
        self.line_avg, = self.ax1.plot([], [], label='Avg Fitness')
        self.line_min, = self.ax1.plot([], [], label='Min Fitness')
        self.line_top5_random, = self.ax3.plot([], [], label='Top 5 Avg Wins vs Random')
        self.line_diversity, = self.ax4.plot([], [], label='Population Diversity')

        for ax, title in zip([self.ax1, self.ax2, self.ax3, self.ax4],
                             ['Fitness over Generations', 'Average Wins vs Improved Player',
                              'Top 5 Players Average Wins', 'Population Diversity']):
            ax.set_xlabel('Generation')
            ax.set_title(title)
            ax.legend()

        self.ax1.set_ylabel('Fitness')
        self.ax2.set_ylabel('Average Wins')
        self.ax3.set_ylabel('Average Wins')
        self.ax4.set_ylabel('Diversity')

    def evolve(self):
        for generation in range(self.n_generations):
            print(f"Generation {generation + 1}")
            fitness = self.evaluate_population()

            self.update_stats(fitness)
            sorted_population = self.sort_population(fitness)

            if self.switch_to_top_players:
                self.save_best_model(sorted_population[0], generation)

            new_population = self.create_new_population(sorted_population, fitness)
            self.population = new_population[:(self.population_size - self.n_gen)]

            self.print_generation_stats(generation, fitness)
            self.update_plots()

            self.n_gen += 1

        self.print_final_stats()

    def evaluate_population(self):
        return [self.evaluate_player(player) for player in tqdm(self.population, "Evaluating population")]

    def evaluate_player(self, player, nr_matches_total=50):
        matches = nr_matches_total + self.n_gen * 3

        opponents = self.population[:10] if self.switch_to_top_players else [RandomPlayer()]
        total_score = 0

        for opponent in opponents:
            wins, draws, losses = compete(player, opponent, n_games=int(matches / len(opponents)))
            total_score += wins - losses
            if isinstance(opponent, RandomPlayer):
                self.stats['vs_random'].append((wins, draws, losses))
        return total_score

    def update_stats(self, fitness):
        self.stats['max_fitness'].append(max(fitness))
        self.stats['avg_fitness'].append(sum(fitness) / len(fitness))
        self.stats['min_fitness'].append(min(fitness))
        self.stats['top5_vs_random'].append(self.evaluate_top_n(self.population[:5], RandomPlayer()))
        self.stats['diversity'].append(self.calculate_diversity(self.population))

    def sort_population(self, fitness):
        return [x for _, x in sorted(zip(fitness, self.population), key=lambda pair: pair[0], reverse=True)]

    def create_new_population(self, sorted_population, fitness):
        new_population = sorted_population[:self.elite_size]
        while len(new_population) < self.population_size:
            parents = self.selection(sorted_population, fitness)
            offspring = self.crossover_and_mutate(parents)
            new_population.extend(offspring)
        return new_population

    def save_best_model(self, best_player, generation):
        torch.save(best_player.model.state_dict(), f'best_model_gen_{generation + 1}.pt')
        print(f"Best model from generation {generation + 1} saved.")

    def evaluate_top_n(self, top_n_players, opponent, n_games=100):
        total_wins = sum(compete(player, opponent, n_games=n_games)[0] for player in top_n_players)
        return total_wins / len(top_n_players)

    def calculate_diversity(self, population):
        distances = [
            np.sqrt(sum(
                torch.sum((p1 - p2) ** 2).item() for p1, p2 in zip(pop1.model.parameters(), pop2.model.parameters())))
            for i, pop1 in enumerate(population)
            for pop2 in population[i + 1:]
        ]
        return np.mean(distances)

    def selection(self, sorted_population, fitness):
        tournament_size = 5
        return [max(sample(list(enumerate(sorted_population)), tournament_size), key=lambda x: fitness[x[0]])[1] for _
                in range(2)]

    def crossover_and_mutate(self, parents):
        children = self.crossover(parents[0], parents[1])
        for child in children:
            self.mutate(child)
        return children

    def crossover(self, parent1, parent2):
        child1, child2 = deepcopy(parent1), deepcopy(parent2)
        if random() < self.crossover_rate:
            for p1, p2 in zip(child1.model.parameters(), child2.model.parameters()):
                mask = torch.rand(p1.shape) < 0.5
                p1.data[mask], p2.data[mask] = p2.data[mask], p1.data[mask]
        return child1, child2

    def mutate(self, child):
        for param in child.model.parameters():
            mask = torch.rand(param.shape) < self.mutation_rate
            param.data[mask] += torch.randn(param.shape)[mask] * (1e-2 / ((1 + self.n_gen)/2))

    def print_generation_stats(self, generation, fitness):
        print(f"Generation {generation} stats:")
        print(f"  Max fitness: {max(fitness)}")
        print(f"  Avg fitness: {sum(fitness) / len(fitness)}")
        print(f"  Min fitness: {min(fitness)}")
        print(f"  vs Random (last 10 games): {self.stats['vs_random'][-10:]}")

    def print_final_stats(self):
        print("\nFinal Statistics:")
        print(f"Max fitness achieved: {max(self.stats['max_fitness'])}")
        print(f"Average fitness in last generation: {self.stats['avg_fitness'][-1]}")
        print(f"Best performance vs Random: {max(self.stats['vs_random'], key=lambda x: x[0] - x[2])}")

    def update_plots(self):
        generations = list(range(len(self.stats['max_fitness'])))
        plot_data = [
            (self.line_max, self.stats['max_fitness']),
            (self.line_avg, self.stats['avg_fitness']),
            (self.line_min, self.stats['min_fitness']),
            (self.line_top5_random, self.stats['top5_vs_random']),
            (self.line_diversity, self.stats['diversity'])
        ]

        for line, data in plot_data:
            line.set_data(generations, data)
            line.axes.relim()
            line.axes.autoscale_view()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


if __name__ == '__main__':
    ga = GeneticAlgorithm()
    ga.evolve()
