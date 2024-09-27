from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from random import random, sample, choice
import numpy as np
import matplotlib.pyplot as plt

from main import RandomPlayer, ImprovedPlayer
from qwixx.game import Color, Action, Scoreboard, compete
from qwixx.player import Player


class MLPPlayer(Player):
    def __init__(self, input_size=71, hidden_size=128, output_size=1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def move(self, actions, is_main, scoreboard, scoreboards):
        if not actions:
            return []

        best_action = None
        best_value = float('-inf')

        for action in actions:
            input_data = self.extract_features(scoreboard, scoreboards, action, is_main)
            input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                value = self.model(input_tensor).item()

            if value > best_value:
                best_value = value
                best_action = action

        return best_action

    def extract_features(self, scoreboard, scoreboards, action, is_main):
        features = []

        # Scoreboard features
        for color in Color:
            features.extend(self.encode_color(scoreboard[color]))
            features.append(scoreboard.crossed[color])

        # Action features
        action_encoding = [0] * 4  # One-hot encoding for action color
        if action:
            action_encoding[list(Color).index(action[0].color)] = 1
            features.extend(action_encoding)
            features.append(action[0].n / 13)  # Normalize the number
        else:
            features.extend(action_encoding)
            features.append(0)  # No action

        # Other players' scoreboard features
        for other_scoreboard in scoreboards:
            for color in Color:
                features.append(len(other_scoreboard[color]) / 13)  # Normalize the length
                features.append(other_scoreboard.crossed[color])

        # Game state features
        features.append(is_main)
        features.append(scoreboard.crosses / 4)  # Normalize crosses

        return features

    def encode_color(self, color_scores):
        encoding = [0] * 13  # 13 possible positions for each color
        for score in color_scores:
            encoding[score - 2] = 1  # Scores start from 2
        return encoding


class GeneticAlgorithm:
    def __init__(self, population_size=50, mutation_rate=0.05, crossover_rate=0.7, n_generations=10):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.n_generations = n_generations
        self.population = [MLPPlayer() for _ in range(population_size)]
        self.elite_size = int(0.1 * population_size)  # Keep top 10% as elite
        self.stats = defaultdict(list)

    def evolve(self):
        for generation in range(self.n_generations):
            print(f"Generation {generation + 1}")
            fitness = self.evaluate_population()

            self.stats['max_fitness'].append(max(fitness))
            self.stats['avg_fitness'].append(sum(fitness) / len(fitness))
            self.stats['min_fitness'].append(min(fitness))

            sorted_population = [x for _, x in
                                 sorted(zip(fitness, self.population), key=lambda pair: pair[0], reverse=True)]

            new_population = sorted_population[:self.elite_size]

            while len(new_population) < self.population_size:
                parents = self.selection(sorted_population, fitness)
                offspring = self.crossover_and_mutate(parents)
                new_population.extend(offspring)

            self.population = new_population[:self.population_size]

            if generation % 10 == 0:
                self.print_generation_stats(generation, fitness)

        self.print_final_stats()
        self.plot_stats()

    def evaluate_population(self):
        fitness_scores = []
        for player in self.population:
            score = self.evaluate_player(player)
            fitness_scores.append(score)
        return fitness_scores

    def evaluate_player(self, player):
        opponents = [RandomPlayer(), ImprovedPlayer()]  # Play against both Random and Improved players
        total_score = 0
        for opponent in opponents:
            wins, draws, losses = compete(player, opponent, n_games=50)
            total_score += wins - losses
            if isinstance(opponent, RandomPlayer):
                self.stats['vs_random'].append((wins, draws, losses))
            else:
                self.stats['vs_improved'].append((wins, draws, losses))
        return total_score

    def selection(self, sorted_population, fitness):
        tournament_size = 5
        selected = []
        for _ in range(2):
            tournament = sample(list(enumerate(sorted_population)), tournament_size)
            winner = max(tournament, key=lambda x: fitness[x[0]])
            selected.append(winner[1])
        return selected

    def crossover_and_mutate(self, parents):
        child1, child2 = self.crossover(parents[0], parents[1])
        self.mutate(child1)
        self.mutate(child2)
        return [child1, child2]

    def crossover(self, parent1, parent2):
        child1, child2 = deepcopy(parent1), deepcopy(parent2)
        if random() < self.crossover_rate:
            for p1, p2 in zip(child1.model.parameters(), child2.model.parameters()):
                mask = torch.rand(p1.shape) < 0.5
                temp = p1.data.clone()
                p1.data[mask] = p2.data[mask]
                p2.data[mask] = temp[mask]
        return child1, child2

    def mutate(self, child):
        for param in child.model.parameters():
            if random() < self.mutation_rate:
                param.data += torch.randn(param.shape) * 0.01  # Add Gaussian noise

    def print_generation_stats(self, generation, fitness):
        print(f"Generation {generation} stats:")
        print(f"  Max fitness: {max(fitness)}")
        print(f"  Avg fitness: {sum(fitness) / len(fitness)}")
        print(f"  Min fitness: {min(fitness)}")
        print(f"  vs Random (last 10 games): {self.stats['vs_random'][-10:]}")
        print(f"  vs Improved (last 10 games): {self.stats['vs_improved'][-10:]}")

    def print_final_stats(self):
        print("\nFinal Statistics:")
        print(f"Max fitness achieved: {max(self.stats['max_fitness'])}")
        print(f"Average fitness in last generation: {self.stats['avg_fitness'][-1]}")
        print(f"Best performance vs Random: {max(self.stats['vs_random'], key=lambda x: x[0] - x[2])}")
        print(f"Best performance vs Improved: {max(self.stats['vs_improved'], key=lambda x: x[0] - x[2])}")

    def plot_stats(self):
        plt.figure(figsize=(12, 8))
        plt.plot(self.stats['max_fitness'], label='Max Fitness')
        plt.plot(self.stats['avg_fitness'], label='Avg Fitness')
        plt.plot(self.stats['min_fitness'], label='Min Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Fitness over Generations')
        plt.legend()
        plt.savefig('fitness_plot.png')
        plt.close()

        plt.figure(figsize=(12, 8))
        vs_random = list(zip(*self.stats['vs_random']))
        vs_improved = list(zip(*self.stats['vs_improved']))
        plt.plot(vs_random[0], label='Wins vs Random')
        plt.plot(vs_improved[0], label='Wins vs Improved')
        plt.xlabel('Evaluation')
        plt.ylabel('Wins')
        plt.title('Wins against Random and Improved Players')
        plt.legend()
        plt.savefig('wins_plot.png')
        plt.close()


if __name__ == '__main__':
    ga = GeneticAlgorithm()
    ga.evolve()
