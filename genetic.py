from collections import defaultdict
from copy import deepcopy
from random import random, sample

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from numpy.ma.extras import average
from tqdm import tqdm

from main import RandomPlayer
from qwixx.game import Color, compete
from qwixx.player import Player


class MLPPlayer(Player):
    def __init__(self, saved_model=None, input_size=71, hidden_size=128, output_size=1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        if saved_model:
            self.model.load_state_dict(torch.load(saved_model, weights_only=True))

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
    def __init__(self, population_size=50, mutation_rate=0.1, crossover_rate=0.7, n_generations=100):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.n_generations = n_generations
        self.population = [MLPPlayer(saved_model='best.pt') for _ in range(population_size)]

        self.elite_size = int(0.1 * population_size) + 1
        self.stats = defaultdict(list)

        #       -------------------
        self.switch_to_top_players = True  # Track when to switch opponents
        #       -------------------

        # Initialize plot (same as before)
        plt.ion()
        self.fig, (self.ax1, self.ax2, self.ax3, self.ax4) = plt.subplots(4, 1, figsize=(12, 24))
        self.line_max, = self.ax1.plot([], [], label='Max Fitness')
        self.line_avg, = self.ax1.plot([], [], label='Avg Fitness')
        self.line_min, = self.ax1.plot([], [], label='Min Fitness')
        self.line_random, = self.ax2.plot([], [], label='Wins vs Random')
        self.line_top5_random, = self.ax3.plot([], [], label='Top 5 Avg Wins vs Random')
        self.line_diversity, = self.ax4.plot([], [], label='Population Diversity')

        self.ax1.set_xlabel('Generation')
        self.ax1.set_ylabel('Fitness')
        self.ax1.set_title('Fitness over Generations')
        self.ax1.legend()

        self.ax2.set_xlabel('Evaluation')
        self.ax2.set_ylabel('Wins')
        self.ax2.set_title('Wins against Random and Improved Players')
        self.ax2.legend()

        self.ax3.set_xlabel('Generation')
        self.ax3.set_ylabel('Average Wins')
        self.ax3.set_title('Top 5 Players Average Wins')
        self.ax3.legend()

        self.ax4.set_xlabel('Generation')
        self.ax4.set_ylabel('Diversity')
        self.ax4.set_title('Population Diversity')
        self.ax4.legend()

    def evolve(self):
        for generation in range(self.n_generations):
            print(f"Generation {generation + 1}")
            fitness = self.evaluate_population()

            self.stats['max_fitness'].append(max(fitness))
            self.stats['avg_fitness'].append(sum(fitness) / len(fitness))
            self.stats['min_fitness'].append(min(fitness))

            if average(fitness) > 5 and not self.switch_to_top_players:
                self.switch_to_top_players = True
                print(f"\nMinimum fitness exceeded 75 in generation {generation + 1}. "
                      f"Switching to top 2 players for competition.")

            # Sort population by fitness
            sorted_population = [x for _, x in
                                 sorted(zip(fitness, self.population), key=lambda pair: pair[0], reverse=True)]
            self.stats['top5_vs_random'].append(self.evaluate_top_n(sorted_population[:5], RandomPlayer()))
            self.stats['diversity'].append(self.calculate_diversity(sorted_population))

            if self.switch_to_top_players:
                print(f"Top 5 players are now competing against the top 2 from generation {generation + 1}.")

                # Save the best model
                self.save_best_model(sorted_population[0], generation)
                print(f"Best model from generation {generation + 1} saved.")

            new_population = sorted_population[:self.elite_size]
            while len(new_population) < self.population_size:
                parents = self.selection(sorted_population, fitness)
                offspring = self.crossover_and_mutate(parents)
                new_population.extend(offspring)

            self.population = new_population[:self.population_size]
            self.print_generation_stats(generation, fitness)
            self.update_plots()

        self.print_final_stats()

    def evaluate_population(self):
        fitness_scores = []
        for player in tqdm(self.population, "Evaluating population"):
            fitness_scores.append(self.evaluate_player(player))
        return fitness_scores

    def evaluate_player(self, player, nr_matches_total=100):
        if self.switch_to_top_players:
            # Compete against the top 2 players instead of Random/Improved
            opponents = self.population[:5]  # Top 5 players
        else:
            opponents = [RandomPlayer()]

        total_score = 0
        for opponent in opponents:
            wins, draws, losses = compete(player, opponent, n_games=int(nr_matches_total / len(opponents)))
            total_score += wins - losses

            # Track results against Random and Improved players
            if isinstance(opponent, RandomPlayer):
                self.stats['vs_random'].append((wins, draws, losses))
        return total_score

    def save_best_model(self, best_player, generation):
        torch.save(best_player.model.state_dict(), f'best_model_gen_{generation + 1}.pt')

    def evaluate_top_n(self, top_n_players, opponent, n_games=100):
        total_wins = 0
        for player in top_n_players:
            wins, _, _ = compete(player, opponent, n_games=n_games)
            total_wins += wins
        return total_wins / len(top_n_players)

    def calculate_diversity(self, population):
        # Calculate diversity as the average pairwise Euclidean distance between model parameters
        distances = []
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                dist = 0
                for p1, p2 in zip(population[i].model.parameters(), population[j].model.parameters()):
                    dist += torch.sum((p1 - p2) ** 2).item()
                distances.append(np.sqrt(dist))
        return np.mean(distances)

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

    def print_final_stats(self):
        print("\nFinal Statistics:")
        print(f"Max fitness achieved: {max(self.stats['max_fitness'])}")
        print(f"Average fitness in last generation: {self.stats['avg_fitness'][-1]}")
        print(f"Best performance vs Random: {max(self.stats['vs_random'], key=lambda x: x[0] - x[2])}")

    def update_plots(self):
        generations = list(range(len(self.stats['max_fitness'])))

        # Update the first axis (ax1) for fitness data
        self.line_max.set_data(generations, self.stats['max_fitness'])
        self.line_avg.set_data(generations, self.stats['avg_fitness'])
        self.line_min.set_data(generations, self.stats['min_fitness'])
        self.ax1.relim()
        self.ax1.autoscale_view()

        # Clear ax2 before plotting new data
        self.ax2.cla()  # Clear the second axis to remove previous plots

        # Update the second axis (ax2) for win data
        vs_random = list(zip(*self.stats['vs_random']))
        evaluations = list(range(len(vs_random[0])))

        self.line_random, = self.ax2.plot(evaluations, vs_random[0], label='Wins vs Random', color='purple')

        self.ax2.relim()
        self.ax2.autoscale_view()
        self.ax2.legend()  # Add legend back after clearing

        # Update the third axis (ax3) for top5 data
        self.line_top5_random.set_data(generations, self.stats['top5vs_random'])
        self.ax3.relim()
        self.ax3.autoscale_view()

        # Update the fourth axis (ax4) for diversity data
        self.line_diversity.set_data(generations, self.stats['diversity'])
        self.ax4.relim()
        self.ax4.autoscale_view()

        # Redraw the figure
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


if __name__ == '__main__':
    ga = GeneticAlgorithm()
    ga.evolve()
