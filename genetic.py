from time import perf_counter_ns
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import random
from collections import defaultdict, Counter

# HYPERPARAMETERS
GENERATION_SIZE = 25
SELECTION_SIZE = 12  # must be even
PROBA_CROISEMENT = 0.8
PROBA_MUTATION = 1/6


def get_possible_dices(base_df, sub=[]):
    possible_dices = []
    for i in range(len(base_df)):
        for _ in range(base_df.iloc[i]["Quantity"]):
            possible_dices.append(i)
    try:
        for n in sub:
            possible_dices.remove(n)
    except ValueError:
        pass
    return possible_dices

def initialisation_population(base_df):
    random.seed()
    population = []
    possible_dices = get_possible_dices(base_df)
    for _ in range(GENERATION_SIZE-1): 
        population.append(np.random.choice(possible_dices, 6, replace=False))
    idx_normal_dice = base_df.index[base_df["Name"] == "Normal die"].tolist()[0]
    population.insert(random.randint(0, len(population)-1), np.array([idx_normal_dice]*6))
    return population

memo_expect_sol = {}
def evaluation_solution(dice_indices, score_of_combi, proba_array):
    if tuple(dice_indices) in memo_expect_sol:
        return memo_expect_sol[tuple(dice_indices)]
    dice_probs = proba_array[dice_indices]
    score_probabilities = defaultdict(float)
    for combi, score in score_of_combi.items():
        proba = 1.0
        for i, face_value in enumerate(combi):
            proba *= dice_probs[i, face_value]
            if proba == 0:
                break
        score_probabilities[score] += proba
    expected_score = sum(score * proba for score, proba in score_probabilities.items())
    memo_expect_sol[tuple(dice_indices)] = expected_score
    return expected_score

def selection_reproduction(population, score_of_combi, base_df):
    score_population = np.array([evaluation_solution(sol, score_of_combi, base_df) for sol in population])
    proba_population = score_population / score_population.sum()
    indices = np.random.choice(len(population), size=SELECTION_SIZE, replace=True, p=proba_population)
    return [population[i] for i in indices]

def croisement(selection, base_df):
    random.seed()
    final = []
    while len(selection) > 0:
        parent1 = selection.pop(random.randint(0, len(selection)-1))
        parent2 = selection.pop(random.randint(0, len(selection)-1))
        enfant1 = []
        enfant2 = []
        if random.random() > PROBA_CROISEMENT:
            enfant1 = parent1
            enfant2 = parent2
        else:
            point_croisement = random.randint(1, len(parent1)-1)
            enfant1 = np.concatenate((parent1[0:point_croisement], parent2[point_croisement:]))
            enfant2 = np.concatenate((parent2[0:point_croisement], parent1[point_croisement:]))
        final.append(mutation(enfant1, base_df))
        final.append(mutation(enfant2, base_df))
    return final

def mutation(solution, base_df):
    random.seed()
    mutated_sol = []
    possible_dices = get_possible_dices(base_df, solution)
    for n in solution:
        if random.random() < PROBA_MUTATION:
            mutated_dice = random.choice(possible_dices)
            mutated_sol.append(mutated_dice)
            possible_dices.remove(mutated_dice)
            possible_dices.append(n)
        else:
            mutated_sol.append(n)
    return np.array(mutated_sol)

def is_valid(solution, base_df):
    cnt = Counter(solution)
    quantity_array = base_df["Quantity"].to_numpy()
    for n, quantity in cnt.items():
        if quantity > quantity_array[n]:
            return False, n
    return True, -1

def reparation(population, base_df):
    repaired_pop = []
    for solution in population:
        valid, number = is_valid(solution, base_df)
        while not valid:
            possible_dices = get_possible_dices(base_df, solution)
            idx_pb = np.argmax(solution == number)
            solution = np.delete(solution, idx_pb)
            solution = np.insert(solution, idx_pb, random.choice(possible_dices))
            valid, number = is_valid(solution, base_df)
        repaired_pop.append(solution)
    return repaired_pop

def selection_survie(population, score_of_combi, proba_array):
    score_population = [(sol, evaluation_solution(sol, score_of_combi, proba_array)) for sol in population]
    return sorted(score_population, key=lambda x: x[1], reverse=True)[:GENERATION_SIZE]

def log(msg, streamlit_log=None):
    print(msg, end="\r")
    if streamlit_log:
        streamlit_log(msg)

def algo_genetique(score_of_combi, base_df, max_time=61, max_stag=11, streamlit_log=None):
    max_time *= 1e9
    best_tracking = []
    stag_tracking = []
    start = perf_counter_ns()
    proba_array = base_df.iloc[:, 2:].to_numpy()
    population = initialisation_population(base_df)
    best_solution, best_score = selection_survie(population, score_of_combi, proba_array)[0]
    time_cnt = 1
    stag_cnt = 0
    while stag_cnt <= max_stag and perf_counter_ns() - start <= max_time:
        log(f"\rGeneration #{time_cnt} (stagnation={stag_cnt})", streamlit_log)
        best_tracking.append([best_score])
        selection = selection_reproduction(population, score_of_combi, proba_array)
        new_gen = croisement(selection, base_df)
        new_gen = reparation(new_gen, base_df)
        population += new_gen
        population = selection_survie(population, score_of_combi, proba_array)
        gen_best_sol, gen_best_score = population[0]
        population = [x[0] for x in population]
        stag_cnt += 1
        if gen_best_score > best_score:
            best_solution = gen_best_sol
            best_score = gen_best_score
            stag_cnt = 0
        stag_tracking.append(stag_cnt)
        time_cnt += 1

    best_solution = sorted(best_solution)
    print(f"\r\n\nBest set has an expected score of {best_score:.2f}")
    print(f"Best solution (indices): {best_solution}")
    print(f"Best set: {', '.join(base_df.iloc[best_solution, 0].values)}\n")

    # Debug tracking (optional)
    # print(best_tracking)
    # plt.plot(range(1, time_cnt), best_tracking)
    # plt.show()

    # Debug stagnation (optional)
    # print(stag_tracking)
    # plt.plot(stag_tracking)
    # plt.show()

    return best_solution, best_score
