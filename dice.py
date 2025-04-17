from time import perf_counter_ns
import csv
import streamlit as st
import pandas as pd
import numpy as np
import itertools as itt
from collections import Counter, defaultdict

from genetic import algo_genetique


score_table = {(1, 3): 1000, (1, 4): 2000, (1, 5): 4000, (1, 6): 8000,
                    (2, 3): 200, (2, 4): 400, (2, 5): 800, (2, 6): 1600,
                    (3, 3): 300, (3, 4): 600, (3, 5): 1200, (3, 6): 2400,
                    (4, 3): 400, (4, 4): 800, (4, 5): 1600, (4, 6): 3200,
                    (5, 3): 500, (5, 4): 1000, (5, 5): 2000, (5, 6): 4000,
                    (6, 3): 600, (6, 4): 1200, (6, 5): 2400, (6, 6): 4800}

memo_score_combi = {}
def calcul_score_combi(combi):
    sorted_combi = sorted(combi)
    if str(sorted_combi) in memo_score_combi:
        return memo_score_combi[str(sorted_combi)]
    score = 0
    if 0 in combi:
        idx = combi.index(0)
        alt_combi = list(combi)
        for i in range(1,7):
            alt_combi[idx] = i
            alt_combi = alt_combi
            alt_score = calcul_score_combi(tuple(alt_combi))
            if alt_score > score:
                score = alt_score
        memo_score_combi[str(sorted_combi)] = score
        return score    
    set_combi = set(combi)
    cnt_combi = Counter(combi)
    # détection suite complète
    if sorted_combi == [1,2,3,4,5,6]:
        score = 1500
        memo_score_combi[str(sorted_combi)] = score
        return score
    # détection suite 1-5
    elif set([1,2,3,4,5]).issubset(set_combi):
        score = 500
        if cnt_combi[5] > 1:
            score += 50
        if cnt_combi[1] > 1:
            score += 100
        memo_score_combi[str(sorted_combi)] = score
        return score
    # détection suite 2-6
    elif set([2,3,4,5,6]).issubset(set_combi):
        score = 750
        if cnt_combi[5] > 1:
            score += 50
        memo_score_combi[str(sorted_combi)] = score
        return score
    # détection n-kind
    for n, quantity in cnt_combi.items():
        if quantity > 2:
            score += score_table[n, quantity]
        else:
            if n == 1:
                score += 100*quantity
            if n == 5:
                score += 50*quantity
    memo_score_combi[str(sorted_combi)] = score
    return score

def get_scores_combis():
    score_of_combi = {}
    try:
        with open("score_combi.csv", "r") as f:
            for line in f:
                combi, score = line.split(',')
                score_of_combi[tuple(map(int, combi))] = int(score)
    except:
        all_combinations = itt.product(range(0, 7), repeat=6)
        for combi in all_combinations:
            score = calcul_score_combi(combi)
            if score > 0:
                score_of_combi[combi] = score
        with open("score_combi.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            for key, value in score_of_combi.items():
                key = [str(x) for x in key]
                writer.writerow([''.join(key), value])
            f.close()
    return score_of_combi

def get_probas_score(dice_indices, score_of_combi, proba_array):
    dice_probs = proba_array[dice_indices]
    score_probabilities = defaultdict(float)
    for combi, score in score_of_combi.items():
        proba = 1.0
        for i, face_value in enumerate(combi):
            proba *= dice_probs[i, face_value]
            if proba == 0:
                break
        score_probabilities[score] += proba
    return score_probabilities


def get_dices_sets(current_set, die_index, remaining, base_df):
    if remaining == 0:
        return [current_set.copy()]
    if die_index >= len(base_df):
        return []
    
    final_sets = []
    max_allowed = base_df.iloc[die_index]["Quantity"]
    nb_max = min(remaining, max_allowed)
    
    for nb in range(nb_max+1):
        for _ in range(nb):
            current_set.append(die_index)

        res = get_dices_sets(current_set, die_index+1, remaining-nb, base_df)
        final_sets.extend(res)

        for _ in range(nb):
            current_set.pop()

    return final_sets

def expectation_calculation(score_proba):
    return sum(score*proba for score, proba in score_proba.items())

def getNeighbors(dices_indices, base_df):
    neighbors = []
    available_dices = set(base_df.index).difference(set(dices_indices))
    for i in range(len(dices_indices)):
        for dice in available_dices:
            neigh = dices_indices.copy()
            neigh[i] = dice
            neighbors.append(neigh)
    return neighbors

#################################################################################

def main():
    st.title(f"Dice Optimizer for KCD2  /// {str(np.random.randint(0,10000))}")

    # Initialise la session une seule fois
    if "validated" not in st.session_state:
        st.session_state.validated = False

    base_df = pd.read_csv("dice.csv")

    if not st.session_state.validated:
        st.subheader("Indique combien de dés vous possédez pour chaque type :")
        updated_quantities = []

        # Conteneur pour les inputs
        with st.form("form_dice"):
            # Tous les dés sauf le dernier
            for i, row in base_df[:-1].iterrows():
                qty = st.number_input(
                    label=f"{row['Name']}",
                    min_value=0,
                    max_value=10,
                    value=int(row["Quantity"]),
                    key=f"qty_{i}"
                )
                updated_quantities.append(qty)

            # Dé normal bloqué à 6
            st.markdown("### Normal Die")
            qty = st.number_input(
                label="Normal Die",
                min_value=6,
                max_value=6,
                value=6,
                key=f"qty_{len(base_df)-1}",
                disabled=True
            )
            updated_quantities.append(qty)

            # Bouton de validation
            submitted = st.form_submit_button("Valider mes dés")
            if submitted:
                base_df["Quantity"] = updated_quantities
                st.session_state.validated = True
                st.session_state.base_df = base_df.copy()  # stocker si besoin plus tard
                st.rerun()

    else:
        st.info("Computation ongoing, information will appear")
        start = perf_counter_ns()
        print("\nCréation des sets de dés")
        all_dices_sets = get_dices_sets([], 0, 6, st.session_state.base_df)
        print(len(all_dices_sets))
        st.info(f"{len(all_dices_sets)} different dice sets generated")
        print(f"{((perf_counter_ns() - start)*1e-9):2f} s")

        start = perf_counter_ns()
        print("\nImportation des combinaisons")
        score_of_combi = get_scores_combis()
        print(len(score_of_combi))
        st.info(f"{len(score_of_combi)} scoring combinations considered")
        print(f"{((perf_counter_ns() - start)*1e-9):2f} s")

        idx_normal_dice = st.session_state.base_df.index[st.session_state.base_df["Name"] == "Normal die"].tolist()[0]
        proba_array = st.session_state.base_df.iloc[:, 2:].to_numpy()
        probas = get_probas_score([idx_normal_dice]*6, score_of_combi, proba_array)
        exp_score = expectation_calculation(probas)
        print(f"\r\n\nLe set de base a une espérance de {exp_score:.2f}")
        st.info(f"The base set has an expected score of {exp_score:.2f}")

        start = perf_counter_ns()
        print(f"\nAlgo génétique ")
        st.info("Genetic algorithm started (up to 5 minutes)")
        best_solution, best_score = algo_genetique(score_of_combi, st.session_state.base_df, max_time=300, max_stag=30, streamlit_log=st.empty().text)
        st.info(f"Execution time : {round((perf_counter_ns() - start)*1e-9, 2)} s")
        st.info(f"The best set has an expected score of {best_score:.2f}")
        st.info(f"Best set: {', '.join(st.session_state.base_df.iloc[best_solution, 0].values)}")
        print(f"{((perf_counter_ns() - start)*1e-9):2f} s\n")

if __name__ == "__main__":
    main()
