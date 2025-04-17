# KCD2 Dice Optimizer

This project provides a tool to evaluate and optimize dice combinations in the mini-game from *Kingdom Come: Deliverance 2*.

Given the dice available to a player, the program computes the most effective set of 6 dice to maximize the expected score of a single roll, based on official game rules.

## Goal

The objective is to identify the best possible 6-dice set from a custom pool, maximizing the **expected value** of a full roll. Scoring follows the official KCD2 rules: full straight (1–6), partial straight (1–5 or 2–6), triplets, quadruplets, quintuplets, and bonus points for rolling 1s and 5s.

## Method

- Each die is defined by a name, its available quantity, and a probability distribution over faces 1 to 6 (optionally including 0 for joker).
- All 6-dice face combinations are evaluated using a scoring function that matches the in-game rules.
- If a combination includes one or more joker faces (`0`), all possible substitutions (from 1 to 6) are tested, and the **highest resulting score** is used.
- Expected value is computed as a weighted average of scores for all combinations, using the face probabilities of the selected dice.
- Because the number of dice sets grows exponentially, a **genetic algorithm** is used to approximate the best combination:
  - Random initialization of valid dice sets
  - Fitness = expected value of a complete roll
  - Selection using roulette-wheel sampling
  - Crossover and mutation
  - Repair of invalid individuals (respecting quantity constraints)
  - Termination on stagnation or timeout

## Limitations

- The program simulates **a single roll** of 6 dice; it does **not** support re-rolls or sequential gameplay logic.
- Joker faces are handled by brute-force substitution of all possible values (1–6), which can increase computational cost.
- The genetic algorithm finds a **near-optimal** solution but does not guarantee the global optimum.
- Rare or skewed dice distributions can slightly affect convergence speed.
- Combination scores are precomputed and stored in `score_combi.csv` to speed up evaluations.

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/kcd2-dice-optimizer.git
cd kcd2-dice-optimizer
pip install -r requirements.txt
streamlit run dice.py
```
This will open the web interface in your browser. You can enter how many dice you own for each type and let the program compute the best 6-dice configuration.

## Parameters
In genetic.py
```bash
GENERATION_SIZE = 25       # Population size
SELECTION_SIZE = 12        # Number of individuals selected per generation
PROBA_CROISEMENT = 0.8     # Crossover probability
PROBA_MUTATION = 1/6       # Mutation probability (per die)
```
In the call to the genetic algorithm (inside dice.py)
```bash
MAX_TIME = 300             # Max runtime in seconds
MAX_STAG = 30              # Max generations without improvement
```

## Input format
The dice.csv file must contain:

Name: the name of the die

Quantity: how many you own (editable via the interface)

Probabilities for faces 1 to 6 (and optionally 0 for joker face)

The last die must be the normal die, which is always available in quantity 6 and cannot be modified.
