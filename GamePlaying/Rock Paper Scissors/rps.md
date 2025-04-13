**Rock Paper Scissors**

**# main.py**

# This entrypoint file to be used in development. Start by reading README.md

from RPS_game import play, mrugesh, abbey, quincy, kris, human, random_player

from RPS import player

from unittest import main

play(player, quincy, 1000)

play(player, abbey, 1000)

play(player, kris, 1000)

play(player, mrugesh, 1000)

# Uncomment line below to play interactively against a bot:

play(human, abbey, 20, verbose=True)

# Uncomment line below to play against a bot that plays randomly:

play(human, random_player, 1000)

# Uncomment line below to run unit tests automatically

main(module='test_module', exit=False)

—————————————————————————————————————————————————————————————

**# RPS_game.py**

# DO NOT MODIFY THIS FILE

import random

def play(player1, player2, num_games, verbose=False):

**    **p1_prev_play = ""

**    **p2_prev_play = ""

**    **results = {"p1": 0, "p2": 0, "tie": 0}

**    **for _ in range(num_games):

**        **p1_play = player1(p2_prev_play)

**        **p2_play = player2(p1_prev_play)

**        **if p1_play == p2_play:

**            **results["tie"] += 1

**            **winner = "Tie."

**        **elif (p1_play == "P" and p2_play == "R") or (

**                **p1_play == "R" and p2_play == "S") or (p1_play == "S"

**                                                       **and p2_play == "P"):

**            **results["p1"] += 1

**            **winner = "Player 1 wins."

**        **elif p2_play == "P" and p1_play == "R" or p2_play == "R" and p1_play == "S" or p2_play == "S" and p1_play == "P":

**            **results["p2"] += 1

**            **winner = "Player 2 wins."

**        **if verbose:

**            **print("Player 1:", p1_play, "| Player 2:", p2_play)

**            **print(winner)

**            **print()

**        **p1_prev_play = p1_play

**        **p2_prev_play = p2_play

**    **games_won = results['p2'] + results['p1']

**    **if games_won == 0:

**        **win_rate = 0

**    **else:

**        **win_rate = results['p1'] / games_won * 100

**    **print("Final results:", results)

**    **print(f"Player 1 win rate: {win_rate}%")

**    **return (win_rate)

—————————————————————————————————————————————————————————————

**# Opponent Bots – High-Level Logic**

def quincy(prev_play, counter=[0]):

**    **counter[0] += 1

**    **choices = ["R", "R", "P", "P", "S"]

**    **return choices[counter[0] % len(choices)]

—————————————————————————————————————————————————————————————

def mrugesh(prev_opponent_play, opponent_history=[]):

**    **opponent_history.append(prev_opponent_play)

**    **last_ten = opponent_history[-10:]

**    **most_frequent = max(set(last_ten), key=last_ten.count)

**    **if most_frequent == '':

**        **most_frequent = "S"

**    **ideal_response = {'P': 'S', 'R': 'P', 'S': 'R'}

**    **return ideal_response[most_frequent]

—————————————————————————————————————————————————————————————

def kris(prev_opponent_play):

**    **if prev_opponent_play == ‘’”:

**        **prev_opponent_play = "R"

**    **ideal_response = {'P': 'S', 'R': 'P', 'S': 'R'}

**    **return ideal_response[prev_opponent_play]

—————————————————————————————————————————————————————————————

def abbey(prev_opponent_play,

**          **opponent_history=[],

**          **play_order=[{

**              **"RR": 0,

**              **"RP": 0,

**              **"RS": 0,

**              **"PR": 0,

**              **"PP": 0,

**              **"PS": 0,

**              **"SR": 0,

**              **"SP": 0,

**              **"SS": 0,

**          **}]):

**    **if not prev_opponent_play:

**        **prev_opponent_play = 'R'

**    **opponent_history.append(prev_opponent_play)

**    **last_two = "".join(opponent_history[-2:])

**    **if len(last_two) == 2:

**        **play_order[0][last_two] += 1

**    **potential_plays = [

**        **prev_opponent_play + "R",

**        **prev_opponent_play + "P",

**        **prev_opponent_play + "S",

**    **]

**    **sub_order = {

**        **k: play_order[0][k]

**        **for k in potential_plays if k in play_order[0]

**    **}

**    **prediction = max(sub_order, key=sub_order.get)[-1:]

**    **ideal_response = {'P': 'S', 'R': 'P', 'S': 'R'}

**    **return ideal_response[prediction]

—————————————————————————————————————————————————————————————

def human(prev_opponent_play):

**    **play = ""

**    **while play not in ['R', 'P', 'S']:

**        **play = input("[R]ock, [P]aper, [S]cissors? ")

**        **print(play)

**    **return play

—————————————————————————————————————————————————————————————

def random_player(prev_opponent_play):

**    **return random.choice(['R', 'P', 'S'])

—————————————————————————————————————————————————————————————

**# RPS.py**

import random

# Improved player function using history and counter-strategy

def player(prev_play, opponent_history=[]):

**    **# Append the current play to the opponent's history

**    **opponent_history.append(prev_play)

**    **# If it's the first play, pick randomly since we have no history

**    **if len(opponent_history) == 1:

**        **return random.choice(["R", "P", "S"])

**    **# Get the last move made by the opponent

**    **last_move = opponent_history[-1]

---

**    **# Predict based on last move - counter strategy:

**    **# Rock -> Paper (P beats R)

**    **# Paper -> Scissors (S beats P)

**    **# Scissors -> Rock (R beats S)

**    **if last_move == "R":

**        **return "P"**  **# Paper beats Rock

**    **elif last_move == "P":

**        **return "S"**  **# Scissors beats Paper

**    **else:

**        **return "R"**  **# Rock beats Scissors

**    **# If no obvious pattern is detected, pick a random move

**    **return random.choice(["R", "P", "S"])

—————————————————————————————————————————————————————————————

**# test_module.py**

import unittest

from RPS_game import play, mrugesh, abbey, quincy, kris

from RPS import player

class UnitTests(unittest.TestCase):

**    **print()

**    **def test_player_vs_quincy(self):

**        **print("Testing game against quincy...")

**        **actual = play(player, quincy, 1000) >= 60

**        **self.assertTrue(

**            **actual,

**            **'Expected player to defeat quincy at least 60% of the time.')

**    **def test_player_vs_abbey(self):

**        **print("Testing game against abbey...")

**        **actual = play(player, abbey, 1000) >= 60

**        **self.assertTrue(

**            **actual,

**            **'Expected player to defeat abbey at least 60% of the time.')

**    **def test_player_vs_kris(self):

**        **print("Testing game against kris...")

**        **actual = play(player, kris, 1000) >= 60

**        **self.assertTrue(

**            **actual, 'Expected player to defeat kris at least 60% of the time.')

**    **def test_player_vs_mrugesh(self):

**        **print("Testing game against mrugesh...")

**        **actual = play(player, mrugesh, 1000) >= 60

**        **self.assertTrue(

**            **actual,

**            **'Expected player to defeat mrugesh at least 60% of the time.')

if __name__ == "__main__":

**    **unittest.main()

—————————————————————————————————————————————————————————————

**verbose**

* “Verbose" is about controlling the level of detail in a program's output, allowing us to get as much or as little information as we need.
* In Python programming, "verbose" refers to a setting that controls the level of detail in a program's output. Essentially:
* **It means providing more detailed information.**
* **It's often used for debugging or monitoring progress.**

So, a "verbose" mode will display more output, giving you a clearer picture of what the program is doing.

* **verbose = True:** Gives you detailed output for debugging and monitoring.
* **verbose = False:** Gives minimal output, suitable for production.

Essentially, True means "show me everything," and False means "keep it brief.

🎮** 1. Game Engine Algorithm (play() function in RPS_game.py)**

**Algorithm: play(player1, player2, num_games, verbose=False)**

 **Input** : two functions player1 and player2, number of games num_games

 **Output** : Player 1's win rate

**Steps:**

1. **Initialize game history** :

* p1_prev_play = ""
* p2_prev_play = ""
* results = {"p1": 0, "p2": 0, "tie": 0}

1. **Repeat num_games times** : a. Call player1(p2_prev_play) → get p1_play

   b. Call player2(p1_prev_play) → get p2_play

1. **Decide round winner** :

* If p1_play == p2_play, increment results["tie"]
* Else if:
  * p1_play == "R" and p2_play == "S"
  * p1_play == "P" and p2_play == "R"
  * p1_play == "S" and p2_play == "P"

    → Player 1 wins → results["p1"] += 1
* Else → Player 2 wins → results["p2"] += 1

1. **If verbose=True** , print round results
2. **Update history** :

* p1_prev_play = p1_play
* p2_prev_play = p2_play

1. **After loop** , compute:

* win_rate = results['p1'] / (results['p1'] + results['p2']) * 100

1. **Print final result and return win_rate**

🤖** 2. Bot's Algorithm (player() in RPS.py)**

**Goal: Beat opponent by predicting their next move based on last move.**

**Algorithm: player(prev_play, opponent_history=[])**

1. Append prev_play to opponent_history
2. If len(opponent_history) == 1:
   * Return random choice from ["R", "P", "S"]
3. Set last_move = opponent_history[-1] (opponent’s last move)
4. Predict that opponent will repeat last_move
5. Return counter-move:
   * If last_move == "R" → return "P"
   * If last_move == "P" → return "S"
   * If last_move == "S" → return "R"
6. (Unreachable line) return random.choice(...) as fallback

🧠** 3. Opponent Bots – High-Level Logic**

| **Bot**           | **Strategy**                                                                                         |
| ----------------------- | ---------------------------------------------------------------------------------------------------------- |
| **Quincy**        | Cycles through list ["R", "R", "P", "P", "S"] repeatedly.                                                  |
| **Mrugesh**       | Tracks last 10 moves of the opponent → finds most frequent → plays counter-move.                         |
| **Kris**          | Always counters the opponent’s**last move** .                                                       |
| **Abbey**         | Tracks opponent’s last**two moves**and builds frequency map → predicts next move → plays counter. |
| **Random Player** | Chooses randomly from R, P, S.                                                                             |
| **Human**         | Asks user for input at runtime.                                                                            |

✅** 4. Test Algorithm (test_module.py)**

**Algorithm:**

1. Import play() function and all bots
2. Define a test class using unittest
3. For each test:
   * Run play(player, bot, 1000)
   * Check if win rate >= 60%
   * If yes → test passes
   * If no → test fails with message
4. unittest.main() runs all tests when script is run.

🧾** Combined Flow Summary**

📂 main.py

├──▶ play(player, quincy, 1000)

│**    **└──▶ RPS_game.play()

│ **        **├──▶ player(prev_play, opponent_history=[]) **        **[from RPS.py]

│ **        **└──▶ quincy(prev_play)

│

├──▶ play(player, abbey, 1000)

│**    **└──▶ RPS_game.play()

│ **        **├──▶ player(prev_play, opponent_history=[])

│ **        **└──▶ abbey(prev_play)

│ **              **└──▶ uses play_order + opponent_history

│

├──▶ play(player, kris, 1000)

│**    **└──▶ RPS_game.play()

│ **        **├──▶ player(prev_play, opponent_history=[])

│ **        **└──▶ kris(prev_play)

│

├──▶ play(player, mrugesh, 1000)

│**    **└──▶ RPS_game.play()

│ **        **├──▶ player(prev_play, opponent_history=[])

│ **        **└──▶ mrugesh(prev_play)

│ **              **└──▶ uses most frequent in history

│

├──▶ play(human, abbey, 20, verbose=True) **  **← [interactive play]

│**    **└──▶ human(prev_opponent_play)**    **← prompts for user input

│

├──▶ play(human, random_player, 1000)

│**    **└──▶ random_player(prev_opponent_play)**    **← selects randomly

│

└──▶ main(module='test_module')

**     **└──▶ UnitTests (All tests internally call ▶ play())

**          **├──▶ test_player_vs_quincy(self)

**          **├──▶ test_player_vs_abbey(self)

**          **├──▶ test_player_vs_kris(self)

**          **└──▶ test_player_vs_mrugesh(self)

**               **└──▶ All tests internally call ▶ play()

**Function Details**

* **play(player1, player2, num_games)** : Orchestrates the game loop.
* Calls the player functions (like quincy(), mrugesh(), etc.) by passing the opponent's previous move.
* Compares results and prints final scores.
* **Bot Strategies** :
* quincy(): Cycles through a predefined list.
* mrugesh(): Counters the most frequent move in the last 10.
* abbey(): Predicts based on the last 2 plays and transition likelihood.
* kris(): Basic counter to opponent’s last move.
* random_player(): Randomly selects 'R', 'P', or 'S'.
* **RPS.player()** : Our custom player that adapts to the opponent’s previous move.
* **test_module.py** :
* Contains unit tests to ensure our custom player wins at least 60% against each bot.
