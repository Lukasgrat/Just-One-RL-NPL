CS4100 Project - Just One AI Agent (Extensions)
=================================================

This document describes the extensions and improvements made to the base
Just One RL project. The original system (Q-Learning agent, environment,
clue selection, embeddings/clustering) was provided as a starting point.
The changes below are the additions and modifications made on top of it.


----------------------------------------------------------------
SECTION 1: BUG FIXES / GETTING THE PROJECT TO RUN
----------------------------------------------------------------

Files changed: environment.py, fix_pickles.py, scripts/Q_Learning.py,
               scripts/choose_clues.py, data/cluster.pkl, data/embeddings.pkl

  fix_pickles.py  [NEW]
    The provided pickle files (cluster.pkl, embeddings.pkl) were saved with a
    pandas version using StringDtype, causing a NotImplementedError on load.
    This one-time script patches the unpickler to convert StringDtype columns
    to plain object dtype and re-saves the files.

    Run once if you hit errors loading the data:
        python3 fix_pickles.py

  environment.py
    Fixed a crash in reset() where None values in the observation list caused
    action_space.remove() to raise a ValueError.

  scripts/Q_Learning.py
    Fixed hashObs() to normalize words with str() before hashing, preventing
    mismatches between np.str_ and str types that caused state lookup failures.
    Fixed hashAction() to use list.index() instead of a manual loop.

  scripts/choose_clues.py
    Fixed get_n_clues() to skip None clues instead of appending them, which
    previously caused downstream errors in state encoding and guessing.

  scripts/recluster.py  [NEW]
    The original HDBSCAN clustering left some words unassigned (noise cluster),
    making them unplayable. This script regenerates data/cluster.pkl using
    k-means so all 633 words are assigned to a cluster.

    Run from project root:
        python3 -m scripts.recluster
        python3 -m scripts.recluster --k 40    # custom k


----------------------------------------------------------------
SECTION 2: REWARD SYSTEM CHANGE
----------------------------------------------------------------

Files changed: environment.py

  The original step() function returned only +50 (correct) or -10 (wrong).
  Added a partial reward: if the agent guesses a word in the same semantic
  cluster as the answer, it receives +5 instead of -10. This gives the agent
  a gradient signal for near-correct guesses, improving training stability.

  Old rewards:  correct = +50,  wrong = -10
  New rewards:  correct = +50,  same cluster = +5,  wrong = -10


----------------------------------------------------------------
SECTION 3: DQN AGENT
----------------------------------------------------------------

Files changed: scripts/DQN_Learning.py [NEW], main.py, scripts/Q_Learning.py

  scripts/DQN_Learning.py
    A full Deep Q-Network agent added alongside the existing Q-Learning agent.

    Architecture:
      - DQN: 4-layer neural net (input -> 1024 -> 512 -> 256 -> NUM_WORDS)
      - ReplayBuffer: experience replay, capacity 10,000 transitions
      - encode_state(): state = sorted clue embeddings + mean vector
        (order-invariant, fixed size = 384 * (NUM_CLUES + 1))

    Training:
      - Epsilon-greedy exploration with configurable decay
      - Target network synced every 200 episodes
      - MSE loss, Adam optimizer
      - 100,000 episodes, decay_rate = 0.9999

    Evaluation:
      - Loads saved .pt model, runs 1000 episodes
      - Prints accuracy, saves dqn_rewards.png

  main.py
    Wired DQN into the main entry point alongside Q-Learning.
    CLI dispatch via sys.argv:

      Train Q-Learning:   python3 main.py train
      Evaluate Q-Learning: python3 main.py

      Train DQN:          python3 main.py dqn train
      Evaluate DQN:       python3 main.py dqn

  Saved model files:
      DQN_model_100000_0.9999.pt      (100k episode run)
      DQN_model_10000_0.999999.pt     (10k episode run)
      Q_table_100000_0.9999.pickle    (100k episode run)
      Q_table_10000_0.999999.pickle   (10k episode run)


----------------------------------------------------------------
SECTION 4: DROP_PCT EXPERIMENT
----------------------------------------------------------------

Files changed: scripts/experiment_drop_pct.py [NEW], scripts/choose_clues.py

  scripts/choose_clues.py
    Extended get_clue() and get_n_clues() to accept a drop_pct parameter
    (default 0.0), which controls how many top-ranked clue candidates are
    dropped before selecting a clue. This introduces noise/variability into
    the clue generation process.

  scripts/experiment_drop_pct.py
    Experiment script that measures how drop_pct affects DQN agent performance.
    For each value in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
      1. Trains a fresh DQN for 10,000 episodes using that drop_pct
      2. Evaluates over 500 episodes
      3. Records correct guesses

    Produces a plot of correct guesses vs. drop_pct saved to
    drop_pct_experiment.png.

    Run from project root:
        python3 -m scripts.experiment_drop_pct


----------------------------------------------------------------
HOW TO RUN (FULL SEQUENCE)
----------------------------------------------------------------

Step 1 - (Optional) Regenerate clusters:
    python3 -m scripts.recluster

Step 2 - Train Q-Learning agent:
    python3 main.py train

Step 3 - Evaluate Q-Learning agent:
    python3 main.py

Step 4 - Train DQN agent:
    python3 main.py dqn train

Step 5 - Evaluate DQN agent:
    python3 main.py dqn

Step 6 - Run drop_pct experiment:
    python3 -m scripts.experiment_drop_pct

NOTE: If you get a NotImplementedError when loading the pickle files, run:
    python3 fix_pickles.py
This patches cluster.pkl and embeddings.pkl for your pandas version.
The files pushed to the repo should work in most environments without this.
