from abc import ABC, abstractmethod
from loguru import logger
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import os

# =========================== LOGGER ============================#
logger.add("bandit_log.log", rotation="1 MB")  # Save logs to file


# =========================== ABSTRACT BASE CLASS ============================#
class Bandit(ABC):
    """
    Abstract base class defining the required interface for multi-armed bandit strategies.
    """

    @abstractmethod
    def __init__(self, p):
        """
        Initialize the bandit with true arm probabilities.

        @param p : list of true mean reward probabilities for each arm
        """
        pass

    @abstractmethod
    def __repr__(self):
        """
        Return string representation of the bandit class.
        """
        pass

    @abstractmethod
    def pull(self, arm):
        """
        Simulate pulling the given arm and returning a reward.

        @param arm : int, index of the arm to pull
        @return : reward (int), typically 0 or 1
        """
        pass

    @abstractmethod
    def update(self, arm, reward):
        """
        Update internal estimates with the observed reward.

        @param arm : int, the arm that was pulled
        @param reward : int, the reward received
        """
        pass

    @abstractmethod
    def experiment(self):
        """
        Run the bandit experiment for the total number of trials.
        """
        pass

    @abstractmethod
    def report(self, scenario_name):
        """
        Report the average reward and cumulative regret, and store results in CSV.

        @param scenario_name : str, label of the current test case
        """
        pass


# =========================== VISUALIZATION CLASS ============================#
class Visualization:
    """
    Visualization class for plotting cumulative rewards and regrets.
    """

    def plot1(self, eg_rewards, ts_rewards):
        """
        Plot cumulative reward comparison between algorithms.

        @param eg_rewards : list of rewards from Epsilon-Greedy
        @param ts_rewards : list of rewards from Thompson Sampling
        """
        eg_cum = np.cumsum(eg_rewards)
        ts_cum = np.cumsum(ts_rewards)

        plt.figure(figsize=(12, 6))
        plt.plot(eg_cum, label="Epsilon-Greedy", linestyle='--', color='blue', alpha=0.9, linewidth=2, marker='o',
                 markevery=2000)
        plt.plot(ts_cum, label="Thompson Sampling", linestyle='-', color='orange', alpha=0.9, linewidth=2, marker='x',
                 markevery=2000)
        plt.xlabel("Trials")
        plt.ylabel("Cumulative Reward")
        plt.title("Cumulative Reward Comparison")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot2(self, eg_regrets, ts_regrets):
        """
        Plot cumulative regret comparison between algorithms.

        @param eg_regrets : list of regrets from Epsilon-Greedy
        @param ts_regrets : list of regrets from Thompson Sampling
        """
        plt.figure(figsize=(12, 6))
        plt.plot(np.cumsum(eg_regrets), label="Epsilon-Greedy", linestyle='--')
        plt.plot(np.cumsum(ts_regrets), label="Thompson Sampling", linestyle='-')
        plt.xlabel("Trials")
        plt.ylabel("Cumulative Regret")
        plt.title("Cumulative Regret Comparison")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# =========================== EPSILON-GREEDY STRATEGY ============================#
class EpsilonGreedy(Bandit):
    """
    Epsilon-Greedy implementation for solving the multi-armed bandit problem.
    """

    def __init__(self, p, trials=20000):
        """
        Initialize the epsilon-greedy strategy.

        @param p : list of true reward probabilities
        @param trials : int, total number of trials to run
        """
        super().__init__(p)
        self.p = p
        self.n = len(p)
        self.trials = trials
        self.counts = [0] * self.n
        self.values = [0.0] * self.n
        self.epsilon = 1.0
        self.rewards = []
        self.regrets = []
        self.best_mean = max(p)
        self.history = []

    def __repr__(self):
        return f"EpsilonGreedy({self.p})"

    def pull(self, arm):
        return np.random.binomial(1, self.p[arm] / max(self.p))

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]

    def experiment(self):
        for arm in range(self.n):
            reward = self.pull(arm)
            self.update(arm, reward)
            self.rewards.append(reward)
            self.regrets.append(self.best_mean - self.p[arm])
            self.history.append(("EpsilonGreedy", arm, reward))

        for t in range(self.n, self.trials):
            eps = self.epsilon / (t + 1)
            if random.random() < eps:
                arm = np.random.randint(self.n)
            else:
                arm = int(np.argmax(self.values))
            reward = self.pull(arm)
            self.update(arm, reward)
            self.rewards.append(reward)
            self.regrets.append(self.best_mean - self.p[arm])
            self.history.append(("EpsilonGreedy", arm, reward))

    def report(self, scenario_name):
        avg_reward = np.mean(self.rewards)
        total_regret = np.sum(self.regrets)
        logger.critical(f"[EpsilonGreedy - {scenario_name}] Avg Reward: {avg_reward:.4f}")
        logger.info(f"[EpsilonGreedy - {scenario_name}] Total Regret: {total_regret:.4f}")

        df = pd.DataFrame(self.history, columns=["Algorithm", "Bandit", "Reward"])
        df["Scenario"] = scenario_name
        df["Trial"] = range(1, len(df) + 1)

        df.to_csv("epsilon_greedy_results.csv", mode='a', index=False,
                  header=not os.path.exists("epsilon_greedy_results.csv"))


# =========================== THOMPSON SAMPLING STRATEGY ============================#
class ThompsonSampling(Bandit):
    """
    Thompson Sampling implementation for solving the multi-armed bandit problem.
    """

    def __init__(self, p, trials=20000):
        super().__init__(p)
        self.p = p
        self.n = len(p)
        self.trials = trials
        self.alpha = [1] * self.n
        self.beta = [1] * self.n
        self.rewards = []
        self.regrets = []
        self.best_mean = max(p)
        self.history = []

    def __repr__(self):
        return f"ThompsonSampling({self.p})"

    def pull(self, arm):
        return np.random.binomial(1, self.p[arm] / max(self.p))

    def update(self, arm, reward):
        if reward == 1:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1

    def experiment(self):
        for _ in range(self.trials):
            theta_samples = [np.random.beta(self.alpha[i], self.beta[i]) for i in range(self.n)]
            arm = int(np.argmax(theta_samples))
            reward = self.pull(arm)
            self.update(arm, reward)
            self.rewards.append(reward)
            self.regrets.append(self.best_mean - self.p[arm])
            self.history.append(("ThompsonSampling", arm, reward))

    def report(self, scenario_name):
        avg_reward = np.mean(self.rewards)
        total_regret = np.sum(self.regrets)
        logger.critical(f"[ThompsonSampling - {scenario_name}] Avg Reward: {avg_reward:.4f}")
        logger.info(f"[ThompsonSampling - {scenario_name}] Total Regret: {total_regret:.4f}")

        df = pd.DataFrame(self.history, columns=["Algorithm", "Bandit", "Reward"])
        df["Scenario"] = scenario_name
        df["Trial"] = range(1, len(df) + 1)

        df.to_csv("thompson_sampling_results.csv", mode='a', index=False,
                  header=not os.path.exists("thompson_sampling_results.csv"))


# =========================== RUN ONE EXPERIMENT ============================#
def run_experiment(name, rewards, trials=20000):
    """
    Run a single bandit experiment using both algorithms.

    @param name : str, name of the scenario
    @param rewards : list, true reward probabilities for arms
    @param trials : int, number of iterations to run
    """
    logger.info(f"=== Running experiment: {name} ===")

    eg = EpsilonGreedy(rewards, trials)
    ts = ThompsonSampling(rewards, trials)

    logger.info(f"[{name}] Epsilon-Greedy starting...")
    eg.experiment()
    eg.report(name)

    logger.info(f"[{name}] Thompson Sampling starting...")
    ts.experiment()
    ts.report(name)

    viz = Visualization()
    logger.info(f"[{name}] Plotting results...")
    viz.plot1(eg.rewards, ts.rewards)
    viz.plot2(eg.regrets, ts.regrets)


# =========================== ALL EXPERIMENTS ============================#
def comparison():
    """
    Run experiments for all predefined scenarios.
    """
    test_cases = {
        "Original": [1, 2, 3, 4],
        "HardGap": [0.1, 0.2, 0.3, 0.9],
        "SmallGap": [0.4, 0.5, 0.6, 0.7],
        "AlmostEqualHigh": [0.96, 0.97, 0.98, 0.99]
    }

    for name, rewards in test_cases.items():
        run_experiment(name, rewards)


# =========================== BEST CONFIG SUMMARY ============================#
def summarize_best_config():
    """
    Analyze all results and determine which scenario + algorithm had lowest regret.
    """
    logger.info("Analyzing CSVs to find best-performing algorithm+scenario...")

    def evaluate(file):
        df = pd.read_csv(file)
        summary = []
        for scenario in df["Scenario"].unique():
            df_s = df[df["Scenario"] == scenario]
            avg_reward = df_s["Reward"].mean()
            regret = len(df_s) * df_s["Reward"].max() - df_s["Reward"].sum()
            summary.append((file, scenario, avg_reward, regret))
        return summary

    all_results = evaluate("epsilon_greedy_results.csv") + evaluate("thompson_sampling_results.csv")
    best = sorted(all_results, key=lambda x: x[3])[0]

    logger.success(f"Best Config: {best[1]} in {best[0]}")
    logger.info(f"Average Reward: {best[2]:.4f}, Total Regret: {best[3]:.4f}")


# =========================== MAIN ============================#
if __name__ == '__main__':
    logger.debug("Debug log for testing.")
    logger.info("Info log for testing.")
    logger.warning("Warning log for testing.")
    logger.error("Error log for testing.")
    logger.critical("Critical log for testing.")

    comparison()
    summarize_best_config()