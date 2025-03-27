# 🎰 Multi-Armed Bandit Simulation – Marketing Homework 2

This project explores two bandit algorithms — **Epsilon-Greedy** and **Thompson Sampling** — to evaluate their performance across different reward configurations.

---

## 📁 Project Structure

<pre>
├── Bandit.py                   # Main implementation
├── epsilon_greedy_results.csv  # Epsilon-Greedy results with trial & scenario
├── thompson_sampling_results.csv
├── bandit_log.log              # Full logs
├── requirements.txt            # Python dependencies
└── README.md                   # This file
</pre>

---

## 📌 What I Did

- Implemented a generic `Bandit` base class with:
  - `EpsilonGreedy` strategy
  - `ThompsonSampling` strategy
- Created 4 experimental setups:
  - `Original`: `[1, 2, 3, 4]`
  - `HardGap`: `[0.1, 0.2, 0.3, 0.9]`
  - `SmallGap`: `[0.4, 0.5, 0.6, 0.7]`
  - `AlmostEqualHigh`: `[0.96, 0.97, 0.98, 0.99]`
- Ran experiments and logged:
  - Cumulative reward
  - Cumulative regret
- Stored results in CSVs
- Visualized learning and performance with Matplotlib

---

## 📊 Results Summary

| Scenario           | Algorithm         | Avg Reward | Total Regret |
|--------------------|-------------------|-------------|---------------|
| Original           | Epsilon-Greedy    | 0.9996      | 30.0000       |
|                    | Thompson Sampling | 0.9999      | 14.0000       |
| HardGap            | Epsilon-Greedy    | 0.9997      | 8.1000        |
|                    | Thompson Sampling | 0.9998      | 3.7000        |
| SmallGap           | Epsilon-Greedy    | 0.9998      | 3.4000        |
|                    | Thompson Sampling | 0.9999      | 0.9000        |
| AlmostEqualHigh 🏆 | Epsilon-Greedy    | 0.9999      | 3.0000        |
|                    | Thompson Sampling | 0.9998      | 2.1400        |

> ✅ **Best Configuration**: `Epsilon-Greedy` on `AlmostEqualHigh` with regret of just **3.0**

---

## 📦 Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/GhukasyanNarek/Marketing-homework2.git
cd Marketing-homework2
pip install -r requirements.txt
python Bandit.py
```

## 📁 Output Files

- `epsilon_greedy_results.csv` — Rewards from all Epsilon-Greedy trials  
- `thompson_sampling_results.csv` — Rewards from all Thompson Sampling trials  
- `bandit_log.log` — Full logs and experiment tracking
- `plot_rewards\plot_regrets.png` - The plot for both algorithms

---

## 🧠 Suggested Improvements (Bonus Plan)

While both strategies performed well, we can explore more optimal policies:

To evaluate robustness, I added **three more test configurations**:

- `HardGap`: `[0.1, 0.2, 0.3, 0.9]` — big reward gap between best arm and others  
- `SmallGap`: `[0.4, 0.5, 0.6, 0.7]` — rewards are close together  
- `AlmostEqualHigh`: `[0.96, 0.97, 0.98, 0.99]` — all arms have high and very close values  

These configurations help test how well each algorithm handles different exploration vs. exploitation pressures.

---

## 🌊 Add UCB1 (Upper Confidence Bound)

UCB1 balances exploration and exploitation using confidence bounds. Unlike Epsilon-Greedy (which requires manual epsilon decay) or Thompson Sampling (which relies on prior-based randomness), **UCB1 is deterministic** and adjusts naturally based on how often each arm has been pulled.

### ✅ Why is UCB1 an improvement in this case?

- In our experiments, we tested four distinct reward settings — including **HardGap**, **SmallGap**, and **AlmostEqualHigh** — all of which challenge how quickly and confidently an algorithm can identify the optimal arm.
- **Epsilon-Greedy**, while simple, requires careful tuning of the decay schedule (`epsilon / (t + 1)`) which may not generalize across all scenarios. It also spends too much time exploring, especially early on.
- **Thompson Sampling** performs well, but relies on **random sampling**, which may introduce variance across runs and is sensitive to prior assumptions.
- **UCB1 offers the best of both worlds**:
  - It **explores arms less often as their uncertainty drops**, making it ideal for `HardGap` and `AlmostEqualHigh` scenarios.
  - It performs **consistently and deterministically**, so its results are reproducible.
  - The **logarithmic exploration term** adapts faster than naive epsilon decay, especially in tighter reward settings like `SmallGap`.

