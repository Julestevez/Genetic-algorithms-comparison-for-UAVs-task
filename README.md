# Single-Objective Bio-Inspired Optimization for UAV Swarm Coordination
Main author: Mohammed Safar (University of the Basque Country, PhD student)

This code presents a rigorous comparative study of three prominent metaheuristics: Particle Swarm Optimization (PSO), Grey Wolf Optimizer (GWO), and Ant Colony Optimization (ACO), evaluated across six diverse mission profiles including dynamic obstacle avoidance, formation flight, and multi-target engagement

Project Structure
```text

Paper\_A\_Single\_Objective/
├── requirements.txt                   # Python dependencies
├── setup.py                           # Package setup
├── benchmark\_single\_objective.py      # 
├── benchmark\_single\_objective\_independent.py
├── create\_scenario\_gifs.py            # GIF generation
├── configs/
│   └── default\_config.yaml            # Configuration (tuned parameters)
├── src/
│   ├── algorithms/
│   │   ├── pso.py                     # Particle Swarm Optimization
│   │   ├── gwo.py                     # Grey Wolf Optimizer
│   │   ├── aco.py                     # Ant Colony Optimization
│   │   └── base\_optimizer.py          # Base class
│   ├── scenarios/
│   │   ├── obstacle\_avoidance.py      # Scenario 1
│   │   ├── dynamic\_obstacle\_avoidance.py  # Scenario 2
│   │   ├── formation\_flight.py        # Scenario 3
│   │   ├── area\_coverage.py           # Scenario 4
│   │   ├── target\_tracking.py         # Scenario 5
│   │   └── multi\_target\_engagement.py # Scenario 6
│   ├── environment/
│   │   ├── environment.py             # 3D environment
│   │   └── obstacles.py               # Obstacle manager
│   ├── simulation/
│   │   ├── uav.py                     # UAV model
│   │   ├── swarm.py                   # Swarm controller
│   │   └── physics.py                 # Physics engine
│   ├── evaluation/
│   │   ├── metrics.py                 # Performance metrics
│   │   ├── benchmarking.py            # Benchmark runner
│   │   └── statistical\_analysis.py    # Statistical tests
│   ├── visualization/
│   │   ├── plotter.py                 # 2D/3D plotting
│   │   └── animator.py                # Animation/GIF generation
└── results\_paper\_a/                   # Generated results (git-ignored)
    └── YYYYMMDD\_HHMMSS/
        ├── benchmark\_results.json     # Raw results
        ├── summary\_statistics.csv     # Aggregated stats
        ├── algorithm\_comparison.png   # Comparison plots
        ├── gifs/                      # Generated GIFs (6+)
        │   ├── obstacle\_avoidance\_all\_algorithms.gif
        │   ├── dynamic\_obstacle\_avoidance\_all\_algorithms.gif
        │   ├── formation\_flight\_all\_algorithms.gif
        │   ├── area\_coverage\_all\_algorithms.gif
        │   ├── target\_tracking\_all\_algorithms.gif
        │   └── multi\_target\_engagement\_all\_algorithms.gif
        └── per\_scenario/   


## 1. Activate the Environment


python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt


## 2. Run the Benchmark

Run:

```bash
python benchmark\_single\_objective\_independent.py \\
  --trials 30 \\
  --base-seed 20260528 \\
  --output results\_paper\_a\_independent \\
  --config configs/default\_config.yaml
```
What this script does:
runs `PSO`, `GWO`, and `ACO`
across all 6 scenarios
creates fresh algorithm and scenario instances for each trial
uses a distinct deterministic seed for each `algorithm + scenario + trial`
Output:
```text
results\_paper\_a\_independent/<timestamp>/
```
Main files created:
`benchmark\_\*.json`
`benchmark\_stats\_\*.json`
`summary.json`
`analysis\_summary.json`
`friedman\_ranks.csv`
`wilcoxon\_exact\_pvalues.csv`
3. Create Scenario GIFs
Use the raw benchmark JSON file from the independent run:
```bash
python create\_scenario\_gifs.py \\
  results\_paper\_a\_independent/-------------/benchmark\_--------------.json
```
This creates:
```text
results/scenario_gifs/
├── per\_scenario/
├── combined/
└── screenshots/
```
Files produced:
one GIF per scenario in `per\_scenario/`
one combined GIF in `combined/`
sample PNG frames in `screenshots/`



# GIF results of 6 scenarios (combined with three algorithms)


