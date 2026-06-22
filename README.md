# CoReLIN: Constraint-based Reasoning for Zero-shot Lifelong Interactive Navigation

**Accepted at the European Conference on Computer Vision (ECCV) 2026.**

Official implementation and dataset for **“CoReLIN: Constraint-based Reasoning for Zero-shot Lifelong Interactive Navigation.”**

CoReLIN addresses **Lifelong Interactive Navigation**: a mobile manipulator must complete a sequence of object-placement tasks in a partially observed, cluttered environment. When clutter blocks a route, the agent must decide whether to detour or permanently relocate an obstacle, while accounting for the effect of that decision on future tasks.

The framework combines:

- an LLM-based high-level planner for exploration and constraint resolution;
- an incrementally constructed scene-graph representation;
- grid-based navigation and path analysis;
- AI2-THOR / ProcTHOR physics-enabled simulation; and
- pick-and-place manipulation primitives for rearranging clutter and completing tasks.

**Paper:** [Project PDF](https://accgen99.github.io/assets/pdf/vashisth2026move.pdf) · [arXiv:2602.20055](https://arxiv.org/abs/2602.20055)

> **Release status.** This repository contains the simulator implementation, prompts, and generated train/test task configurations used by the project. The current public entry point runs one evaluation episode at a time; scripts for reproducing every table, baseline, and real-robot experiment from the paper are not included in this snapshot.

## Method overview

At each planning step, CoReLIN serializes the currently observed environment into a structured representation containing rooms, objects, task progress, robot state, path blockers, detour costs, and manipulation costs. The LLM then selects among four high-level operations:

1. **Scan a room** to discover task-relevant objects and connectivity.
2. **Navigate to another room** for targeted exploration.
3. **Inspect path constraints** for a selected goal object.
4. **Execute a plan** that detours around selected blockers or relocates them to valid receptacles.

Low-level navigation and manipulation are executed by the simulator rather than generated directly by the LLM. Environment modifications persist across the sequence of 20 tasks, making early rearrangement decisions relevant to later navigation efficiency.

## Repository structure

```text
.
├── llm_planner.py              # LLM planner and evaluation entry point
├── env.py                      # AI2-THOR environment and action execution
├── params.py                   # Robot, grid, cost, and output parameters
├── environment.yml             # Conda environment specification
├── api_key.txt                 # Local OpenAI API key file; keep untracked
├── prompts/
│   ├── system.txt              # Task and scene-representation prompt
│   └── developer.txt           # Long-horizon planning policy
├── classes/
│   ├── get_floorplan.py        # ProcTHOR floorplan utilities
│   ├── grid_graph.py           # Reachability graph construction
│   ├── price_of_clutter.py     # Price of Clutter metric
│   ├── receptacle_info.py      # Object-to-receptacle compatibility
│   ├── representation.py       # Structured scene representation
│   └── valid_actions.py        # Feasible navigation/action analysis
└── dataset/
    ├── train/                   # 10,000 generated training configurations
    └── test/                    # 100 test configurations grouped by room count
```

Each dataset JSON file contains a ProcTHOR house identifier, a sequence of 20 object-to-receptacle goals, procedurally placed clutter objects, and the robot’s initial pose and orientation.

## Requirements

The provided environment uses:

- Python 3.10
- AI2-THOR 5.0.0
- OpenAI Python SDK
- NumPy, NetworkX, SciPy, Shapely, scikit-image, and Matplotlib

A machine capable of launching AI2-THOR is required. Depending on the host platform, AI2-THOR may also require a graphical display or a supported headless rendering setup.

The planner makes paid API calls to an OpenAI model. API availability, model access, latency, and cost depend on your account and selected model.

## Installation

Clone the repository and create the provided Conda environment:

```bash
git clone https://github.com/IDEAS-Lab-Purdue/CoReLIN.git
cd CoReLIN

conda env create -f environment.yml
conda activate corelin
```

The environment file is a fully pinned development snapshot and may contain packages not required by the simulator-only release. For a different operating system or architecture, you may need to relax platform-specific pins while retaining the Python dependencies listed under `pip`.

## API configuration

The current implementation reads the API key from `api_key.txt` in the repository root:

```bash
printf '%s' 'YOUR_OPENAI_API_KEY' > api_key.txt
```

Do **not** commit this file. Add it to `.gitignore` before publishing:

```gitignore
api_key.txt
```

For a public deployment, using the `OPENAI_API_KEY` environment variable instead of a plaintext file is recommended.

## Running an episode

The default evaluation example is configured at the bottom of `llm_planner.py`:

```python
dataset_path = "dataset/test/5/3.json"
test = llm("gpt-5-mini-2025-08-07", dataset_path, True)
test.run_episode()
```

Before running, update:

- `dataset_path` to the desired task configuration;
- the model name to a Responses-API model available to your account; and
- the final Boolean argument to `False` when visualizations should not be saved.

Then run:

```bash
python llm_planner.py
```

The bundled example uses the third test configuration for a five-room environment. Test configurations follow this pattern:

```text
dataset/test/<number-of-rooms>/<example-id>.json
```

For example:

```text
dataset/test/1/1.json
dataset/test/5/3.json
dataset/test/10/11.json
```

## Outputs

Output locations are controlled by `FOLDER_NAME` and related paths in `params.py`:

```python
FOLDER_NAME = "final"
```

During execution, the code may create:

```text
gifs/<FOLDER_NAME>/
scene_graphs/<FOLDER_NAME>/
scene_data/
results/<FOLDER_NAME>/
results/llm_outputs.txt
```

The entry point appends episode metrics to:

```text
results/<FOLDER_NAME>/r_<number-of-rooms>.csv
```

The written columns are:

```text
SR, PL, OI, NI, interacted_fraction, relative_betweenness,
Price_of_Clutter, total_cost, steps
```

These files do not include a header in the current implementation.

## Dataset

The release contains:

- **10,000 training configurations**, each with 20 sequential placement tasks; and
- **100 test configurations**, distributed across floorplans containing 1–10 rooms.

Clutter is generated by selecting traversable grid nodes with probability biased by betweenness centrality, increasing the likelihood of obstacles appearing around bottlenecks, hallways, and intersections. The test directory is organized by room count.

The JSON schema is:

```json
{
  "house_id": 47,
  "goals": {
    "1": ["Mug", "CounterTop"],
    "2": ["Knife", "DiningTable"]
  },
  "spawn_objects": [
    {
      "objectName": "Bread|surface|2|3",
      "position": {"x": 1.0, "y": 0.0, "z": 2.5},
      "rotation": {"x": 0.0, "y": 180.0, "z": 0.0}
    }
  ],
  "start_pose": [0.25, 0.9, 3.75],
  "initial_orientation": 270.0
}
```

## Important parameters

Key values in `params.py` include:

| Parameter | Default | Description |
|---|---:|---|
| `ROBOT_RADIUS` | `0.2` | Cylindrical collision radius in meters |
| `ROBOT_HEIGHT` | `1.575` | Robot height in meters |
| `GRID_SIZE` | `0.25` | Navigation-grid spacing in meters |
| `MANIP_COST` | `5.0` | Cost assigned to manipulation |
| `H_LEN` | `2` | Number of recent interaction blocks retained for LLM context |
| `FOLDER_NAME` | `final` | Output subdirectory name |

## Metrics

The paper evaluates task completion, execution efficiency, and the long-term navigability of the modified environment. This code includes support for metrics such as:

- **Success Rate (SR):** fraction of assigned goals completed;
- **timesteps / path-related execution statistics**;
- **object interactions and navigation interactions**; and
- **Price of Clutter (PoC):** degradation in all-pairs shortest-path distances relative to the uncluttered floorplan.

## Metric computation

The `metrics/` directory contains scripts for aggregating experiment outputs and computing the metrics reported in the paper.

### Expected results structure

Store the raw evaluation CSV files under the root-level `results/` directory:

```text
results/
├── ours_k/
│   ├── r_1.csv
│   ├── r_2.csv
│   └── ...
├── h_k/
├── ours_unk/
└── h_uk/
```

Each method directory should contain one CSV file per evaluated room count, named `r_<num_rooms>.csv`. The metric scripts currently evaluate room counts `1, 2, 3, 4, 5, 6, 7, 8, and 10`.

Each CSV row is expected to contain the following fields, without a header row:

```text
success_rate,path_length,num_interacts,num_navs,perc_interacts,relative_bw,overall_poc,goal_poc,cost,timesteps
```

### Per-room metrics

`metrics/per_room.py` aggregates the raw evaluation CSV files for each method and room count. It computes:

- **SR:** average task success rate, normalized by the 20 goals in each episode.
- **TS:** average execution cost.
- **PoC:** average overall Price of Clutter.

Run it from the repository root:

```bash
python metrics/per_room.py
```

The script writes the aggregated metrics to:

```text
metrics/per_room_perf.json
```

The method names and expected result directories are currently configured in `metrics/per_room.py`:

```python
names = ["ours_k", "h_k", "ours_unk", "h_uk"]
```

Edit this list when evaluating additional methods or baselines.

### Lifelong Efficiency Score

`metrics/new_les.py` reads `per_room_perf.json` and computes the Lifelong Efficiency Score (LES). Results are aggregated into three environment-size bins:

- `1–3` rooms
- `4–6` rooms
- `7–10` rooms

Run:

```bash
python metrics/new_les.py \
    --json_path metrics/per_room_perf.json
```

By default, LES uses the following weights:

```text
Success rate:      0.50
Execution cost:    0.25
Price of Clutter:  0.25
```

The weights can be changed from the command line:

```bash
python metrics/new_les.py \
    --json_path metrics/per_room_perf.json \
    --w_sr 0.5 \
    --w_ts 0.25 \
    --w_poc 0.25
```

The script globally normalizes execution cost and Price of Clutter across the evaluated methods, computes LES using a weighted geometric mean, and prints both readable results and rows formatted for direct use in LaTeX tables.


## Reproducing the paper

This snapshot supports running CoReLIN episodes in the supplied ProcTHOR task configurations. Exact reproduction of all paper results additionally requires experiment orchestration, baseline implementations, aggregation scripts, and the real-robot stack described in the paper; those components are not present in this archive.

Results can also vary with the selected LLM model and provider-side model updates. Record the exact model identifier, date, prompt files, package environment, and dataset configuration for every reported run.

## Important!

Replace the model identifier in `llm_planner.py` with a Responses-API model enabled for your OpenAI project.


`api_key.txt` is intentionally empty in the release archive. Populate it locally and ensure it remains ignored by version control.

## Citation

Please cite the paper when using this code or dataset:

```bibtex
@article{vashisth2026corelin,
  title   = {CoReLIN: Constraint-based Reasoning for Zero-shot Lifelong Interactive Navigation},
  author  = {Vashisth, Apoorva and Kulshrestha, Manav and Bakshi, Pranav and Conover, Damon and Sartoretti, Guillaume and Bera, Aniket},
  journal = {European Conference on Computer Vision},
  year    = {2026}
}
```

## Acknowledgments

This project uses [AI2-THOR](https://ai2thor.allenai.org/) and [ProcTHOR](https://procthor.allenai.org/) for physics-enabled indoor simulation, together with the OpenAI Responses API for high-level planning.
