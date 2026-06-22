'''
Get final performance metrics per room.
'''

import pandas as pd
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"

names = ["ours_k", "h_k", "ours_unk", "h_uk"]
header = ["success_rate", "path_length", "num_interacts", "num_navs", "perc_interacts", "relative_bw", "overall_poc", "goal_poc", "cost", "timesteps"]

perf_dict = {}
NUM_GOALS = 20.00
min_Poc = float('inf')
for room in [1, 2, 3, 4, 5, 6, 7, 8, 10]:
    perf_dict[str(room)] = {}
    for name in names:
        csv_name = RESULTS_DIR / name / f"r_{int(room)}.csv"
        df = pd.read_csv(csv_name, names=header)
        mean_SR = df["success_rate"].mean()/NUM_GOALS
        mean_TS = df["cost"].mean()
        mean_poc = df["overall_poc"].mean()
        perf_dict[str(room)][name] = {"SR": mean_SR, "TS": mean_TS, "PoC": mean_poc}
        min_Poc = min(min_Poc, mean_poc)

output_path = Path(__file__).resolve().parent / "per_room_perf.json"
with output_path.open("w") as f:
    json.dump(perf_dict, f, indent=2)
