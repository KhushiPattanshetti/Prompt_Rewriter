import os
import json
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import networkx as nx
import requests

# --- Config ---
ICD10_TREE_PATH = os.path.join(os.path.dirname(__file__), "icd10_tree.json")
GT_CODES_DIR = os.path.join(os.path.dirname(__file__), "gt_codes")
RL_BLOCK_ENDPOINT = os.environ.get(
    "RL_BLOCK_ENDPOINT", None
)  # Set this in your environment


# --- Data Models ---
class RewardRequest(BaseModel):
    enh_codes: List[str]
    org_codes: List[str]
    gt_codes: Optional[List[str]] = None
    gt_file: Optional[str] = None  # Optionally specify a file in gt_codes/


class RewardResponse(BaseModel):
    reward: float


# --- Load ICD-10 Tree ---
def load_icd10_tree():
    with open(ICD10_TREE_PATH, "r") as f:
        tree = json.load(f)
    G = nx.DiGraph()
    for parent, children in tree.items():
        for child in children:
            G.add_edge(parent.upper(), child.upper())
        if not children:
            G.add_node(parent.upper())
    return G


ICD10_GRAPH = load_icd10_tree()


# --- Utility Functions ---
def get_max_path_length(graph):
    # Longest shortest path in the graph
    lengths = dict(nx.all_pairs_shortest_path_length(graph))
    max_len = 0
    for d in lengths.values():
        if d:
            max_len = max(max_len, max(d.values()))
    return max_len or 1


MAX_PATH_LENGTH = get_max_path_length(ICD10_GRAPH)


def load_gt_codes(gt_file: str) -> List[str]:
    path = os.path.join(GT_CODES_DIR, gt_file)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Ground truth file not found: {gt_file}")
    with open(path, "r") as f:
        codes = json.load(f)
    return [c.upper() for c in codes]


def normalize_codes(codes: List[str]) -> List[str]:
    return [c.upper() for c in codes]


def distance_between(a: List[str], b: List[str], graph: nx.DiGraph) -> float:
    # Symmetric average of min shortest path distances
    a = normalize_codes(a)
    b = normalize_codes(b)
    if not a or not b:
        return 1.0  # Max distance if any set is empty
    dists = []
    for code_a in a:
        min_dist = min(
            (
                nx.shortest_path_length(graph, code_a, code_b)
                if nx.has_path(graph, code_a, code_b)
                else MAX_PATH_LENGTH
            )
            for code_b in b
        )
        dists.append(min_dist)
    for code_b in b:
        min_dist = min(
            (
                nx.shortest_path_length(graph, code_b, code_a)
                if nx.has_path(graph, code_b, code_a)
                else MAX_PATH_LENGTH
            )
            for code_a in a
        )
        dists.append(min_dist)
    avg_dist = sum(dists) / len(dists)
    return min(avg_dist / MAX_PATH_LENGTH, 1.0)


def calculate_reward(gt_codes, enh_codes, org_codes, graph):
    d_gt_enh = distance_between(gt_codes, enh_codes, graph)
    d_gt_org = distance_between(gt_codes, org_codes, graph)
    d_enh_org = distance_between(enh_codes, org_codes, graph)
    if d_gt_enh < d_gt_org:
        return round(1.0 - d_gt_enh, 4)
    elif d_gt_enh > d_gt_org:
        return round(-d_gt_enh, 4)
    else:
        if d_enh_org == 0:
            return 1.0
        else:
            return round(-d_gt_enh, 4)


def post_to_rl_block(reward: float):
    if RL_BLOCK_ENDPOINT:
        try:
            requests.post(RL_BLOCK_ENDPOINT, json={"reward": reward})
        except Exception as e:
            print(f"Failed to POST reward to RL block: {e}")


# --- FastAPI App ---
app = FastAPI()


@app.post("/reward", response_model=RewardResponse)
def reward_endpoint(req: RewardRequest):
    if req.gt_codes:
        gt_codes = req.gt_codes
    elif req.gt_file:
        gt_codes = load_gt_codes(req.gt_file)
    else:
        raise HTTPException(status_code=400, detail="gt_codes or gt_file required")
    enh_codes = req.enh_codes
    org_codes = req.org_codes
    reward = calculate_reward(gt_codes, enh_codes, org_codes, ICD10_GRAPH)
    post_to_rl_block(reward)
    return RewardResponse(reward=reward)
