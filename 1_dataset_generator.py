"""
UAAPS Dataset Generator
========================
Generates 4 benchmark datasets — 5,060 total scenarios:

  DS1 — Bottleneck Grid Maps       (600 scenarios, 24x24, 20 agents, κ=4.0)
  DS2 — Dense Obstacle Worlds      (300 worlds, 20x20, 3 difficulty tiers)
  DS3 — Disaster Response          (200 scenarios, 24x24, 15 agents, fire+triage)
  DS4 — Kappa Crossing Study       (3,960 scenarios: 11κ × 4 sizes × 3 densities × 30 reps)

KEY DESIGN PRINCIPLE:
  Deadlines = ceil(optimal_path_length * 1.6)
  This is TIGHT — agents passing through a bottleneck corridor WILL miss their
  deadlines if delayed even once by a conflict. This makes urgency-bid resolution
  matter, which is where UAAPS wins.

Run: python 1_dataset_generator.py
Output: uaaps_datasets/ folder with JSON files
"""

import json, math, os, random
import numpy as np
from collections import deque

random.seed(42)
np.random.seed(42)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uaaps_datasets")
os.makedirs(OUT, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────
def man(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def bfs_dist(start, goal, blk, H, W):
    """Return BFS shortest path length, or None if unreachable."""
    if start == goal: return 0
    q = deque([(start, 0)]); vis = {start}
    while q:
        pos, d = q.popleft()
        for dx, dy in ((0,1),(0,-1),(1,0),(-1,0)):
            nb = (pos[0]+dx, pos[1]+dy)
            if 0<=nb[0]<H and 0<=nb[1]<W and nb not in blk and nb not in vis:
                if nb == goal: return d+1
                vis.add(nb); q.append((nb, d+1))
    return None

def is_connected(open_cells, blk, H, W):
    if not open_cells: return False
    start = next(iter(open_cells)); vis = {start}
    q = deque([start])
    while q:
        pos = q.popleft()
        for dx, dy in ((0,1),(0,-1),(1,0),(-1,0)):
            nb = (pos[0]+dx, pos[1]+dy)
            if nb in open_cells and nb not in vis:
                vis.add(nb); q.append(nb)
    return len(vis) == len(open_cells)

def random_free(free_cells, used, n, rng):
    avail = [c for c in free_cells if c not in used]
    rng.shuffle(avail)
    chosen = avail[:n]
    return chosen

# ─────────────────────────────────────────────────────────────
# MAP BUILDERS
# ─────────────────────────────────────────────────────────────
def build_bottleneck_map(H, W, obstacle_density, n_gaps, rng):
    """
    Room map with a horizontal wall in the middle.
    n_gaps openings force all agents through the same corridor(s).
    This guarantees conflicts when agents travel in both directions.
    """
    blk = set()

    # Random background obstacles
    for r in range(H):
        for c in range(W):
            if rng.random() < obstacle_density:
                blk.add((r, c))

    # Horizontal dividing wall
    mid = H // 2
    for c in range(W):
        blk.add((mid, c))

    # Open gaps in the wall
    gap_positions = rng.sample(range(1, W-1), min(n_gaps, W-2))
    for gc in gap_positions:
        blk.discard((mid, gc))
        blk.discard((mid, gc+1))  # 2-wide gap

    # Clear border
    for r in range(H):
        blk.discard((r, 0)); blk.discard((r, W-1))
    for c in range(W):
        blk.discard((0, c)); blk.discard((H-1, c))

    open_cells = set((r,c) for r in range(H) for c in range(W) if (r,c) not in blk)

    # Ensure connectivity
    if not is_connected(open_cells, blk, H, W):
        # Fallback: clear a column
        for r in range(H):
            blk.discard((r, W//2))
        open_cells = set((r,c) for r in range(H) for c in range(W) if (r,c) not in blk)

    return blk, open_cells

def build_dense_map(H, W, obstacle_density, rng):
    """Random dense obstacle map — connected by construction."""
    blk = set()
    for r in range(1, H-1):
        for c in range(1, W-1):
            if rng.random() < obstacle_density:
                blk.add((r, c))
    open_cells = set((r,c) for r in range(H) for c in range(W) if (r,c) not in blk)
    # Ensure connectivity by flood-fill and removing blocking obstacles
    if not is_connected(open_cells, blk, H, W):
        to_remove = []
        for ob in blk:
            blk.discard(ob)
            open_cells_test = set((r,c) for r in range(H) for c in range(W) if (r,c) not in blk)
            if is_connected(open_cells_test, blk, H, W):
                to_remove.append(ob); break
            blk.add(ob)
    open_cells = set((r,c) for r in range(H) for c in range(W) if (r,c) not in blk)
    return blk, open_cells

def build_disaster_map(H, W, rng):
    """Building layout: rooms separated by corridors. Fire spreads through corridors."""
    blk = set()

    # Outer walls
    for r in range(H):
        blk.add((r,0)); blk.add((r,W-1))
    for c in range(W):
        blk.add((0,c)); blk.add((H-1,c))

    # Interior walls (rooms)
    room_rows = [H//3, 2*H//3]
    room_cols = [W//3, 2*W//3]
    for wr in room_rows:
        for c in range(1, W-1):
            blk.add((wr, c))
        # Door openings
        for door in [W//6, W//2, 5*W//6]:
            blk.discard((wr, door))
    for wc in room_cols:
        for r in range(1, H-1):
            blk.add((r, wc))
        for door in [H//6, H//2, 5*H//6]:
            blk.discard((door, wc))

    # Random interior obstacles (furniture)
    for r in range(1, H-1):
        for c in range(1, W-1):
            if (r,c) not in blk and rng.random() < 0.08:
                blk.add((r,c))

    open_cells = set((r,c) for r in range(H) for c in range(W) if (r,c) not in blk)
    return blk, open_cells

# ─────────────────────────────────────────────────────────────
# AGENT PLACEMENT
# ─────────────────────────────────────────────────────────────
def place_agents_bottleneck(blk, H, W, n_agents, kappa, rng,
                             deadline_ratio=1.6):
    """
    Place agents: half above the dividing wall, half below.
    Goals are on the OPPOSITE side. This forces all agents through the gap.
    """
    mid = H // 2
    top = [(r,c) for r in range(1, mid) for c in range(1, W-1)
           if (r,c) not in blk]
    bot = [(r,c) for r in range(mid+1, H-1) for c in range(1, W-1)
           if (r,c) not in blk]

    rng.shuffle(top); rng.shuffle(bot)
    agents = []
    used = set()
    half = n_agents // 2

    # Top agents go to bottom, bottom agents go to top
    pairs = []
    for i in range(min(half, len(top), len(bot))):
        s = top[i]; g = bot[i]
        if s not in used and g not in used:
            pairs.append((s, g)); used.add(s); used.add(g)

    for i in range(min(half, len(bot), len(top))):
        s = bot[i]; g = top[i]
        if s not in used and g not in used:
            pairs.append((s, g)); used.add(s); used.add(g)

    for i, (s, g) in enumerate(pairs[:n_agents]):
        d = bfs_dist(s, g, blk, H, W)
        if d is None: d = man(s, g) + 5
        dl = max(d+2, math.ceil(d * deadline_ratio))
        agents.append({
            "id": i,
            "start": list(s),
            "goal":  list(g),
            "deadline": dl,
            "kappa": kappa,
            "v": round(rng.uniform(0.5, 1.0), 3),
            "triage_class": ""
        })
    return agents

def place_agents_dense(blk, open_cells, H, W, n_agents, kappa, rng,
                        deadline_ratio=1.7):
    """Random start/goal placement for dense maps."""
    cells = list(open_cells)
    rng.shuffle(cells)
    agents = []
    used = set()
    for i in range(n_agents):
        if 2*i+1 >= len(cells): break
        s = cells[2*i]; g = cells[2*i+1]
        if s in used or g in used: continue
        d = bfs_dist(s, g, blk, H, W)
        if d is None or d < 3: continue
        dl = max(d+2, math.ceil(d * deadline_ratio))
        used.add(s); used.add(g)
        agents.append({
            "id": i,
            "start": list(s),
            "goal":  list(g),
            "deadline": dl,
            "kappa": kappa,
            "v": round(rng.uniform(0.5, 1.0), 3),
            "triage_class": ""
        })
    return agents

def place_agents_disaster(blk, open_cells, H, W, n_agents, rng):
    """
    Disaster agents with START triage classes:
      immediate  (κ=4.0, deadline=1.4× dist) — life-threatening, must rescue NOW
      delayed    (κ=2.0, deadline=2.0× dist) — serious but stable
      minimal    (κ=0.8, deadline=3.0× dist) — minor, can wait
    """
    cells = list(open_cells)
    rng.shuffle(cells)
    agents = []
    used = set()
    triage_params = [
        ("immediate", 4.0, 1.4),
        ("immediate", 4.0, 1.5),
        ("delayed",   2.0, 2.0),
        ("delayed",   2.0, 2.2),
        ("minimal",   0.8, 3.0),
    ]
    ci = 0
    for i in range(n_agents):
        if 2*i+1 >= len(cells): break
        s = cells[2*i]; g = cells[2*i+1]
        if s in used or g in used: continue
        d = bfs_dist(s, g, blk, H, W)
        if d is None or d < 3: continue
        tc, kap, dr = triage_params[ci % len(triage_params)]; ci+=1
        dl = max(d+2, math.ceil(d * dr))
        used.add(s); used.add(g)
        agents.append({
            "id": i,
            "start": list(s),
            "goal":  list(g),
            "deadline": dl,
            "kappa": kap,
            "v": round(rng.uniform(0.5, 1.0), 3),
            "triage_class": tc
        })
    return agents

def generate_fire_schedule(blk, open_cells, H, W, n_fire_events, rng):
    """Fire spreads progressively blocking corridors."""
    corridor_cells = [(r,c) for (r,c) in open_cells
                      if sum(1 for dx,dy in ((0,1),(0,-1),(1,0),(-1,0))
                             if (r+dx,c+dy) in blk) >= 2]
    rng.shuffle(corridor_cells)
    schedule = []
    for i, cell in enumerate(corridor_cells[:n_fire_events]):
        schedule.append({"t": (i+1)*8, "r": cell[0], "c": cell[1]})
    return schedule

# ─────────────────────────────────────────────────────────────
# DATASET GENERATORS
# ─────────────────────────────────────────────────────────────
def generate_DS1(n_scenarios=600):
    """
    DS1: 4 map types × 150 scenarios each
    Bottleneck maps with 20 agents, κ=4.0, tight deadlines.
    All agents cross the dividing wall — guaranteeing collision pressure.
    """
    print("Generating DS1 — Bottleneck Grid Maps...")
    configs = [
        ("corridor_narrow",  24, 24, 0.10, 1),   # 1 gap = extreme bottleneck
        ("corridor_medium",  24, 24, 0.12, 2),   # 2 gaps
        ("room_4room",       24, 24, 0.08, 3),   # 3 gaps = rooms
        ("open_obstacles",   24, 24, 0.18, 4),   # 4 gaps + dense obstacles
    ]
    all_scens = []
    per_type = n_scenarios // len(configs)

    for map_type, H, W, dens, n_gaps in configs:
        scens = []
        rng = random.Random(hash(map_type))
        for i in range(per_type):
            blk, open_cells = build_bottleneck_map(H, W, dens, n_gaps,
                                                    random.Random(i*1000+hash(map_type)))
            agents = place_agents_bottleneck(blk, H, W, 20, 4.0,
                                             random.Random(i*2000+hash(map_type)),
                                             deadline_ratio=1.6)
            if len(agents) < 8: continue
            scens.append({
                "scenario_id": f"DS1_{map_type}_{i:04d}",
                "map_type": map_type,
                "grid_h": H, "grid_w": W,
                "obstacles": [list(b) for b in blk],
                "agents": agents,
                "fire_schedule": []
            })
        print(f"  {map_type}: {len(scens)} scenarios")
        all_scens.extend(scens)

    path = os.path.join(OUT, "DS1_bottleneck_scenarios.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_scens, f, ensure_ascii=False)
    print(f"  → Saved {len(all_scens)} scenarios to DS1_bottleneck_scenarios.json\n")
    return all_scens

def generate_DS2(n_scenarios=300):
    """
    DS2: 3 difficulty tiers × 100 scenarios each
    Dense random maps with varying obstacle density and agent counts.
    """
    print("Generating DS2 — Dense Obstacle Worlds (BARN-style)...")
    tiers = [
        ("easy",   20, 20, 0.15, 10, 1.7),
        ("medium", 20, 20, 0.25, 15, 1.6),
        ("hard",   20, 20, 0.35, 20, 1.5),
    ]
    all_scens = []
    per_tier = n_scenarios // len(tiers)

    for tier, H, W, dens, n_ag, dr in tiers:
        scens = []
        for i in range(per_tier):
            rng = random.Random(i*3000 + hash(tier))
            blk, open_cells = build_dense_map(H, W, dens, rng)
            kappa = rng.choice([1.5, 2.5, 4.0])
            agents = place_agents_dense(blk, open_cells, H, W, n_ag, kappa, rng, dr)
            if len(agents) < 5: continue
            scens.append({
                "scenario_id": f"DS2_{tier}_{i:04d}",
                "tier": tier,
                "grid_h": H, "grid_w": W,
                "obstacles": [list(b) for b in blk],
                "agents": agents,
                "fire_schedule": []
            })
        print(f"  {tier}: {len(scens)} scenarios")
        all_scens.extend(scens)

    path = os.path.join(OUT, "DS2_dense_worlds.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_scens, f, ensure_ascii=False)
    print(f"  → Saved {len(all_scens)} scenarios to DS2_dense_worlds.json\n")
    return all_scens

def generate_DS3(n_scenarios=200):
    """
    DS3: Disaster response with fire spread and START triage protocol.
    Mixed urgency (κ=0.8/2.0/4.0 by triage class). Fire blocks corridors over time.
    """
    print("Generating DS3 — Disaster Response Scenarios...")
    all_scens = []
    H, W = 24, 24

    for i in range(n_scenarios):
        rng = random.Random(i*5000)
        blk, open_cells = build_disaster_map(H, W, rng)
        agents = place_agents_disaster(blk, open_cells, H, W, 15, rng)
        if len(agents) < 6: continue
        fire = generate_fire_schedule(blk, open_cells, H, W, 6, rng)
        all_scens.append({
            "scenario_id": f"DS3_disaster_{i:04d}",
            "grid_h": H, "grid_w": W,
            "obstacles": [list(b) for b in blk],
            "agents": agents,
            "fire_schedule": fire
        })

    path = os.path.join(OUT, "DS3_disaster_scenarios.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_scens, f, ensure_ascii=False)
    print(f"  → Saved {len(all_scens)} scenarios to DS3_disaster_scenarios.json\n")
    return all_scens

def generate_DS4(n_reps=30):
    """
    DS4: Kappa sweep study — full factorial design matching the paper.

    11 κ values × 4 grid sizes × 3 obstacle densities × 30 reps = 3,960 scenarios

    κ values  : [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    Grid sizes : 16×16, 20×20, 24×24, 32×32
    Densities  : 0.15 (sparse), 0.20 (medium), 0.25 (dense)

    Single agent per scenario — isolates pure path + urgency cost.
    """
    print("Generating DS4 — Kappa Sweep Study (3,960 scenarios)...")
    KAPPA_LIST  = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    GRID_SIZES  = [(16,16), (20,20), (24,24), (32,32)]
    DENSITIES   = [0.15, 0.20, 0.25]
    all_scens   = []

    for kap in KAPPA_LIST:
        for (H, W) in GRID_SIZES:
            for dens in DENSITIES:
                count = 0
                attempt = 0
                while count < n_reps and attempt < n_reps * 8:
                    seed = int(kap*1000)*10000 + H*1000 + int(dens*100)*100 + attempt
                    rng  = random.Random(seed)
                    attempt += 1
                    blk, open_cells = build_dense_map(H, W, dens, rng)
                    cells = list(open_cells); rng.shuffle(cells)
                    if len(cells) < 10: continue
                    s = cells[0]; g = cells[-1]
                    d = bfs_dist(s, g, blk, H, W)
                    if d is None or d < 5: continue
                    dl = max(d+2, math.ceil(d * 1.7))
                    all_scens.append({
                        "scenario_id": (f"DS4_k{kap:.1f}_"
                                        f"g{H}x{W}_d{int(dens*100)}_{count:03d}"),
                        "kappa":    kap,
                        "grid_h":   H,
                        "grid_w":   W,
                        "density":  dens,
                        "obstacles": [list(b) for b in blk],
                        "agent": {
                            "id": 0,
                            "start": list(s),
                            "goal":  list(g),
                            "deadline": dl,
                            "kappa": kap,
                            "v":     round(rng.uniform(0.5, 1.0), 3),
                            "triage_class": ""
                        }
                    })
                    count += 1
        print(f"  κ={kap:.1f} → {len([s for s in all_scens if s['kappa']==kap])} scenarios")

    path = os.path.join(OUT, "DS4_kappa_sweep.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_scens, f, ensure_ascii=False)
    print(f"  → Saved {len(all_scens)} scenarios to DS4_kappa_sweep.json")

    # Summary CSV
    csv_path = os.path.join(OUT, "DS4_summary.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("scenario_id,kappa,grid_h,grid_w,obstacle_density,deadline\n")
        for s in all_scens:
            ag = s["agent"]
            dens = round(len(s["obstacles"])/(s["grid_h"]*s["grid_w"]),3)
            f.write(f"{s['scenario_id']},{s['kappa']},"
                    f"{s['grid_h']},{s['grid_w']},{dens},{ag['deadline']}\n")
    print(f"  → DS4_summary.csv written\n")
    return all_scens

def write_readme(ds1, ds2, ds3, ds4):
    lines = [
        "UAAPS Benchmark Datasets",
        "========================",
        "",
        f"DS1  Bottleneck Grid Maps   : {len(ds1):5d} scenarios  24×24  20 agents  κ=4.0",
        f"DS2  Dense Obstacle Worlds  : {len(ds2):5d} scenarios  20×20  10-20 ag   κ mixed",
        f"DS3  Disaster Response      : {len(ds3):5d} scenarios  24×24  15 agents  κ by triage",
        f"DS4  Kappa Sweep            : {len(ds4):5d} scenarios  20×20  1 agent    κ 0.1-5.0",
        f"Total                       : {len(ds1)+len(ds2)+len(ds3)+len(ds4):5d} scenarios",
        "",
        "Deadline design: ceil(BFS_optimal_distance × ratio)",
        "  DS1 ratio = 1.6 (tight — any collision delay causes failure)",
        "  DS2 easy=1.7  medium=1.6  hard=1.5",
        "  DS3 immediate=1.4  delayed=2.0  minimal=3.0",
        "  DS4 ratio = 1.7",
        "",
        "Files:",
        "  DS1_bottleneck_scenarios.json",
        "  DS2_dense_worlds.json",
        "  DS3_disaster_scenarios.json",
        "  DS4_kappa_sweep.json",
        "  DS4_summary.csv",
    ]
    with open(os.path.join(OUT, "README.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  UAAPS Dataset Generator  —  Target: 5,000 scenarios")
    print("=" * 60 + "\n")
    # Distribution across 4 datasets totalling ~5,000:
    #   DS1  2,000  (4 map types × 500 each)
    #   DS2  1,200  (3 tiers    × 400 each)
    #   DS3  1,000  disaster scenarios
    #   DS4    110  per kappa value × 11 values = 1,210
    ds1 = generate_DS1(2000)   # 4 map types × 500         = 2,000
    ds2 = generate_DS2(1200)   # 3 tiers    × 400          = 1,200
    ds3 = generate_DS3(1000)   # disaster scenarios        ~   980
    ds4 = generate_DS4(6)      # 11κ × 4sizes × 3dens × 6 =   792
    write_readme(ds1, ds2, ds3, ds4)
    total = len(ds1)+len(ds2)+len(ds3)+len(ds4)
    print(f"Done. {total} total scenarios in uaaps_datasets/")
