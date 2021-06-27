import numpy as np
import pandas as pd
from ortools.sat.python import cp_model
from ortools.linear_solver import pywraplp
from tqdm import tqdm

from . import layers, superitems, utils


STATUS_STRING = {
    cp_model.OPTIMAL: "optimal",
    cp_model.FEASIBLE: "feasible",
    cp_model.INFEASIBLE: "infeasible",
    cp_model.MODEL_INVALID: "invalid",
    cp_model.UNKNOWN: "unknown",
}


def main_problem(fsi, zsl, ol, tlim=None, relaxation=True):
    # Solver
    if relaxation:
        slv = pywraplp.Solver.CreateSolver("CLP")
    else:
        slv = pywraplp.Solver.CreateSolver("CBC")

    # Utility
    infinity = slv.infinity()
    n_superitems, n_layers = zsl.shape
    n_items = fsi.shape[-1]

    # Variables
    if relaxation:
        al = [slv.NumVar(0, infinity, f"alpha_{l}") for l in range(n_layers)]
    else:
        al = [slv.IntVar(0, 1, f"alpha_{l}") for l in range(n_layers)]

    # Constraints
    for i in range(n_items):
        if relaxation:
            slv.Add(
                -sum(
                    fsi[s, i] * zsl[s, l] * al[l]
                    for s in range(n_superitems)
                    for l in range(n_layers)
                )
                <= -1
            )
        else:
            slv.Add(
                sum(
                    fsi[s, i] * zsl[s, l] * al[l]
                    for s in range(n_superitems)
                    for l in range(n_layers)
                )
                >= 1
            )

    # Objective
    obj = sum(h * al[l] for l, h in enumerate(ol))
    if relaxation:
        slv.Minimize(obj)
    else:
        slv.Maximize(-obj)

    # Set a time limit
    if tlim is not None:
        slv.SetTimeLimit(1000 * tlim)

    # Solve
    status = slv.Solve()

    # Extract results
    sol, duals = None, None
    if status in (slv.OPTIMAL, slv.FEASIBLE):
        sol = {f"alpha_{l}": al[l].solution_value() for l in range(n_layers)}
        sol["objective"] = slv.Objective().Value()
        if relaxation:
            duals = np.array([-c.dual_value() for c in slv.constraints()])

    # Return results
    return sol, slv.WallTime() / 1000, duals


def pricing_problem_no_placement(fsi, ws, ds, hs, W, D, duals, feasibility=None, tlim=None):
    # Solver
    slv = pywraplp.Solver.CreateSolver("CBC")

    # Utility
    infinity = slv.infinity()
    n_superitems, n_items = fsi.shape

    # Variables
    ol = slv.NumVar(0, infinity, f"o_l")
    zsl = [slv.IntVar(0, 1, f"z_{s}_l") for s in range(n_superitems)]

    # Constraints
    # Redundant valid cuts that force the area of
    # a layer to fit within the area of a bin
    slv.Add(sum(ws[s] * ds[s] * zsl[s] for s in range(n_superitems)) <= W * D)
    for s in range(n_superitems):
        # Define the height of layer l
        slv.Add(ol >= hs[s] * zsl[s])

    # Enforce feasible placement
    if feasibility is not None:
        print("Adding feasibility constraint: num selected <=", feasibility - 1)
        slv.Add(sum(zsl[s] for s in range(n_superitems)) <= feasibility - 1)

    # Compute reward for greater number of selected superitems
    upper_bound_reward = min(duals[i] for i in range(n_items) if duals[i] > 0) + n_superitems
    reward = (
        sum(zsl[s] for i in range(n_items) for s in range(n_superitems) if duals[i] == 0)
        / upper_bound_reward
    )
    print(
        "Reward about selecting superitems with duals ==0, Upper bound:",
        upper_bound_reward,
    )
    # Objective
    obj = (
        ol
        - sum(duals[i] * fsi[s, i] * zsl[s] for i in range(n_items) for s in range(n_superitems))
        - reward
    )
    slv.Minimize(obj)

    # Set a time limit
    if tlim is not None:
        slv.SetTimeLimit(1000 * tlim)

    # Solve
    status = slv.Solve()

    # Extract results
    sol = dict()
    if status in (slv.OPTIMAL, slv.FEASIBLE):
        sol[f"o_l"] = ol.solution_value()
        for s in range(n_superitems):
            sol[f"z_{s}_l"] = zsl[s].solution_value()
        sol["objective"] = slv.Objective().Value()

    return sol, slv.WallTime() / 1000


def pricing_problem_placement_v2(superitems_in_layer, ws, ds, W, D, tlim=None):
    # Solver
    slv = pywraplp.Solver.CreateSolver("CBC")

    # Variables
    cix = {s: slv.IntVar(0, int(W - ws[s]), f"c_{s}_x") for s in superitems_in_layer}
    ciy = {s: slv.IntVar(0, int(D - ds[s]), f"c_{s}_y") for s in superitems_in_layer}
    xsj, ysj = dict(), dict()
    for s in superitems_in_layer:
        for j in superitems_in_layer:
            if j != s:
                xsj[(s, j)] = slv.IntVar(0, 1, f"x_{s}_{j}")
                ysj[(s, j)] = slv.IntVar(0, 1, f"y_{s}_{j}")

    # Constraints
    # Redundant valid cuts that force the area of
    # a layer to fit within the area of a bin
    slv.Add(sum(ws[s] * ds[s] for s in superitems_in_layer) <= W * D)

    # Enforce at least one relative positioning relationship
    # between each pair of items in a layer
    for s in superitems_in_layer:
        for j in superitems_in_layer:
            if j > s:
                slv.Add(xsj[s, j] + xsj[j, s] + ysj[s, j] + ysj[j, s] >= 1)

    # Ensure that there is at most one spatial relationship
    # between items i and j along the width and depth dimensions
    for s in superitems_in_layer:
        for j in superitems_in_layer:
            if j > s:
                slv.Add(xsj[s, j] + xsj[j, s] <= 1)
                slv.Add(ysj[s, j] + ysj[j, s] <= 1)

    # Non-overlapping constraints
    for s in superitems_in_layer:
        for j in superitems_in_layer:
            if j != s:
                slv.Add(cix[s] + ws[s] <= cix[j] + W * (1 - xsj[s, j]))
                slv.Add(ciy[s] + ds[s] <= ciy[j] + D * (1 - ysj[s, j]))

    # Set a time limit
    if tlim is not None:
        slv.SetTimeLimit(1000 * tlim)

    # Solve
    status = slv.Solve()

    # Extract results
    sol = dict()
    if status in (slv.OPTIMAL, slv.FEASIBLE):
        for s in superitems_in_layer:
            sol[f"c_{s}_x"] = cix[s].solution_value()
            sol[f"c_{s}_y"] = ciy[s].solution_value()
        sol["objective"] = slv.Objective().Value()

    return sol, slv.WallTime() / 1000


def pricing_problem_placement(
    layer_height,
    superitems_in_layer,
    items_in_layer,
    fsi,
    ws,
    ds,
    hs,
    W,
    D,
    duals,
    tlim=None,
):
    # Solver
    slv = pywraplp.Solver.CreateSolver("CBC")

    # Variables
    zsl = {s: slv.IntVar(0, 1, f"z_{s}_l") for s in superitems_in_layer}
    cix = {s: slv.IntVar(0, int(W - ws[s]), f"c_{s}_x") for s in superitems_in_layer}
    ciy = {s: slv.IntVar(0, int(D - ds[s]), f"c_{s}_y") for s in superitems_in_layer}
    xsj, ysj = dict(), dict()
    for s in superitems_in_layer:
        for j in superitems_in_layer:
            if j != s:
                xsj[(s, j)] = slv.IntVar(0, 1, f"x_{s}_{j}")
                ysj[(s, j)] = slv.IntVar(0, 1, f"y_{s}_{j}")

    # Constraints
    # Redundant valid cuts that force the area of
    # a layer to fit within the area of a bin
    slv.Add(sum(ws[s] * ds[s] * zsl[s] for s in superitems_in_layer) <= W * D)

    # Define the height of layer l
    for s in superitems_in_layer:
        slv.Add(layer_height >= hs[s] * zsl[s])

    # Enforce at least one relative positioning relationship
    # between each pair of items in a layer
    for s in superitems_in_layer:
        for j in superitems_in_layer:
            if j > s:
                slv.Add(xsj[s, j] + xsj[j, s] + ysj[s, j] + ysj[j, s] >= zsl[s] + zsl[j] - 1)

    # Ensure that there is at most one spatial relationship
    # between items i and j along the width and depth dimensions
    for s in superitems_in_layer:
        for j in superitems_in_layer:
            if j > s:
                slv.Add(xsj[s, j] + xsj[j, s] <= 1)
                slv.Add(ysj[s, j] + ysj[j, s] <= 1)

    # Non-overlapping constraints
    for s in superitems_in_layer:
        for j in superitems_in_layer:
            if j != s:
                slv.Add(cix[s] + ws[s] <= cix[j] + W * (1 - xsj[s, j]))
                slv.Add(ciy[s] + ds[s] <= ciy[j] + D * (1 - ysj[s, j]))

    # Objective
    obj = layer_height - sum(
        duals[i] * fsi[s, i] * zsl[s] for i in items_in_layer for s in superitems_in_layer
    )
    slv.Minimize(obj)

    # Set a time limit
    if tlim is not None:
        slv.SetTimeLimit(1000 * tlim)

    # Solve
    status = slv.Solve()

    # Extract results
    sol = dict()
    if status in (slv.OPTIMAL, slv.FEASIBLE):
        for s in superitems_in_layer:
            sol[f"z_{s}_l"] = zsl[s].solution_value()
            sol[f"c_{s}_x"] = cix[s].solution_value()
            sol[f"c_{s}_y"] = ciy[s].solution_value()
        sol["objective"] = slv.Objective().Value()

    return sol, slv.WallTime() / 1000


def column_generation(
    layer_pool,
    pallet_dims,
    max_iter=20,
    max_stag_iters=20,
    tlim=None,
    use_maxrect=False,
):
    W, D, _ = pallet_dims
    fsi, _, _ = layer_pool.superitems_pool.get_fsi()
    zsl = layer_pool.get_zsl()
    ol = layer_pool.get_ol()
    ws, ds, hs = layer_pool.superitems_pool.get_superitems_dims()

    n_superitems, n_items = fsi.shape
    best_rmp_obj, num_stag_iters = float("inf"), 0
    for i in range(max_iter):
        print(f"Iteration {i + 1}/{max_iter}")

        # Reduced master problem
        print("Solving RMP...")
        rmp_sol, rmp_time, duals = main_problem(fsi, zsl, ol, tlim=tlim, relaxation=True)
        if rmp_sol is None:
            print("Unfeasible main problem")
            break
        print("Duals:", duals)
        print("RMP time:", rmp_time)

        # Keep best RMP objective value
        if rmp_sol["objective"] < best_rmp_obj:
            best_rmp_obj = rmp_sol["objective"]
            num_stag_iters = 0
        else:
            num_stag_iters += 1

        # Break if RMP objective does not improve
        if num_stag_iters == max_stag_iters:
            break

        # Check feasibility
        n_layers = zsl.shape[-1]
        print("Alpha:", [rmp_sol[f"alpha_{l}"] for l in range(n_layers)])
        if not all([rmp_sol[f"alpha_{l}"] in (0, 1) for l in range(n_layers)]):
            print("RMP: solution not feasible (at least one alpha value is not binary)")

        # Pricing sub-problem
        feasibility, placed = None, False
        while not placed:
            print("Solving SP (no placement)...")
            sp_np_sol, sp_np_time = pricing_problem_no_placement(
                fsi, ws, ds, hs, W, D, duals, feasibility=feasibility, tlim=tlim
            )
            print("SP no placement time:", sp_np_time)
            superitems_in_layer = [s for s in range(n_superitems) if sp_np_sol[f"z_{s}_l"] == 1]
            feasibility = len(superitems_in_layer)

            # Non-negative reduced cost
            print("Reduced cost:", sp_np_sol["objective"])
            if sp_np_sol["objective"] >= 0:
                print("Reached convergence :)")
                return layer_pool, best_rmp_obj
            if use_maxrect:
                placed, layer = utils.maxrects_single_layer(
                    layer_pool.superitems_pool,
                    W,
                    D,
                    superitems_in_layer=superitems_in_layer,
                )
            else:
                print(
                    "New layer: Num selected Items:",
                    len(superitems_in_layer),
                    "/",
                    n_superitems,
                )
                print("Solving SP (with placement)...")
                sp_p_sol, sp_p_time = pricing_problem_placement_v2(
                    superitems_in_layer, ws, ds, W, D, tlim=tlim
                )
                print("SP placement time:", sp_p_time)
                placed = "objective" in sp_p_sol

                if placed:
                    layer = build_layer_from_cp(
                        layer_pool.superitems_pool, superitems_in_layer, sp_p_sol
                    )
                else:
                    print("Unable to place select items, retrying")

            if placed:
                layer_pool.add(layer)
                zsl = layer_pool.get_zsl()
                ol = layer_pool.get_ol()
            else:
                print("FEASIBILITY: ", feasibility)

    return layer_pool, best_rmp_obj


def build_layer_from_cp(superitems_pool, superitems_in_layer, sp_p_sol):
    spool, coords = [], []
    for s in superitems_in_layer:
        spool += [superitems_pool[s]]
        coords += [utils.Coordinate(x=sp_p_sol[f"c_{s}_x"], y=sp_p_sol[f"c_{s}_y"])]
    spool = superitems.SuperitemPool(superitems=spool)
    return layers.Layer(
        spool.get_max_height(),
        spool,
        coords,
    )
