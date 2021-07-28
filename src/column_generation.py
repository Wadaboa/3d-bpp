import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ortools.sat.python import cp_model
from ortools.linear_solver import pywraplp
from tqdm import tqdm

import layers, superitems, utils, maxrects


STATUS_STRING = {
    cp_model.OPTIMAL: "optimal",
    cp_model.FEASIBLE: "feasible",
    cp_model.INFEASIBLE: "infeasible",
    cp_model.MODEL_INVALID: "invalid",
    cp_model.UNKNOWN: "unknown",
}


def print_parameters(parameters):
    values = dict()
    values["RELATIVE_MIP_GAP"] = (
        parameters.GetDoubleParam(pywraplp.MPSolverParameters.RELATIVE_MIP_GAP),
    )
    values["PRIMAL_TOLERANCE"] = (
        parameters.GetDoubleParam(pywraplp.MPSolverParameters.PRIMAL_TOLERANCE),
    )
    values["DUAL_TOLERANCE"] = (
        parameters.GetDoubleParam(pywraplp.MPSolverParameters.PRIMAL_TOLERANCE),
    )
    values["PRESOLVE"] = (parameters.GetIntegerParam(pywraplp.MPSolverParameters.PRESOLVE),)
    values["LP_ALGORITHM"] = (parameters.GetIntegerParam(pywraplp.MPSolverParameters.LP_ALGORITHM),)
    values["INCREMENTALITY"] = (
        parameters.GetIntegerParam(pywraplp.MPSolverParameters.INCREMENTALITY),
    )
    return values


def main_problem(fsi, zsl, ol, tlim=None, relaxation=True):
    # Parameters
    parameters = pywraplp.MPSolverParameters()
    parameters.SetIntegerParam(
        pywraplp.MPSolverParameters.LP_ALGORITHM, pywraplp.MPSolverParameters.PRIMAL
    )

    # Solver
    if relaxation:
        slv = pywraplp.Solver("RMP_relaxed_problem", pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)
    else:
        slv = pywraplp.Solver("RMP_problem", pywraplp.Solver.BOP_INTEGER_PROGRAMMING)
    slv.SetNumThreads(8)
    # slv.EnableOutput()

    # Utility
    infinity = slv.infinity()
    n_superitems, n_layers = zsl.shape
    n_items = fsi.shape[-1]

    # Variables
    if relaxation:
        al = [slv.NumVar(0, infinity, f"alpha_{l}") for l in range(n_layers)]
        # al = [slv.NumVar(0, 1, f"alpha_{l}") for l in range(n_layers)]  # TODO Check if correct assumption
    else:
        al = [slv.BoolVar(f"alpha_{l}") for l in range(n_layers)]

    coefficients = np.matmul(fsi.T, zsl)
    constraints = []
    # Constraints
    # sum(al[l] * zsl[s, l] * fsi[s, i])
    for i in range(n_items):
        c = slv.Constraint(1, infinity, f"c_{i}")
        # c = slv.Constraint(1, n_layers, f"c_{i}") TODO Check if correct assumption
        for l in range(n_layers):
            if coefficients[i, l] > 0:
                c.SetCoefficient(al[l], float(coefficients[i, l]))
        constraints += [c]

    # Objective
    obj = slv.Objective()
    for l, h in enumerate(ol):
        obj.SetCoefficient(al[l], float(h))
    obj.SetMinimization()

    # Set a time limit in milliseconds
    if tlim is not None:
        slv.SetTimeLimit(1000 * tlim)

    # Solve
    status = slv.Solve(parameters)

    # Extract results
    sol, duals = None, None
    if status in (slv.OPTIMAL, slv.FEASIBLE):
        sol = {f"alpha_{l}": al[l].solution_value() for l in range(n_layers)}
        sol["objective"] = slv.Objective().Value()
        if relaxation:
            duals = np.array([c.DualValue() for c in constraints])

        print("RMP var ", slv.NumVariables())
        print("RMP constraints ", slv.NumConstraints())
        print("RMP iterations ", slv.iterations())
        # print(slv.ExportModelAsLpFormat(True))

    # Return results
    return sol, slv.WallTime() / 1000, duals


def superitems_duals(duals, superitems_pool):
    fsi, _, _ = superitems_pool.get_fsi()
    return np.matmul(duals, fsi.T)


def pricing_problem_maxrect(pallet_dims, duals, superitems_pool):
    sduals = superitems_duals(duals, superitems_pool)
    return maxrects.maxrects_single_layer_online(superitems_pool, pallet_dims, sduals)


def pricing_problem_no_placement(pallet_dims, duals, superitems_pool, feasibility=None, tlim=None):
    ws, ds, hs = superitems_pool.get_superitems_dims()
    sduals = superitems_duals(duals, superitems_pool)

    # Solver
    slv = pywraplp.Solver("SP_no_placement_problem", pywraplp.Solver.SCIP_MIXED_INTEGER_PROGRAMMING)
    # slv.SetNumThreads(8)
    # slv.EnableOutput()

    # Utility
    n_superitems = len(superitems_pool)

    # Variables
    ol = slv.IntVar(0, max(hs), f"o_l")
    zsl = [slv.BoolVar(f"z_{s}_l") for s in range(n_superitems)]

    # Constraints
    # Redundant valid cuts that force the area of
    # a layer to fit within the area of a bin
    # ws * ds * zsl <= pallet_dims.area
    area = slv.Constraint(0, pallet_dims.area, "area")
    for s in range(n_superitems):
        area.SetCoefficient(zsl[s], ws[s] * ds[s])

    # Define layer height
    # ol >= zsl * hs
    height_constraints = []
    for s in range(n_superitems):
        hc = slv.Constraint(0, max(hs), f"hc_{s}")
        hc.SetCoefficient(ol, 1)
        hc.SetCoefficient(zsl[s], hs[s])
        height_constraints += [hc]

    # Enforce feasible placement
    # sum(zsl) <= feasibility
    print("Feasibility constraint: num selected <=", feasibility)
    f = slv.Constraint(1, feasibility, "feasibility")
    for s in range(n_superitems):
        f.SetCoefficient(zsl[s], 1)

    # Objective
    # Compute reward for greater number of selected superitems
    # ol - sum(zsl * (sduals + zero_reward))
    reward = 1 / (sduals.max() + n_superitems)
    zero_reward = np.where(duals == 0, reward, 0)
    print("Reward about selecting superitems with duals == 0", reward)
    obj = slv.Objective()
    obj.SetCoefficient(ol, 1)
    for s in range(n_superitems):
        obj.SetCoefficient(zsl[s], -sduals[s] - zero_reward[s])
    obj.SetMinimization()

    # Set a time limit in milliseconds
    if tlim is not None:
        slv.SetTimeLimit(1000 * tlim)

    # Solve
    status = slv.Solve()

    # Extract results
    sol = None
    if status in (slv.OPTIMAL, slv.FEASIBLE):
        sol = {f"o_l": ol.solution_value()}
        for s in range(n_superitems):
            sol[f"z_{s}_l"] = zsl[s].solution_value()
        sol["objective"] = slv.Objective().Value()

    print("SP_no_placement var ", slv.NumVariables())
    print("SP_no_placement constraints ", slv.NumConstraints())
    print("SP_no_placement iterations ", slv.iterations())
    # print(slv.ExportModelAsLpFormat(False))

    return sol, slv.WallTime() / 1000


def pricing_problem_no_placement_cp(
    fsi, ws, ds, hs, pallet_dims, duals, feasibility=None, tlim=None
):
    # Model and Solver
    mdl = cp_model.CpModel()
    slv = cp_model.CpSolver()

    # Utility
    n_superitems, n_items = fsi.shape

    # Variables
    ol = mdl.NewIntVar(0, max(hs), f"o_l")
    zsl = [mdl.NewBoolVar(f"z_{s}_l") for s in range(n_superitems)]

    # Constraints
    # Redundant valid cuts that force the area of
    # a layer to fit within the area of a bin
    mdl.Add(sum(ws[s] * ds[s] * zsl[s] for s in range(n_superitems)) <= pallet_dims.area)
    for s in range(n_superitems):
        # Define the height of layer l
        mdl.Add(ol >= hs[s] * zsl[s])

    # Enforce feasible placement
    if feasibility is not None:
        print("Adding feasibility constraint: num selected <=", feasibility)
        mdl.Add(sum(zsl[s] for s in range(n_superitems)) <= feasibility)

    # No item repetition constraint
    for i in range(n_items):
        mdl.Add(sum([fsi[s, i] * zsl[s] for s in range(n_superitems)]) <= 1)

    # Objective
    obj = ol - sum(
        int(np.ceil(duals[i])) * fsi[s, i] * zsl[s]
        for i in range(n_items)
        for s in range(n_superitems)
    )
    mdl.Minimize(obj)
    duals_sort_index = utils.argsort(
        [sum([fsi[s, i] * duals[i] for i in range(n_items)]) for s in range(n_superitems)]
    )
    mdl.AddDecisionStrategy([ol], cp_model.CHOOSE_FIRST, cp_model.SELECT_MIN_VALUE)
    mdl.AddDecisionStrategy(
        [zsl[s] for s in duals_sort_index],
        cp_model.CHOOSE_FIRST,
        cp_model.SELECT_MAX_VALUE,
    )

    # Set a time limit in seconds
    if tlim is not None:
        slv.parameters.max_time_in_seconds = tlim

    slv.parameters.num_search_workers = 4
    # Solve
    status = slv.Solve(mdl)

    # Extract results
    sol = None
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        sol = {f"o_l": slv.Value(ol)}
        for s in range(n_superitems):
            sol[f"z_{s}_l"] = slv.Value(zsl[s])
        sol["objective"] = slv.ObjectiveValue()

    return sol, slv.WallTime() / 1000


def pricing_problem_placement_cp(superitems_in_layer, sduals, ws, ds, pallet_dims, tlim=None):
    # Model and Solver
    mdl = cp_model.CpModel()
    slv = cp_model.CpSolver()

    # Variables
    cblx = {
        s: mdl.NewIntVar(0, pallet_dims.width - ws[s], f"c_bl_{s}_x") for s in superitems_in_layer
    }
    cbly = {
        s: mdl.NewIntVar(0, pallet_dims.depth - ds[s], f"c_bl_{s}_y") for s in superitems_in_layer
    }

    ctrx = {s: mdl.NewIntVar(ws[s], pallet_dims.width, f"c_tr_{s}_x") for s in superitems_in_layer}
    ctry = {s: mdl.NewIntVar(ds[s], pallet_dims.width, f"c_tr_{s}_y") for s in superitems_in_layer}

    xint = [
        mdl.NewIntervalVar(cblx[s], mdl.NewConstant(ws[s]), ctrx[s], f"xint_{s}")
        for s in superitems_in_layer
    ]
    yint = [
        mdl.NewIntervalVar(cbly[s], mdl.NewConstant(ds[s]), ctry[s], f"yint_{s}")
        for s in superitems_in_layer
    ]

    # Constraints
    mdl.AddNoOverlap2D(xint, yint)

    mdl.AddCumulative(
        xint, [mdl.NewConstant(ds[s]) for s in superitems_in_layer], pallet_dims.depth
    )
    mdl.AddCumulative(
        yint, [mdl.NewConstant(ws[s]) for s in superitems_in_layer], pallet_dims.width
    )

    # Symmetry Breaking
    areas = [ws[s] * ds[s] for s in superitems_in_layer]
    area_ind = utils.argsort(areas, reverse=True)
    biggest_ind = superitems_in_layer[area_ind[0]]
    second_ind = superitems_in_layer[area_ind[1]]
    mdl.Add(cblx[biggest_ind] <= mdl.NewConstant(pallet_dims.width // 2))
    mdl.Add(cbly[biggest_ind] <= mdl.NewConstant(pallet_dims.depth // 2))

    mdl.Add(cblx[biggest_ind] <= cblx[second_ind])
    mdl.Add(cbly[biggest_ind] <= cbly[second_ind])

    # Search Strategy
    indexes = utils.argsort([sduals[s] for s in superitems_in_layer], reverse=True)

    mdl.AddDecisionStrategy(
        [xint[i] for i in indexes], cp_model.CHOOSE_FIRST, cp_model.SELECT_MIN_VALUE
    )
    mdl.AddDecisionStrategy(
        [yint[i] for i in indexes], cp_model.CHOOSE_FIRST, cp_model.SELECT_MIN_VALUE
    )

    # Set a time limit in seconds
    if tlim is not None:
        slv.parameters.max_time_in_seconds = tlim

    slv.parameters.num_search_workers = 8
    slv.parameters.log_search_progress = True
    slv.parameters.search_branching = cp_model.FIXED_SEARCH
    # Solve
    status = slv.Solve(mdl)

    # Extract results
    sol = None
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        sol = {"objective": slv.ObjectiveValue()}
        for s in superitems_in_layer:
            sol[f"c_{s}_x"] = slv.Value(cblx[s])
            sol[f"c_{s}_y"] = slv.Value(cbly[s])

    return sol, slv.WallTime()


def pricing_problem_placement(superitems_in_layer, ws, ds, pallet_dims, tlim=None):

    # Solver
    slv = pywraplp.Solver("SP_placement_problem", pywraplp.Solver.SCIP_MIXED_INTEGER_PROGRAMMING)
    # slv.SetNumThreads(8)
    slv.EnableOutput()

    infinity = slv.infinity()

    # Variables
    cix = {s: slv.IntVar(0, pallet_dims.width - ws[s], f"c_{s}_x") for s in superitems_in_layer}
    ciy = {s: slv.IntVar(0, pallet_dims.depth - ds[s], f"c_{s}_y") for s in superitems_in_layer}
    xsj, ysj, = (
        dict(),
        dict(),
    )
    for s in superitems_in_layer:
        for j in superitems_in_layer:
            if j != s:
                xsj[(s, j)] = slv.BoolVar(f"x_{s}_{j}")
                ysj[(s, j)] = slv.BoolVar(f"y_{s}_{j}")

    # Constraints
    # Enforce at least one relative positioning relationship
    # between each pair of items in a layer
    # xsj[s, j] + xsj[j, s] + ysj[s, j] + ysj[j, s] >= 1
    positioning_constraints = []
    for s in superitems_in_layer:
        for j in superitems_in_layer:
            if j > s:
                # slv.Add(xsj[s, j] + xsj[j, s] + ysj[s, j] + ysj[j, s] >= 1)
                posc = slv.Constraint(1, 2, f"posc_{s}_{j}")
                posc.SetCoefficient(xsj[s, j], 1)
                posc.SetCoefficient(xsj[j, s], 1)
                posc.SetCoefficient(ysj[s, j], 1)
                posc.SetCoefficient(ysj[j, s], 1)
                positioning_constraints += [posc]

    # Ensure that there is at most one spatial relationship
    # between items i and j along the width and depth dimensions
    # xsj[s,j] + xsj[j,s] <= 1
    # ysj[s,j] + ysj[j,s] <= 1
    axis_x_constraints = []
    axis_y_constraints = []
    for s in superitems_in_layer:
        for j in superitems_in_layer:
            if j > s:
                axisxc = slv.Constraint(0, 1, f"axis_xc_{s}_{j}")
                axisxc.SetCoefficient(xsj[s, j], 1)
                axisxc.SetCoefficient(xsj[j, s], 1)
                axis_x_constraints += [axisxc]
                axisyc = slv.Constraint(0, 1, f"axis_yc_{s}_{j}")
                axisyc.SetCoefficient(ysj[s, j], 1)
                axisyc.SetCoefficient(ysj[j, s], 1)
                axis_y_constraints += [axisyc]

    # Non-overlapping constraints
    # cix[s] + ws[s] <= cix[j] + pallet_dims.width * (1 - xsj[s, j])
    # ciy[s] + ds[s] <= ciy[j] + pallet_dims.depth * (1 - ysj[s, j])
    non_overlapping_x_constraints = []
    non_overlapping_y_constraints = []
    for s in superitems_in_layer:
        for j in superitems_in_layer:
            if j != s:
                # slv.Add(cix[s] + ws[s] <= cix[j] + pallet_dims.width * (1 - xsj[s, j]))
                # slv.Add(ciy[s] + ds[s] <= ciy[j] + pallet_dims.depth * (1 - ysj[s, j]))
                # csy - cjy + pallet_dims.depth * ysj[s, j] <= pallet_dims.depth - ds[s]
                # csy - cjy + pallet_dims.depth * ysj[s, j] >= 0
                nov_cx = slv.Constraint(
                    -pallet_dims.width + ws[j], pallet_dims.width - ws[s], f"nov_cx_{s}_{j}"
                )
                nov_cx.SetCoefficient(cix[s], 1)
                nov_cx.SetCoefficient(cix[j], -1)
                nov_cx.SetCoefficient(xsj[s, j], pallet_dims.width)
                non_overlapping_x_constraints += [nov_cx]
                nov_cy = slv.Constraint(
                    -pallet_dims.depth + ds[j], pallet_dims.depth - ds[s], f"nov_cy_{s}_{j}"
                )
                nov_cy.SetCoefficient(ciy[s], 1)
                nov_cy.SetCoefficient(ciy[j], -1)
                nov_cy.SetCoefficient(ysj[s, j], pallet_dims.depth)
                non_overlapping_y_constraints += [nov_cy]

    # Set a time limit
    if tlim is not None:
        slv.SetTimeLimit(1000 * tlim)

    # Solve
    status = slv.Solve()

    # Extract results
    sol = None
    if status in (slv.OPTIMAL, slv.FEASIBLE):
        sol = {"objective": slv.Objective().Value()}
        for s in superitems_in_layer:
            sol[f"c_{s}_x"] = cix[s].solution_value()
            sol[f"c_{s}_y"] = ciy[s].solution_value()

    print("SP_placement var ", slv.NumVariables())
    print("SP_placement constraints ", slv.NumConstraints())
    print("SP_placement iterations ", slv.iterations())
    # print(slv.ExportModelAsLpFormat(False))

    return sol, slv.WallTime() / 1000


def column_generation(
    layer_pool,
    pallet_dims,
    max_iter=20,
    max_stag_iters=20,
    tlim=None,
    use_maxrect=False,
    only_maxrect=False,
    return_only_last=False,
):
    final_layer_pool = layers.LayerPool(layer_pool.superitems_pool, pallet_dims)
    fsi, _, _ = layer_pool.superitems_pool.get_fsi()
    ws, ds, hs = layer_pool.superitems_pool.get_superitems_dims()

    n_superitems, n_items = fsi.shape
    best_rmp_obj, num_stag_iters = float("inf"), 0
    for i in range(max_iter):
        zsl = layer_pool.get_zsl()
        ol = layer_pool.get_ol()
        print(f"Iteration {i + 1}/{max_iter}")
        n_layers = zsl.shape[-1]

        # Reduced master problem
        print("Solving RMP...")
        rmp_sol, rmp_time, duals = main_problem(fsi, zsl, ol, tlim=tlim, relaxation=True)
        if rmp_sol is None:
            print("ERROR-----Unfeasible main problem--------")
            break

        alphas = [rmp_sol[f"alpha_{l}"] for l in range(n_layers)]
        print("RMP objective:", rmp_sol["objective"])
        print("Duals:", duals)
        print("RMP time:", rmp_time)
        print("Alphas:", alphas)
        if return_only_last:
            final_layer_pool = layers.LayerPool(layer_pool.superitems_pool, pallet_dims)
        # Check feasibility
        if not all(alphas[l] in (0, 1) for l in range(n_layers)):
            print("RMP: solution not feasible (at least one alpha value is not binary)")
        for l, alpha in enumerate(alphas):
            if alpha > 0:
                final_layer_pool.add(layer_pool[l])

        # Keep best RMP objective value
        if rmp_sol["objective"] < best_rmp_obj:
            best_rmp_obj = rmp_sol["objective"]
            num_stag_iters = 0
        else:
            num_stag_iters += 1

        # Break if RMP objective does not improve
        if num_stag_iters == max_stag_iters:
            print("Stagnation exit :(")
            break

        if only_maxrect:
            print("Solving Maxrect (higher duals first)...")
            layer = pricing_problem_maxrect(pallet_dims, duals, layer_pool.superitems_pool)
            layer_pool.add(layer)
        else:
            # Pricing sub-problem
            feasibility, placed = len(layer_pool.superitems_pool), False
            while not placed and feasibility > 0:
                print("Solving SP (no placement)...")
                sp_np_sol, sp_np_time = pricing_problem_no_placement(
                    pallet_dims,
                    duals,
                    layer_pool.superitems_pool,
                    feasibility=feasibility,
                    tlim=tlim,
                )
                if sp_np_sol is None:
                    print("ERROR-----Unfeasible SP_no_placement problem--------")
                    break
                print("SP no placement time:", sp_np_time)
                superitems_in_layer = [s for s in range(n_superitems) if sp_np_sol[f"z_{s}_l"] == 1]

                # Non-negative reduced cost
                print("Reduced cost:", sp_np_sol["objective"])
                if sp_np_sol["objective"] >= 0:
                    print("Reached convergence :)")
                    return layer_pool, best_rmp_obj
                if use_maxrect:
                    placed, layer = maxrects.maxrects_single_layer(
                        layer_pool.superitems_pool,
                        pallet_dims,
                        superitems_in_layer=superitems_in_layer,
                    )
                else:
                    print("Solving SP (with placement)...")
                    sduals = superitems_duals(duals, layer_pool.superitems_pool)
                    """
                    CP Version
                    """
                    sp_p_sol, sp_p_time = pricing_problem_placement_cp(
                        superitems_in_layer, sduals, ws, ds, pallet_dims, tlim=tlim
                    )
                    """
                    MIP Version

                    sp_p_sol, sp_p_time = pricing_problem_placement(
                        superitems_in_layer, ws, ds, pallet_dims, tlim=tlim
                    )
                    """

                    print("SP placement time:", sp_p_time)
                    placed = sp_p_sol is not None

                    if placed:
                        print(
                            "New layer: Num selected Items:",
                            len(superitems_in_layer),
                            "/",
                            n_superitems,
                        )
                        layer = utils.build_layer_from_model_output(
                            layer_pool.superitems_pool, superitems_in_layer, sp_p_sol, pallet_dims
                        )
                        layer.plot()
                        plt.show()

                if placed:
                    layer_pool.add(layer)
                else:
                    feasibility = len(superitems_in_layer) - 1
                    print("FEASIBILITY: ", feasibility)

    return final_layer_pool, best_rmp_obj


def build_layer_from_cp(superitems_pool, superitems_in_layer, sp_p_sol, pallet_dims):
    spool, coords = [], []
    for s in superitems_in_layer:
        spool += [superitems_pool[s]]
        coords += [utils.Coordinate(x=sp_p_sol[f"c_{s}_x"], y=sp_p_sol[f"c_{s}_y"])]
    spool = superitems.SuperitemPool(superitems=spool)
    return layers.Layer(spool, coords, pallet_dims)
