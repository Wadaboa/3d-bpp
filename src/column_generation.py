import time
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger

from ortools.sat.python import cp_model
from ortools.linear_solver import pywraplp

import layers, superitems, utils, maxrects


# Define 'pywraplp' solver parameters
INTEGER_PARAMETERS = ["PRESOLVE", "LP_ALGORITHM", "INCREMENTALITY"]
DOUBLE_PARAMETERS = ["RELATIVE_MIP_GAP", "PRIMAL_TOLERANCE", "DUAL_TOLERANCE"]
PARAMETERS_MAPPING = {
    "RELATIVE_MIP_GAP": pywraplp.MPSolverParameters.RELATIVE_MIP_GAP,
    "PRIMAL_TOLERANCE": pywraplp.MPSolverParameters.PRIMAL_TOLERANCE,
    "DUAL_TOLERANCE": pywraplp.MPSolverParameters.PRIMAL_TOLERANCE,
    "PRESOLVE": pywraplp.MPSolverParameters.PRESOLVE,
    "LP_ALGORITHM": pywraplp.MPSolverParameters.LP_ALGORITHM,
    "INCREMENTALITY": pywraplp.MPSolverParameters.INCREMENTALITY,
}


def get_parameter_values(params):
    """
    Given a MPSolverParameters instance, return a dictionary with the values
    associated to the each parameter (only compatible with 'pywraplp' solvers)
    """
    return {
        "RELATIVE_MIP_GAP": params.GetDoubleParam(PARAMETERS_MAPPING["RELATIVE_MIP_GAP"]),
        "PRIMAL_TOLERANCE": params.GetDoubleParam(PARAMETERS_MAPPING["PRIMAL_TOLERANCE"]),
        "DUAL_TOLERANCE": params.GetDoubleParam(PARAMETERS_MAPPING["PRIMAL_TOLERANCE"]),
        "PRESOLVE": params.GetIntegerParam(PARAMETERS_MAPPING["PRESOLVE"]),
        "LP_ALGORITHM": params.GetIntegerParam(PARAMETERS_MAPPING["LP_ALGORITHM"]),
        "INCREMENTALITY": params.GetIntegerParam(PARAMETERS_MAPPING["INCREMENTALITY"]),
    }


def set_parameter_values(params, assignments):
    """
    Given a MPSolverParameters instance and a dictionary like the following
    {
        'LP_ALGORITHM': pywraplp.MPSolverParameters.PRIMAL,
        ...
    }
    set the parameters identified by the keys as the associated value in the dictionary
    """
    for k, v in assignments.items():
        if k in INTEGER_PARAMETERS:
            params.SetIntegerParam(PARAMETERS_MAPPING[k], v)
        elif k in DOUBLE_PARAMETERS:
            params.SetDoubleParam(PARAMETERS_MAPPING[k], v)
    return params


def superitems_duals(duals, superitems_pool):
    """
    Compute Superitems dual from the given Items duals and SuperitemsPool
    """
    fsi, _, _ = superitems_pool.get_fsi()
    return np.matmul(duals, fsi.T)


def main_problem(fsi, zsl, ol, tlim=None, relaxation=True):
    """
    Model with relaxed/not-relaxed version for computing the Reduced Master Problem based on
    Dantzig-Wolfe decomposition using Lagrangian multipliers to achieve better results, used to
    compute Items duals thanks to Linear Solver.
    Returns:
    - the objective value which tries to minimize the sum of alpha[l] * h[l] where l is each layer and h[l] is the height of Layer l,
    - alpha[l] represents a boolean/float variable based on the layer selection,
    - duals for each Item computed on the constraints
    """
    # Solver
    if relaxation:
        slv = pywraplp.Solver("RMP", pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)
    else:
        slv = pywraplp.Solver("MP", pywraplp.Solver.BOP_INTEGER_PROGRAMMING)
    slv.SetNumThreads(8)
    # slv.EnableOutput()

    logger.info("Starting RMP")

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
    status = slv.Solve()

    # Extract results
    sol, duals = None, None
    if status in (slv.OPTIMAL, slv.FEASIBLE):
        sol = {f"alpha_{l}": al[l].solution_value() for l in range(n_layers)}
        sol["objective"] = slv.Objective().Value()
        if relaxation:
            duals = np.array([c.DualValue() for c in constraints])

        logger.debug(
            f"RMP: solved with {slv.NumVariables()} variables, "
            f"{slv.NumConstraints()} constraints, "
            f"in {slv.iterations()} iterations"
        )
        # print(slv.ExportModelAsLpFormat(True))

    # Return results
    return sol, slv.WallTime() / 1000, duals


def pricing_problem_maxrect(pallet_dims, duals, superitems_pool):
    # TODO Generalize Pricing problem structure and returns such that we can have an objective value of the maxrect generated layer and be able to check for convergence
    logger.info("Starting SP MR variant")
    sduals = superitems_duals(duals, superitems_pool)
    return maxrects.maxrects_single_layer_online(superitems_pool, pallet_dims, sduals)


def pricing_problem_no_placement_mip(
    pallet_dims, duals, superitems_pool, feasibility=None, tlim=None
):
    ws, ds, hs = superitems_pool.get_superitems_dims()
    sduals = superitems_duals(duals, superitems_pool)

    # Solver
    slv = pywraplp.Solver("SP-NP", pywraplp.Solver.SCIP_MIXED_INTEGER_PROGRAMMING)
    # slv.SetNumThreads(8)
    # slv.EnableOutput()

    logger.info("Starting SP-NP")

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
    logger.info(f"SP-NP: max number of selected items (Feasibility)<= {feasibility}")
    f = slv.Constraint(1, feasibility, "feasibility")
    for s in range(n_superitems):
        f.SetCoefficient(zsl[s], 1)

    # Objective
    # Compute reward for greater number of selected superitems
    # ol - sum(zsl * (sduals + zero_reward))
    reward = 1 / (sduals.max() + n_superitems)
    zero_reward = np.where(duals == 0, reward, 0)
    logger.debug(f"SP-NP: {reward} reward about selecting superitems with zero dual")
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

    logger.debug(
        f"SP-NP: solved with {slv.NumVariables()} variables, "
        f"{slv.NumConstraints()} constraints, "
        f"in {slv.iterations()} iterations"
    )
    # print(slv.ExportModelAsLpFormat(False))

    return sol, slv.WallTime() / 1000


def pricing_problem_no_placement_cp(
    fsi, ws, ds, hs, pallet_dims, duals, feasibility=None, tlim=None
):
    # Model and Solver
    mdl = cp_model.CpModel()
    slv = cp_model.CpSolver()

    logger.info("Starting SP-NP CP variant")

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
        logger.info(f"SP-NP: setting number of selected items <= {feasibility}")
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

    logger.info("Starting SP-P CP variant")

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


def pricing_problem_placement_mip(superitems_in_layer, ws, ds, pallet_dims, tlim=None):

    # Solver
    slv = pywraplp.Solver("SP-P", pywraplp.Solver.SCIP_MIXED_INTEGER_PROGRAMMING)
    # slv.SetNumThreads(8)
    # slv.EnableOutput()

    logger.info("Starting SP-P")

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

    logger.debug(
        f"SP-P: solved with {slv.NumVariables()} variables, "
        f"{slv.NumConstraints()} constraints, "
        f"in {slv.iterations()} iterations"
    )
    # print(slv.ExportModelAsLpFormat(False))

    return sol, slv.WallTime() / 1000


def pricing_problem_placement_mr(superitems_in_layer, pallet_dims, superitems_pool):
    start = time.time()
    placed, layer = maxrects.maxrects_single_layer_offline(
        superitems_pool,
        pallet_dims,
        superitems_in_layer=superitems_in_layer,
    )
    duration = time.time() - start
    sol = None
    if placed:
        sol = {"objective": placed}
        sol["layer"] = layer
    return sol, duration


def column_generation(
    working_layer_pool,
    pallet_dims,
    max_iter=100,
    max_stag_iters=20,
    tlim=None,
    spp_mr=False,
    spp_cp=True,
    spp_mip=False,
    pricing_problem_maxrect=False,
    return_only_last=False,
):
    assert max_iter > 0, "Maximum number of iterations must be > 0"
    assert max_stag_iters > 0, "Maximum number of stagnation iteration must be > 0"
    assert (
        spp_mr + spp_cp + spp_mip
    ) == 1, "Only one pricing problem placement approach can be selected"

    logger.info("Starting CG")
    final_layer_pool = layers.LayerPool(working_layer_pool.superitems_pool, pallet_dims)
    already_added_layers = set()
    fsi, _, _ = working_layer_pool.superitems_pool.get_fsi()
    ws, ds, hs = working_layer_pool.superitems_pool.get_superitems_dims()

    n_superitems, _ = fsi.shape
    best_rmp_obj, num_stag_iters = float("inf"), 0
    # Starting CG iterations cycle
    for i in range(max_iter):
        zsl = working_layer_pool.get_zsl()
        ol = working_layer_pool.get_ol()
        logger.info(f"CG iteration{i+1}/{max_iter}")
        n_layers = zsl.shape[-1]

        # Reduced master problem (RMP)
        rmp_sol, rmp_time, duals = main_problem(fsi, zsl, ol, tlim=tlim, relaxation=True)
        if rmp_sol is None:
            logger.error("ERROR-----Unfeasible main problem--------")
            break

        alphas = [rmp_sol[f"alpha_{l}"] for l in range(n_layers)]
        logger.debug(
            f"RMP objective {rmp_sol['objective']}\n"
            f"Duals: {duals}\n"
            f"RMP time: {rmp_time}\n"
            f"Alphas: {alphas}"
        )

        if return_only_last:
            # TODO not the best approach, see if it's possible to find a better solution to keep only last selected layers
            final_layer_pool = layers.LayerPool(working_layer_pool.superitems_pool, pallet_dims)
            already_added_layers = set()

        # Check feasibility
        if not all(alphas[l] in (0, 1) for l in range(n_layers)):
            logger.debug("RMP solution not feasible (at least one alpha value is not binary)")
        for l, alpha in enumerate(alphas):
            # Add to final LayerPool only Layers with alpha > 0 which weren't already selected
            if alpha > 0 and l not in already_added_layers:
                final_layer_pool.add(working_layer_pool[l])
                already_added_layers.add(l)

        # Keep best RMP objective value
        if rmp_sol["objective"] < best_rmp_obj:
            best_rmp_obj = rmp_sol["objective"]
            num_stag_iters = 0
        else:
            num_stag_iters += 1

        # Break if RMP objective does not improve
        if num_stag_iters == max_stag_iters:
            logger.error("CG Stagnation exit")
            break

        # Pricing sub-problem (SP)
        if pricing_problem_maxrect:
            # Compute all Pricing sub-problem no-placement and placement heuristically using maxrect
            layer = pricing_problem_maxrect(pallet_dims, duals, working_layer_pool.superitems_pool)
            working_layer_pool.add(layer)
        else:
            # Divide computation of no-placement and placement
            feasibility = len(working_layer_pool.superitems_pool)
            placed = False
            while not placed and feasibility > 0:
                # Pricing sub-problem no-placement (SP-NP)
                # Select a subgroup of Superitems with greater duals and
                # compute reduced cost of new Layer
                sp_np_sol, sp_np_time = pricing_problem_no_placement_mip(
                    pallet_dims,
                    duals,
                    working_layer_pool.superitems_pool,
                    feasibility=feasibility,
                    tlim=tlim,
                )
                if sp_np_sol is None:
                    logger.error("ERROR-----Unfeasible SP_no_placement problem--------")
                    break

                superitems_in_layer = [s for s in range(n_superitems) if sp_np_sol[f"z_{s}_l"] == 1]

                logger.debug(
                    f"SP-NP objective (Reduced cost): {sp_np_sol['objective']}, "
                    f"SP-NP time: {sp_np_time}"
                    f"Num selected Superitems: {len(superitems_in_layer)}"
                )

                # Non-negative reduced cost
                if sp_np_sol["objective"] >= 0:
                    logger.success("Reached convergence")
                    return final_layer_pool, best_rmp_obj

                sduals = superitems_duals(duals, working_layer_pool.superitems_pool)
                if spp_mr:
                    """
                    SP-P MR Variant
                    """
                    sp_p_sol, sp_p_time = pricing_problem_placement_mr()
                elif spp_cp:
                    """
                    SP-P CP Variant
                    """
                    sp_p_sol, sp_p_time = pricing_problem_placement_cp(
                        superitems_in_layer, sduals, ws, ds, pallet_dims, tlim=tlim
                    )
                elif spp_mip:
                    """
                    SP-P MIP Variant
                    """
                    sp_p_sol, sp_p_time = pricing_problem_placement_mip(
                        superitems_in_layer, ws, ds, pallet_dims, tlim=tlim
                    )

                placed = sp_p_sol is not None
                logger.debug(f"SP-P objective (Placed all): {placed}, " f"SP-P time: {sp_p_time}")

                if placed:
                    logger.info("SP-P generated a new Layer")
                    if spp_mr:
                        layer = sp_p_sol["layer"]
                    else:
                        layer = utils.build_layer_from_model_output(
                            working_layer_pool.superitems_pool,
                            superitems_in_layer,
                            sp_p_sol,
                            pallet_dims,
                        )

                if placed:
                    working_layer_pool.add(layer)
                else:
                    feasibility = len(superitems_in_layer) - 1

    return final_layer_pool, best_rmp_obj
