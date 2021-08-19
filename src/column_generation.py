import time
import numpy as np
from loguru import logger

from ortools.sat.python import cp_model
from ortools.linear_solver import pywraplp

import layers, utils, maxrects


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


def superitems_duals(superitems_pool, duals):
    """
    Compute Superitems dual from the given Items duals and SuperitemsPool
    """
    fsi, _, _ = superitems_pool.get_fsi()
    return np.matmul(duals, fsi.T)


def master_problem(layer_pool, tlim=None, relaxation=True, enable_output=False):
    """
    Solve the master problem, either in its full version (MP)
    or in its relaxed version (RMP). Returns the following:
    - Objective value: minimization of sum(alpha[l] * h[l]), with h heights and l layer
    - Alpha values: alpha[l] represents layer selection
    - [RMP] Duals: one dual for each item
    """
    logger.info("RMP defining variables and constraints")

    # Solver
    if relaxation:
        slv = pywraplp.Solver("RMP", pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)
    else:
        slv = pywraplp.Solver("MP", pywraplp.Solver.BOP_INTEGER_PROGRAMMING)

    # Enable verbose output from solver
    if enable_output:
        slv.EnableOutput()

    # Utility
    fsi, _, _ = layer_pool.superitems_pool.get_fsi()
    zsl = layer_pool.get_zsl()
    ol = layer_pool.get_ol()
    infinity = slv.infinity()
    n_layers = len(layer_pool)
    n_items = fsi.shape[-1]

    # Variables
    if relaxation:
        al = [slv.NumVar(0, infinity, f"alpha_{l}") for l in range(n_layers)]
    else:
        al = [slv.BoolVar(f"alpha_{l}") for l in range(n_layers)]

    # Constraints
    constraints = []
    coefficients = np.matmul(fsi.T, zsl)

    # Select each item at least once
    # sum(al[l] * zsl[s, l] * fsi[s, i])
    for i in range(n_items):
        c = slv.Constraint(1, infinity, f"c_{i}")
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
    logger.debug(f"RMP variables: {slv.NumVariables()}")
    logger.debug(f"RMP constraints: {slv.NumConstraints()}")
    status = slv.Solve()
    logger.debug(f"RMP iterations: {slv.iterations()}")

    # Extract results
    duals, alphas = None, None
    objective = float("inf")
    if status in (slv.OPTIMAL, slv.FEASIBLE):
        logger.info(f"RMP solved")

        # Extract alpha values
        alphas = [al[l].solution_value() for l in range(n_layers)]
        logger.debug(f"RMP alphas: {alphas}")
        if not all(alphas[l] in (0, 1) for l in range(n_layers)):
            logger.debug("RMP solution not feasible (at least one alpha value is not binary)")

        # Extract objective value
        objective = slv.Objective().Value()
        logger.debug(f"RMP objective: {objective}")

        # Extract duals
        if relaxation:
            duals = np.array([c.DualValue() for c in constraints])
            logger.debug(f"RMP duals: {duals}")
    else:
        logger.warning("RMP unfeasible")

    logger.debug(f"RMP time: {slv.WallTime() / 1000}")
    return objective, alphas, duals


def pricing_problem_maxrects(superitems_pool, pallet_dims, duals):
    """
    Solve the whole pricing subproblem heuristically, using maxrects
    to place superitems by biggest duals first
    """
    start = time.time()
    logger.info("SP-MR starting computation")
    sduals = superitems_duals(superitems_pool, duals)
    layer = maxrects.maxrects_single_layer_online(superitems_pool, pallet_dims, sduals)
    duration = time.time() - start

    if layer is not None:
        logger.debug("SP-MR solved")
    else:
        logger.warning("SP-MR unfeasible")

    logger.debug(f"SP-MR time: {duration}")
    return layer


def pricing_problem_no_placement_mip(
    superitems_pool, pallet_dims, duals, feasibility=None, tlim=None, enable_output=False
):
    """
    Solve the pricing subproblem no-placement using a MIP approach
    """
    logger.info("SP-NP-MIP defining variables and constraints")

    # Solver
    slv = pywraplp.Solver("SP-NP-MIP", pywraplp.Solver.SCIP_MIXED_INTEGER_PROGRAMMING)

    # Enable verbose output from solver
    if enable_output:
        slv.EnableOutput()

    # Utility
    ws, ds, hs = superitems_pool.get_superitems_dims()
    sduals = superitems_duals(superitems_pool, duals)
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
    logger.info(f"SP-NP-MIP feasibility: max number of selected items <= {feasibility}")
    f = slv.Constraint(1, feasibility, "feasibility")
    for s in range(n_superitems):
        f.SetCoefficient(zsl[s], 1)

    # Compute reward for greater number of selected superitems
    reward = 1 / (sduals.max() + n_superitems)
    zero_reward = np.where(sduals == 0, reward, 0)
    logger.debug(f"SP-NP-MIP zero duals reward: {reward}")

    # Objective
    # ol - sum(zsl * (sduals + zero_reward))
    obj = slv.Objective()
    obj.SetCoefficient(ol, 1)
    for s in range(n_superitems):
        obj.SetCoefficient(zsl[s], -sduals[s] - zero_reward[s])
    obj.SetMinimization()

    # Set a time limit in milliseconds
    if tlim is not None:
        slv.SetTimeLimit(1000 * tlim)

    # Solve
    logger.debug(f"SP-NP-MIP variables: {slv.NumVariables()}")
    logger.debug(f"SP-NP-MIP constraints: {slv.NumConstraints()}")
    status = slv.Solve()
    logger.debug(f"SP-NP-MIP iterations: {slv.iterations()}")

    # Extract results
    objective = float("inf")
    superitems_in_layer = None
    if status in (slv.OPTIMAL, slv.FEASIBLE):
        logger.info(f"SP-NP-MIP solved")

        # Extract objective value
        objective = slv.Objective().Value()
        logger.debug(f"SP-NP-MIP objective: {objective}")

        # Extract selected superitems
        superitems_in_layer = [s for s in range(n_superitems) if zsl[s].solution_value() == 1]
        logger.debug(f"SP-NP-MIP selected {len(superitems_in_layer)}/{n_superitems} superitems")

        logger.debug(f"SP-NP-MIP computed layer height: {ol.solution_value()}")
    else:
        logger.warning("SP-NP-MIP unfeasible")

    logger.debug(f"SP-NP-MIP time: {slv.WallTime() / 1000}")
    return objective, superitems_in_layer


def pricing_problem_no_placement_cp(
    superitems_pool, pallet_dims, duals, feasibility=None, tlim=None, enable_output=False
):
    """
    Solve the pricing subproblem no-placement using a CP approach
    """
    logger.info("SP-NP-CP defining variables and constraints")

    # Model and solver
    mdl = cp_model.CpModel()
    slv = cp_model.CpSolver()

    # Utility
    fsi, _, _ = superitems_pool.get_fsi()
    ws, ds, hs = superitems_pool.get_superitems_dims()
    n_superitems, n_items = fsi.shape

    # Variables
    ol = mdl.NewIntVar(0, max(hs), f"o_l")
    zsl = [mdl.NewBoolVar(f"z_{s}_l") for s in range(n_superitems)]

    # Constraints
    # Redundant valid cuts that force the area of
    # a layer to fit within the area of a bin
    mdl.Add(
        cp_model.LinearExpr.Sum(ws[s] * ds[s] * zsl[s] for s in range(n_superitems))
        <= pallet_dims.area
    )

    # Define the height of layer l
    for s in range(n_superitems):
        mdl.Add(ol >= hs[s] * zsl[s])

    # Enforce feasible placement
    if feasibility is not None:
        logger.info(f"SP-NP-MIP feasibility: max number of selected items <= {feasibility}")
        mdl.Add(cp_model.LinearExpr.Sum(zsl[s] for s in range(n_superitems)) <= feasibility)

    # No item repetition constraint
    for i in range(n_items):
        mdl.Add(cp_model.LinearExpr.Sum([fsi[s, i] * zsl[s] for s in range(n_superitems)]) <= 1)

    # Objective
    obj = ol - cp_model.LinearExpr.Sum(
        int(np.ceil(duals[i])) * fsi[s, i] * zsl[s]
        for i in range(n_items)
        for s in range(n_superitems)
    )
    mdl.Minimize(obj)

    # Search strategy
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

    # Solve
    slv.parameters.num_search_workers = 4
    slv.parameters.log_search_progress = enable_output
    slv.parameters.search_branching = cp_model.FIXED_SEARCH
    status = slv.Solve(mdl)

    # Extract results
    objective = float("inf")
    superitems_in_layer = None
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        logger.info(f"SP-NP-CP solved")

        # Extract objective value
        objective = slv.ObjectiveValue()
        logger.debug(f"SP-NP-CP objective: {objective}")

        # Extract selected superitems
        superitems_in_layer = [s for s in range(n_superitems) if slv.Value(zsl[s]) == 1]
        logger.debug(f"SP-NP-CP selected {len(superitems_in_layer)}/{n_superitems} superitems")

        logger.debug(f"SP-NP-CP computed layer height: {slv.Value(ol)}")
    else:
        logger.warning("SP-NP-CP unfeasible")

    logger.debug(f"SP-NP-CP time: {slv.WallTime()}")
    return objective, superitems_in_layer


def pricing_problem_placement_cp(
    superitems_pool, superitems_in_layer, pallet_dims, duals, tlim=None, enable_output=False
):
    """
    Solve the pricing subproblem placement using a CP approach
    """
    logger.info("SP-P-CP defining variables and constraints")

    # Utility
    ws, ds, _ = superitems_pool.get_superitems_dims()
    sduals = superitems_duals(superitems_pool, duals)

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

    # Search strategy
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

    # Solve
    slv.parameters.num_search_workers = 4
    slv.parameters.log_search_progress = enable_output
    slv.parameters.search_branching = cp_model.FIXED_SEARCH
    status = slv.Solve(mdl)

    # Extract results
    layer = None
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        logger.info(f"SP-P-CP solved")

        # Extract coordinates
        sol = dict()
        for s in superitems_in_layer:
            sol[f"c_{s}_x"] = slv.Value(cblx[s])
            sol[f"c_{s}_y"] = slv.Value(cbly[s])

        # Build layer
        layer = utils.build_layer_from_model_output(
            superitems_pool,
            superitems_in_layer,
            sol,
            pallet_dims,
        )
    else:
        logger.warning("SP-P-CP unfeasible")

    logger.debug(f"SP-P-CP time: {slv.WallTime()}")
    return layer


def pricing_problem_placement_mip(
    superitems_pool, superitems_in_layer, pallet_dims, tlim=None, enable_output=False
):
    """
    Solve the subproblem placement using a MIP formulation
    """
    logger.info("SP-P-MIP defining variables and constraints")

    # Store superitems dimensions
    ws, ds, _ = superitems_pool.get_superitems_dims()

    # Solver
    slv = pywraplp.Solver("SP-P-MIP", pywraplp.Solver.SCIP_MIXED_INTEGER_PROGRAMMING)

    # Enable verbose output from solver
    if enable_output:
        slv.EnableOutput()

    # Variables
    cix = {s: slv.IntVar(0, pallet_dims.width - ws[s], f"c_{s}_x") for s in superitems_in_layer}
    ciy = {s: slv.IntVar(0, pallet_dims.depth - ds[s], f"c_{s}_y") for s in superitems_in_layer}
    xsj, ysj = dict(), dict()
    for s in superitems_in_layer:
        for j in superitems_in_layer:
            if j != s:
                xsj[(s, j)] = slv.BoolVar(f"x_{s}_{j}")
                ysj[(s, j)] = slv.BoolVar(f"y_{s}_{j}")

    # Constraints
    # Enforce at least one relative positioning relationship
    # between each pair of items in a layer
    # xsj[s, j] + xsj[j, s] + ysj[s, j] + ysj[j, s] >= 1
    precedence_constraints = []
    for s in superitems_in_layer:
        for j in superitems_in_layer:
            if j > s:
                c = slv.Constraint(1, 2, f"p_{s}_{j}")
                c.SetCoefficient(xsj[s, j], 1)
                c.SetCoefficient(xsj[j, s], 1)
                c.SetCoefficient(ysj[s, j], 1)
                c.SetCoefficient(ysj[j, s], 1)
                precedence_constraints += [c]

    # Ensure that there is at most one spatial relationship
    # between items i and j along the width and depth dimensions
    # xsj[s,j] + xsj[j,s] <= 1
    # ysj[s,j] + ysj[j,s] <= 1
    precedence_x_constraints = []
    precedence_y_constraints = []
    for s in superitems_in_layer:
        for j in superitems_in_layer:
            if j > s:
                c = slv.Constraint(0, 1, f"px_{s}_{j}")
                c.SetCoefficient(xsj[s, j], 1)
                c.SetCoefficient(xsj[j, s], 1)
                precedence_x_constraints += [c]
                c = slv.Constraint(0, 1, f"py_{s}_{j}")
                c.SetCoefficient(ysj[s, j], 1)
                c.SetCoefficient(ysj[j, s], 1)
                precedence_y_constraints += [c]

    # Non-overlapping constraints
    # cix[s] + ws[s] <= cix[j] + pallet_dims.width * (1 - xsj[s, j])
    # ciy[s] + ds[s] <= ciy[j] + pallet_dims.depth * (1 - ysj[s, j])
    non_overlapping_x_constraints = []
    non_overlapping_y_constraints = []
    for s in superitems_in_layer:
        for j in superitems_in_layer:
            if j != s:
                # csy[s] - cjy[s] + pallet_dims.depth * ysj[s, j] <= pallet_dims.depth - ds[s]
                # csy[s] - cjy[s] + pallet_dims.depth * ysj[s, j] >= 0
                c = slv.Constraint(
                    -pallet_dims.width + ws[j], pallet_dims.width - ws[s], f"ox_{s}_{j}"
                )
                c.SetCoefficient(cix[s], 1)
                c.SetCoefficient(cix[j], -1)
                c.SetCoefficient(xsj[s, j], pallet_dims.width)
                non_overlapping_x_constraints += [c]
                c = slv.Constraint(
                    -pallet_dims.depth + ds[j], pallet_dims.depth - ds[s], f"oy_{s}_{j}"
                )
                c.SetCoefficient(ciy[s], 1)
                c.SetCoefficient(ciy[j], -1)
                c.SetCoefficient(ysj[s, j], pallet_dims.depth)
                non_overlapping_y_constraints += [c]

    # Set a time limit
    if tlim is not None:
        slv.SetTimeLimit(1000 * tlim)

    # Solve
    logger.debug(f"SP-P-MIP variables: {slv.NumVariables()}")
    logger.debug(f"SP-P-MIP constraints: {slv.NumConstraints()}")
    status = slv.Solve()
    logger.debug(f"SP-P-MIP iterations: {slv.iterations()}")

    # Extract results
    layer = None
    if status in (slv.OPTIMAL, slv.FEASIBLE):
        logger.info(f"SP-P-MIP solved")

        # Extract coordinates
        sol = dict()
        for s in superitems_in_layer:
            sol[f"c_{s}_x"] = cix[s].solution_value()
            sol[f"c_{s}_y"] = ciy[s].solution_value()

        # Build layer
        layer = utils.build_layer_from_model_output(
            superitems_pool,
            superitems_in_layer,
            sol,
            pallet_dims,
        )
    else:
        logger.warning("SP-P-MIP unfeasible")

    logger.debug(f"SP-P-MIP time: {slv.WallTime() / 1000}")
    return layer


def pricing_problem_placement_mr(superitems_pool, superitems_in_layer, pallet_dims):
    """
    Solve the pricing subproblem (placement) using maxrects: try to place
    the entire set of superitems selected by the no-placement subproblem
    (if not placeable, return None)
    """
    start = time.time()
    layer = maxrects.maxrects_single_layer_offline(
        superitems_pool,
        pallet_dims,
        superitems_in_layer=superitems_in_layer,
    )
    duration = time.time() - start
    logger.debug(f"SP-P-MR time: {duration}")
    return layer


def column_generation(
    layer_pool,
    pallet_dims,
    max_iter=100,
    max_stag_iters=20,
    tlim=None,
    sp_mr=False,
    sp_np_type="mip",
    sp_p_type="cp",
    return_only_last=False,
    enable_solver_output=False,
):
    """
    Main column generation procedure
    """
    assert max_iter > 0, "Maximum number of iterations must be > 0"
    assert max_stag_iters > 0, "Maximum number of stagnation iteration must be > 0"
    assert sp_np_type in ("cp", "mip"), "Unsupported subproblem no-placement procedure"
    assert sp_p_type in (
        "cp",
        "mr",
        "mip",
    ), "Unsupported subproblem placement procedure"

    logger.info("Starting CG")
    final_layer_pool = layers.LayerPool(layer_pool.superitems_pool, pallet_dims)
    best_rmp_obj, num_stag_iters = float("inf"), 0

    # Starting CG iterations cycle
    for i in range(max_iter):
        logger.info(f"CG iteration {i + 1}/{max_iter}")

        # Store number of layers
        n_layers = len(layer_pool)

        # Reduced master problem (RMP)
        rmp_obj, alphas, duals = master_problem(
            layer_pool, tlim=tlim, relaxation=True, enable_output=enable_solver_output
        )

        # Check RMP objective
        if rmp_obj is None:
            break

        # Return only those layers in the last CG iteration
        if return_only_last:
            final_layer_pool = layers.LayerPool(layer_pool.superitems_pool, pallet_dims)

        # Add to final layer pool only layers with alpha > 0 which weren't already selected
        final_layer_pool.extend(
            layer_pool.subset([i for i, _ in enumerate(layer_pool) if alphas[i] > 0])
        )

        # Keep best RMP objective value
        if rmp_obj < best_rmp_obj:
            best_rmp_obj = rmp_obj
            num_stag_iters = 0
        else:
            num_stag_iters += 1
            logger.debug(f"CG stagnation {num_stag_iters}/{max_stag_iters}")

        # Break if RMP objective does not improve
        if num_stag_iters == max_stag_iters:
            logger.error("CG exiting for stagnation")
            break

        # Use maxrects to solve the entire pricing subproblem
        if sp_mr:
            layer = pricing_problem_maxrects(layer_pool.superitems_pool, pallet_dims, duals)
            if layer is not None:
                layer_pool.add(layer)
        # Use no-placement/placement strategy for the pricing subproblem
        else:
            feasibility = len(layer_pool.superitems_pool)
            while feasibility > 0:
                # Subproblem no-placement (SP-NP)
                if sp_np_type == "mip":
                    sp_np_obj, superitems_in_layer = pricing_problem_no_placement_mip(
                        layer_pool.superitems_pool,
                        pallet_dims,
                        duals,
                        feasibility=feasibility,
                        tlim=tlim,
                        enable_output=enable_solver_output,
                    )
                elif sp_np_type == "cp":
                    sp_np_obj, superitems_in_layer = pricing_problem_no_placement_cp(
                        layer_pool.superitems_pool,
                        pallet_dims,
                        duals,
                        feasibility=feasibility,
                        tlim=tlim,
                        enable_output=enable_solver_output,
                    )

                # Check SP-NP solution
                if sp_np_obj is None:
                    break

                # Non-negative reduced cost
                if sp_np_obj >= 0:
                    logger.success("CG reached convergence")
                    return final_layer_pool, best_rmp_obj

                # Subproblem placement (SP-P)
                if sp_p_type == "mr":
                    layer = pricing_problem_placement_mr(
                        layer_pool.superitems_pool, superitems_in_layer, pallet_dims
                    )
                elif sp_p_type == "cp":
                    layer = pricing_problem_placement_cp(
                        layer_pool.superitems_pool,
                        superitems_in_layer,
                        pallet_dims,
                        duals,
                        tlim=tlim,
                        enable_output=enable_solver_output,
                    )
                elif sp_p_type == "mip":
                    layer = pricing_problem_placement_mip(
                        layer_pool.superitems_pool,
                        superitems_in_layer,
                        pallet_dims,
                        tlim=tlim,
                        enable_output=enable_solver_output,
                    )

                # Add the generated layer to the pool
                if layer is not None:
                    layer_pool.add(layer)
                    break
                # Decrement the maximum number of placeable items to ensure feasibility
                else:
                    feasibility = len(superitems_in_layer) - 1

    return final_layer_pool, best_rmp_obj
