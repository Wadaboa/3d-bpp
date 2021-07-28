from ortools.sat.python import cp_model
from loguru import logger

import utils, layers


def baseline_model(fsi, ws, ds, hs, pallet_dims, tlim=None, num_workers=4):
    """
    The baseline model directly assigns superitems to layers and positions
    them by taking into account overlapment and layer height minimization.
    It reproduces model [SPPSI] of the referenced paper (beware that it
    might be very slow and we advice using it only for orders under 30 items).

    Samir Elhedhli, Fatma Gzara, Burak Yildiz,
    "Three-Dimensional Bin Packing and Mixed-Case Palletization",
    INFORMS Journal on Optimization, 2019.
    """
    # Model and solver declaration
    model = cp_model.CpModel()
    solver = cp_model.CpSolver()

    # Utility
    n_superitems, n_items = fsi.shape
    max_layers = n_items

    # Variables
    # Layer heights variables
    ol = {l: model.NewIntVar(0, max(hs), f"o_{l}") for l in range(max_layers)}
    zsl, cix, ciy, xsj, ysj = dict(), dict(), dict(), dict(), dict()
    for s in range(n_superitems):
        # Coordinate variables
        cix[s] = model.NewIntVar(0, int(pallet_dims.width - ws[s]), f"c_{s}_x")
        ciy[s] = model.NewIntVar(0, int(pallet_dims.depth - ds[s]), f"c_{s}_y")

        # Precedence variables
        for j in range(n_superitems):
            if j != s:
                xsj[s, j] = model.NewBoolVar(f"x_{s}_{j}")
                ysj[s, j] = model.NewBoolVar(f"y_{s}_{j}")

        # Superitems to layer assignment variables
        for l in range(max_layers):
            zsl[s, l] = model.NewBoolVar(f"z_{s}_{l}")

    # Channeling variables
    # same[s, j, l] = 1 iff superitems s and j are both in layer l
    same = dict()
    for l in range(max_layers):
        for s in range(n_superitems):
            for j in range(n_superitems):
                if j != s:
                    same[s, j, l] = model.NewBoolVar(f"s_{s}_{j}_{l}")
                    model.Add(same[s, j, l] == 1).OnlyEnforceIf([zsl[s, l], zsl[j, l]])
                    model.Add(same[s, j, l] == 0).OnlyEnforceIf([zsl[s, l].Not(), zsl[j, l]])
                    model.Add(same[s, j, l] == 0).OnlyEnforceIf([zsl[s, l], zsl[j, l].Not()])
                    model.Add(same[s, j, l] == 0).OnlyEnforceIf([zsl[s, l].Not(), zsl[j, l].Not()])

    # Constraints
    # Ensure that every item is included in exactly one layer
    for i in range(n_items):
        model.Add(
            cp_model.LinearExpr.Sum(
                fsi[s, i] * zsl[s, l] for s in range(n_superitems) for l in range(max_layers)
            )
            == 1
        )

    # Define the height of layer l
    for l in range(max_layers):
        for s in range(n_superitems):
            model.Add(ol[l] >= hs[s] * zsl[s, l])

    # Redundant valid cuts that force the area of
    # a layer to fit within the area of a bin
    model.Add(
        cp_model.LinearExpr.Sum(
            ws[s] * ds[s] * zsl[s, l] for l in range(max_layers) for s in range(n_superitems)
        )
        <= pallet_dims.area
    )

    # Enforce at least one relative positioning relationship
    # between each pair of items in a layer
    for l in range(max_layers):
        for s in range(n_superitems):
            for j in range(n_superitems):
                if j > s:
                    model.Add(xsj[s, j] + xsj[j, s] + ysj[s, j] + ysj[j, s] >= 1).OnlyEnforceIf(
                        [same[s, j, l]]
                    )

    # Ensure that there is at most one spatial relationship
    # between items i and j along the width and depth dimensions
    for l in range(max_layers):
        for s in range(n_superitems):
            for j in range(n_superitems):
                if j > s:
                    model.Add(xsj[s, j] + xsj[j, s] <= 1).OnlyEnforceIf([same[s, j, l]])
                    model.Add(ysj[s, j] + ysj[j, s] <= 1).OnlyEnforceIf([same[s, j, l]])

    # Non-overlapping constraints
    for l in range(max_layers):
        for s in range(n_superitems):
            for j in range(n_superitems):
                if j != s:
                    model.Add(
                        cix[s] + ws[s] <= cix[j] + pallet_dims.width * (1 - xsj[s, j])
                    ).OnlyEnforceIf([same[s, j, l]])
                    model.Add(
                        ciy[s] + ds[s] <= ciy[j] + pallet_dims.depth * (1 - ysj[s, j])
                    ).OnlyEnforceIf([same[s, j, l]])

    # Minimize the sum of layer heights
    obj = cp_model.LinearExpr.Sum(ol[l] for l in range(max_layers))
    model.Minimize(obj)

    # Search by biggest area first
    indices_by_area = utils.argsort([ws[s] * ds[s] for s in range(n_superitems)], reverse=True)
    model.AddDecisionStrategy(
        [cix[s] for s in indices_by_area],
        cp_model.CHOOSE_LOWEST_MIN,
        cp_model.SELECT_MIN_VALUE,
    )

    # Set a time limit
    if tlim is not None:
        solver.parameters.max_time_in_seconds = tlim

    # Set solver parameters
    solver.parameters.num_search_workers = num_workers
    solver.parameters.log_search_progress = True
    solver.parameters.search_branching = cp_model.FIXED_SEARCH

    # Solve
    status = solver.Solve(model)

    # Extract results
    sol = dict()
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for l in range(max_layers):
            sol[f"o_{l}"] = solver.Value(ol[l])
            for s in range(n_superitems):
                sol[f"z_{s}_{l}"] = solver.Value(zsl[s, l])
        for s in range(n_superitems):
            sol[f"c_{s}_x"] = solver.Value(cix[s])
            sol[f"c_{s}_y"] = solver.Value(ciy[s])
        sol["objective"] = solver.ObjectiveValue()

    # Return solution and solving time
    return sol, solver.WallTime()


def baseline(superitems_pool, pallet_dims, tlim=None, num_workers=4):
    """
    Call the baseline model with the given parameters and return
    a layer pool
    """
    # Initialize model variables
    fsi, _, _ = superitems_pool.get_fsi()
    ws, ds, hs = superitems_pool.get_superitems_dims()
    n_superitems, n_items = fsi.shape
    max_layers = n_items

    # Call the baseline model
    logger.info("Solving baseline model")
    sol, solve_time = baseline_model(
        fsi, ws, ds, hs, pallet_dims, tlim=tlim, num_workers=num_workers
    )
    logger.info(f"Solved baseline model in {solve_time:.2f} seconds")

    # Build the layer pool from the model's solution
    layer_pool = layers.LayerPool(superitems_pool, pallet_dims)
    for l in range(max_layers):
        if sol[f"o_{l}"] > 0:
            superitems_in_layer = [s for s in range(n_superitems) if sol[f"z_{s}_{l}"] == 1]
            layer = utils.build_layer_from_model_output(
                superitems_pool, superitems_in_layer, sol, pallet_dims
            )
            layer_pool.add(layer)

    return layer_pool
