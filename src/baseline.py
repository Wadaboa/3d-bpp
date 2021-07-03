from ortools.sat.python import cp_model

from . import utils, superitems, layers


def baseline_model(fsi, ws, ds, hs, W, D, tlim=None, num_workers=4):
    # Model and solver declaration
    model = cp_model.CpModel()
    solver = cp_model.CpSolver()

    # Utility
    n_superitems, n_items = fsi.shape
    max_layers = n_items

    # Variables
    ol = {l: model.NewIntVar(0, max(hs), f"o_{l}") for l in range(max_layers)}
    zsl, cix, ciy, xsj, ysj = dict(), dict(), dict(), dict(), dict()
    for s in range(n_superitems):
        cix[s] = model.NewIntVar(0, int(W - ws[s]), f"c_{s}_x")
        ciy[s] = model.NewIntVar(0, int(D - ds[s]), f"c_{s}_y")
        for j in range(n_superitems):
            if j != s:
                xsj[s, j] = model.NewBoolVar(f"x_{s}_{j}")
                ysj[s, j] = model.NewBoolVar(f"y_{s}_{j}")
        for l in range(max_layers):
            zsl[s, l] = model.NewBoolVar(f"z_{s}_{l}")

    # Channeling variables
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
            sum(fsi[s, i] * zsl[s, l] for s in range(n_superitems) for l in range(max_layers)) == 1
        )

    # Define the height of layer l
    for l in range(max_layers):
        for s in range(n_superitems):
            model.Add(ol[l] >= hs[s] * zsl[s, l])

    # Redundant valid cuts that force the area of
    # a layer to fit within the area of a bin
    model.Add(
        sum(ws[s] * ds[s] * zsl[s, l] for l in range(max_layers) for s in range(n_superitems))
        <= W * D
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
                    model.Add(cix[s] + ws[s] <= cix[j] + W * (1 - xsj[s, j])).OnlyEnforceIf(
                        [same[s, j, l]]
                    )
                    model.Add(ciy[s] + ds[s] <= ciy[j] + D * (1 - ysj[s, j])).OnlyEnforceIf(
                        [same[s, j, l]]
                    )

    # Objective
    obj = sum(ol[l] for l in range(max_layers))
    model.Minimize(obj)

    # Search strategy
    indices_by_area = utils.argsort([ws[s] * ds[s] for s in range(n_superitems)], reverse=True)
    model.AddDecisionStrategy(
        [cix[s] for s in indices_by_area],
        cp_model.CHOOSE_LOWEST_MIN,
        cp_model.SELECT_MIN_VALUE,
    )

    # Set a time limit
    if tlim is not None:
        solver.SetTimeLimit(1000 * tlim)

    # Set solver parameters
    solver.parameters.num_search_workers = num_workers

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

    return sol, solver.WallTime() / 1000


def call_baseline_model(superitems_pool, pallet_dims, tlim=None, num_workers=4):
    W, D, _ = pallet_dims
    fsi, _, _ = superitems_pool.get_fsi()
    n_superitems, n_items = fsi.shape
    max_layers = n_items
    ws, ds, hs = superitems_pool.get_superitems_dims()
    sol, solve_time = baseline_model(fsi, ws, ds, hs, W, D, tlim=tlim, num_workers=num_workers)

    layer_pool = layers.LayerPool(superitems_pool)
    for l in range(max_layers):
        if sol[f"o_{l}"] > 0:
            superitems_in_layer = [s for s in range(n_superitems) if sol[f"z_{s}_{l}"] == 1]
            spool, coords = [], []
            for s in superitems_in_layer:
                spool += [superitems_pool[s]]
                coords += [utils.Coordinate(x=sol[f"c_{s}_x"], y=sol[f"c_{s}_y"])]
            spool = superitems.SuperitemPool(superitems=spool)
            layer = layers.Layer(
                spool.get_max_height(),
                spool,
                coords,
            )
            layer_pool.add(layer)

    return layer_pool
