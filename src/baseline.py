from ortools.sat.python import cp_model


def baseline_model(fsi, ws, ds, hs, W, D, tlim=None):
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

    # Constraints
    # Define the height of layer l
    model.Add(ol[l] >= hs[s] * zsl[s, l] for s in range(n_superitems) for l in range(max_layers))

    # Redundant valid cuts that force the area of
    # a layer to fit within the area of a bin
    model.Add(
        sum(ws[s] * ds[s] * zsl[s, l] for s in range(n_superitems) for l in range(max_layers))
        <= W * D
    )

    # Enforce at least one relative positioning relationship
    # between each pair of items in a layer
    for s in range(n_superitems):
        for j in range(n_superitems):
            if j > s:
                model.Add(xsj[s, j] + xsj[j, s] + ysj[s, j] + ysj[j, s] >= 1)

    # Ensure that there is at most one spatial relationship
    # between items i and j along the width and depth dimensions
    for s in range(n_superitems):
        for j in range(n_superitems):
            if j > s:
                model.Add(xsj[s, j] + xsj[j, s] <= 1)
                model.Add(ysj[s, j] + ysj[j, s] <= 1)

    # Non-overlapping constraints
    for s in range(n_superitems):
        for j in range(n_superitems):
            if j != s:
                model.Add(cix[s] + ws[s] <= cix[j] + W * (1 - xsj[s, j]))
                model.Add(ciy[s] + ds[s] <= ciy[j] + D * (1 - ysj[s, j]))

    # Objective
    obj = sum(ol[l] for l in range(max_layers))
    model.Minimize(obj)

    # Set a time limit
    if tlim is not None:
        solver.SetTimeLimit(1000 * tlim)

    # Set solver parameters
    solver.parameters.num_search_workers = 4

    # Solve
    status = solver.Solve(model)

    # Extract results
    sol = dict()
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for l in range(max_layers):
            sol[f"o_{l}"] = solver.Value(ol[l])
            for s in range(n_superitems):
                sol[f"z_{s}_{l}"] = solver.Value(zsl[s, l])
            sol["objective"] = solver.ObjectiveValue()

    return sol, solver.WallTime() / 1000
