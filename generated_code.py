from gurobipy import Model, GRB, quicksum

def Industrial_example(P, S, C, K, A, c, scenarios, M):
    model = Model("SupplyChainDesign")
    
    # Indices and sets
    P = list(P)  # Potential facilities
    S = list(S)  # Supplier facilities
    C = list(C)  # Customer nodes
    K = list(K)  # Products
    A = [tuple(e) for e in A]  # Feasible arcs
    W = range(len(scenarios))  # Scenarios

    # Decision variables
    y = model.addVars(P, vtype=GRB.BINARY, name="y")  # Facility opening decisions
    x = model.addVars(A, K, W, vtype=GRB.CONTINUOUS, name="x")  # Shipment flows
    u = model.addVars(S, K, W, vtype=GRB.CONTINUOUS, name="u")  # Injected supplies
    z = model.addVars(C, K, W, vtype=GRB.CONTINUOUS, name="z")  # Shortage amounts

    # Objective function
    model.setObjective(
        quicksum(c[i] * y[i] for i in P) +
        quicksum(scenarios[w]['probability'] * (
            quicksum(scenarios[w]['q'].get((i, j, k), 0) * x[i, j, k, w] for i, j, k in A if (i, j) in A and k in K) +
            quicksum(scenarios[w]['h'][j][k] * z[j, k, w] for j in C for k in K)
        ) for w in W),
        GRB.MINIMIZE
    )

    # Constraints
    # Flow conservation at non-supplier facilities
    model.addConstrs(
        (quicksum(x[i, j, k, w] for i, j in A if j == p) == quicksum(x[j, i, k, w] for j, i in A if i == p)
         for p in P if p not in S for k in K for w in W), "flowConservationNonSuppliers")

    # Flow conservation at supplier facilities
    model.addConstrs(
        (quicksum(x[i, j, k, w] for i, j in A if j == s) + u[s, k, w] == quicksum(x[j, i, k, w] for j, i in A if i == s)
         for s in S for k in K for w in W), "flowConservationSuppliers")

    # Supply limits at suppliers
    model.addConstrs(
        (u[s, k, w] <= scenarios[w]['s'][s][k] * y[s] for s in S for k in K for w in W), "supplyLimits")

    # Demand satisfaction at customers
    model.addConstrs(
        (quicksum(x[i, j, k, w] for i, j in A if j == c) + z[c, k, w] >= scenarios[w]['d'][c][k]
         for c in C for k in K for w in W), "demandSatisfaction")

    # Linking constraints (no flow from/to unopened facilities)
    model.addConstrs(
        (x[i, j, k, w] <= M * y[i] for i, j in A for k in K for w in W), "linkingOutflow")
    model.addConstrs(
        (x[i, j, k, w] <= M * y[j] for i, j in A for k in K for w in W), "linkingInflow")

    # Optimize the model
    model.optimize()

    # Return the optimal objective value if the model has an optimal solution
    if model.status == GRB.OPTIMAL:
        return model.objVal
    else:
        return None

# Example usage
P = ['Fac1', 'Fac2', 'Fac3']
S = ['Fac1', 'Fac2']
C = ['Cust1', 'Cust2']
K = ['Prod1', 'Prod2']
A = [('Fac1', 'Fac2'), ('Fac2', 'Cust1'), ('Fac1', 'Cust2')]
c = {'Fac1': 1000, 'Fac2': 1500, 'Fac3': 1200}
scenarios = [
    {'probability': 0.5, 'q': {('Fac1', 'Fac2', 'Prod1'): 2, ('Fac2', 'Cust1', 'Prod1'): 3, ('Fac1', 'Cust2', 'Prod1'): 4},
     'h': {'Cust1': {'Prod1': 10}, 'Cust2': {'Prod1': 15}}, 's': {'Fac1': {'Prod1': 100}, 'Fac2': {'Prod1': 150}},
     'd': {'Cust1': {'Prod1': 80}, 'Cust2': {'Prod1': 120}}},
    {'probability': 0.5, 'q': {('Fac1', 'Fac2', 'Prod2'): 3, ('Fac2', 'Cust1', 'Prod2'): 2, ('Fac1', 'Cust2', 'Prod2'): 5},
     'h': {'Cust1': {'Prod2': 12}, 'Cust2': {'Prod2': 18}}, 's': {'Fac1': {'Prod2': 90}, 'Fac2': {'Prod2': 160}},
     'd': {'Cust1': {'Prod2': 85}, 'Cust2': {'Prod2': 115}}}
]
M = 10000  # Large constant for big-M method

result = Industrial_example(P, S, C, K, A, c, scenarios, M)
print("Optimal expected total cost:", result)