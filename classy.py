# stabling_geometry.py
"""
Stabling Geometry Optimizer for Kochi Metro Rail Limited (KMRL)

Objective:
- Assign trainsets to depot bays with minimal shunting & optimal morning turnout.
- Takes into account train IDs, bay capacities, and operational constraints.

Dependencies:
    pip install ortools
"""

from ortools.linear_solver import pywraplp

class StablingGeometryOptimizer:
    def __init__(self, trains, bays, constraints=None):
        """
        trains: list of dicts [{id: "T1", priority: 1, length: 4}, ...]
        bays: list of dicts [{id: "B1", capacity: 4}, ...]
        constraints: additional rules if needed (dict)
        """
        self.trains = trains
        self.bays = bays
        self.constraints = constraints or {}
        self.solver = pywraplp.Solver.CreateSolver("SCIP")

    def solve(self):
        # Create binary decision variables: x[train][bay] = 1 if assigned
        x = {}
        for t_idx, train in enumerate(self.trains):
            for b_idx, bay in enumerate(self.bays):
                x[t_idx, b_idx] = self.solver.IntVar(0, 1, f"x_{train['id']}_{bay['id']}")

        # Each train assigned to exactly one bay
        for t_idx, _ in enumerate(self.trains):
            self.solver.Add(sum(x[t_idx, b_idx] for b_idx in range(len(self.bays))) == 1)

        # Bay capacity constraints
        for b_idx, bay in enumerate(self.bays):
            self.solver.Add(
                sum(x[t_idx, b_idx] * self.trains[t_idx]["length"] for t_idx in range(len(self.trains)))
                <= bay["capacity"]
            )

        # Objective: minimize "shunting cost" + maximize priority placement
        # Example cost: bay index (closer to 0 is easier for turnout)
        objective = self.solver.Objective()
        for t_idx, train in enumerate(self.trains):
            for b_idx, bay in enumerate(self.bays):
                cost = b_idx * 10 - train["priority"] * 5  # lower index better, higher priority better
                objective.SetCoefficient(x[t_idx, b_idx], cost)

        objective.SetMinimization()

        # Solve
        status = self.solver.Solve()
        if status == pywraplp.Solver.OPTIMAL:
            assignment = []
            for t_idx, train in enumerate(self.trains):
                for b_idx, bay in enumerate(self.bays):
                    if x[t_idx, b_idx].solution_value() == 1:
                        assignment.append({
                            "train_id": train["id"],
                            "bay_id": bay["id"],
                            "priority": train["priority"]
                        })
            return assignment
        else:
            return None


if __name__ == "__main__":
    # Example usage
    trains = [
        {"id": "T1", "priority": 3, "length": 4},
        {"id": "T2", "priority": 1, "length": 4},
        {"id": "T3", "priority": 2, "length": 4},
    ]

    bays = [
        {"id": "B1", "capacity": 8},
        {"id": "B2", "capacity": 4},
        {"id": "B3", "capacity": 6},
    ]

    optimizer = StablingGeometryOptimizer(trains, bays)
    solution = optimizer.solve()

    if solution:
        print("\nOptimal Train → Bay Assignment:\n")
        for s in solution:
            print(f"🚆 Train {s['train_id']} → 🅱️ Bay {s['bay_id']} (Priority {s['priority']})")
    else:
        print("❌ No feasible assignment found.")
