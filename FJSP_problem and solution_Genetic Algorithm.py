import os
import pandas as pd
import json
import numpy as np
from deap import base, creator, tools, algorithms
from tqdm import tqdm


# Parameters for generating random FJSP problems
num_problems = 10000
max_locations = 5
max_departments = 7
max_target_groups = 4
max_teams = 5
max_travel_time = 30


def generate_random_fjsp():
    num_locations = np.random.randint(2, max_locations + 1)
    num_departments = np.random.randint(3, max_departments + 1)
    num_target_groups = np.random.randint(2, max_target_groups + 1)

    departments = [f"Dept_{i + 1}" for i in range(num_departments)]
    target_groups = [f"Group_{i + 1}" for i in range(num_target_groups)]

    travel_times = {
        dept: np.random.randint(1, max_travel_time + 1, size=(max_teams, num_locations)).tolist()
        for dept in departments
    }

    return {
        "num_locations": num_locations,
        "num_departments": num_departments,
        "num_target_groups": num_target_groups,
        "departments": departments,
        "target_groups": target_groups,
        "travel_times": travel_times,
    }

# Check if problem files exist
problem_json_path = "fjsp_problemsGenetic Algorithm.json"
problem_excel_path = "fjsp_problemsGenetic Algorithm.xlsx"

if os.path.exists(problem_json_path) and os.path.exists(problem_excel_path):
    with open(problem_json_path, "r") as file:
        fjsp_problems = json.load(file)
    print("Existing problem files found. Skipping problem generation.")
else:
    fjsp_problems = [generate_random_fjsp() for _ in range(num_problems)]

    # Save problems as JSON
    with open(problem_json_path, "w") as json_file:
        json.dump(fjsp_problems, json_file, indent=4)

    # Save problems as Excel
    problem_excel_data = []
    for idx, problem in enumerate(fjsp_problems):
        for dept, times in problem["travel_times"].items():
            for team_idx, location_times in enumerate(times):
                problem_excel_data.append({
                    "Problem_ID": idx + 1,
                    "Num_Locations": problem["num_locations"],
                    "Num_Departments": problem["num_departments"],
                    "Num_Target_Groups": problem["num_target_groups"],
                    "Department": dept,
                    "Team_ID": team_idx + 1,
                    "Location_Times": location_times,
                    "Target_Groups": ", ".join(problem["target_groups"])
                })

    pd.DataFrame(problem_excel_data).to_excel(problem_excel_path, index=False, engine='openpyxl')

#  solve  FJSP problem using multi-objective Genetic Algorithm
def solve_fjsp_with_ga(problem):

    num_locationsMulti = problem["num_locations"]
    num_departmentsMulti = problem["num_departments"]
    travel_timesMulti = problem["travel_times"]
    departmentsMulti = problem["departments"]


    num_teamsMulti = len(travel_timesMulti[departmentsMulti[0]])
    num_variablesMulti = num_locationsMulti * num_departmentsMulti


    if "FitnessMulti" in creator.__dict__:
        del creator.FitnessMulti
    if "IndividualMulti" in creator.__dict__:
        del creator.IndividualMulti

    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
    creator.create("IndividualMulti", list, fitness=creator.FitnessMulti)
    toolbox = base.Toolbox()

    toolbox.register("attr_teamMulti", np.random.randint, 0, num_teamsMulti)

    toolbox.register("individualMulti1", tools.initRepeat, creator.IndividualMulti, toolbox.attr_teamMulti, n=num_variablesMulti)
    toolbox.register("populationMulti", tools.initRepeat, list, toolbox.individualMulti1)


    #  multi-objective
    def evaluate(individual):
        arrival_timesMulti = []
        for locMulti in range(num_locationsMulti):
            loc_timesMulti = []
            for dept_idxMulti, deptMulti in enumerate(departmentsMulti):
                team_idxMulti = individual[locMulti * num_departmentsMulti + dept_idxMulti]
                loc_timesMulti.append(travel_timesMulti[deptMulti][team_idxMulti][locMulti])
            arrival_timesMulti.append(loc_timesMulti)

        makespan = max(max(timesMulti) for timesMulti in arrival_timesMulti)
        sync_cost = sum(max(timesMulti) - min(timesMulti) for timesMulti in arrival_timesMulti)
        return makespan, sync_cost



    # Register genetic operators
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=num_teamsMulti - 1, indpb=0.2)
    toolbox.register("select", tools.selNSGA2)  # Multi-objective selection
    toolbox.register("evaluate", evaluate)

    # Run Genetic Algorithm
    populationMulti = toolbox.populationMulti(n=100)
    algorithms.eaMuPlusLambda(
        populationMulti, toolbox, mu=100, lambda_=200, cxpb=0.7, mutpb=0.2, ngen=100, verbose=False
    )

    # Extract best solution (Pareto front)
    best_individualMulti = tools.selBest(populationMulti, k=1)[0]
    makespan, sync_cost = toolbox.evaluate(best_individualMulti)


    assignmentsMulti = {}

    for loc in range(num_locationsMulti):
        assignmentsMulti[f"Location {loc + 1}"] = {}
        for dept_idx, dept in enumerate(departmentsMulti):
            team_idxMulti = best_individualMulti[loc * num_departmentsMulti + dept_idx]
            assignmentsMulti[f"Location {loc + 1}"][dept] = {
                "TeamMulti": f"Team_{team_idxMulti + 1}",
                "Travel TimeMulti": travel_timesMulti[dept][team_idxMulti][loc]
            }

    return {
        "AssignmentsMulti": assignmentsMulti,
        "MakespanMulti": makespan,
        "Sync CostMulti": sync_cost
    }



# single-objective
def solve_makespan_only(problem):
    num_locations = problem["num_locations"]
    num_departments = problem["num_departments"]
    travel_times = problem["travel_times"]
    departments = problem["departments"]

    num_teams = len(travel_times[departments[0]])
    num_variables = num_locations * num_departments


    if "FitnessMin" in creator.__dict__:
        del creator.FitnessMin
    if "IndividualMin" in creator.__dict__:
        del creator.IndividualMin

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("IndividualMin", list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()

    toolbox.register("attr_team", np.random.randint, 0, num_teams)
    toolbox.register("individual", tools.initRepeat, creator.IndividualMin, toolbox.attr_team, n=num_variables)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Evaluation function for single-objective
    def evaluate_makespan(individual):
        arrival_times = []
        for loc in range(num_locations):
            loc_times = []
            for dept_idx, dept in enumerate(departments):
                team_idx = individual[loc * num_departments + dept_idx]
                loc_times.append(travel_times[dept][team_idx][loc])
            arrival_times.append(max(loc_times))
        makespan = max(arrival_times)
        return makespan,

    # Register genetic operators
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=num_teams - 1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate_makespan)

    # Run Genetic Algorithm
    population = toolbox.population(n=100)
    result_population, _ = algorithms.eaSimple(
        population, toolbox, cxpb=0.7, mutpb=0.2, ngen=100, verbose=False
    )

    # Extract best solution
    best_individual = tools.selBest(result_population, k=1)[0]
    best_makespan = toolbox.evaluate(best_individual)[0]

    assignments = {}
    for loc in range(num_locations):
        assignments[f"Location {loc + 1}"] = {}
        for dept_idx, dept in enumerate(departments):
            team_idx = best_individual[loc * num_departments + dept_idx]
            assignments[f"Location {loc + 1}"][dept] = {
                "Team": f"Team_{team_idx + 1}",
                "Travel Time": travel_times[dept][team_idx][loc]
            }

    return {
        "AssignmentsSingle": assignments,
        "Makespan_onlyConsiderMakespan": best_makespan
    }


# Solve all problems
solutions = []

for idx, problem in tqdm(enumerate(fjsp_problems), total=num_problems, desc="Solving FJSP problems"):
    # Solve using both multi-objective and makespan-only optimization
    solution_full = solve_fjsp_with_ga(problem)
    solution_makespan = solve_makespan_only(problem)


    solutions.append({
        "Problem_ID": idx + 1,
        "SolutionMulti": solution_full,
        "Travel_Time_onlyConsiderMakespan": solution_makespan["AssignmentsSingle"],
        "Makespan_onlyConsiderMakespan": solution_makespan["Makespan_onlyConsiderMakespan"]
    })

# Save solutions as JSON
solution_json_file_path = "fjsp_solutionsGenetic_Algorithm.json"
with open(solution_json_file_path, "w") as json_file:
    json.dump(solutions, json_file, indent=4)

solution_excel_data = []
for solution in solutions:
    problem_id = solution["Problem_ID"]

    assignmentsMulti = solution["SolutionMulti"]["AssignmentsMulti"]
    MakespanMulti = solution["SolutionMulti"]["MakespanMulti"]
    SyncCostMulti = solution["SolutionMulti"]["Sync CostMulti"]


    assignments_makespan = solution["Travel_Time_onlyConsiderMakespan"]
    makespan_only = solution["Makespan_onlyConsiderMakespan"]


    for loc, loc_data in assignmentsMulti.items():
        for dept, dept_data in loc_data.items():
            row = {
                "Problem_ID": problem_id,
                "Location": loc,
                "Department": dept,
                "Assigned_TeamMulti": dept_data["TeamMulti"],
                "Travel_TimeMulti": dept_data["Travel TimeMulti"],
                "MakespanMulti": MakespanMulti,
                "SyncCostMulti": SyncCostMulti
            }


            if loc in assignments_makespan and dept in assignments_makespan[loc]:
                dept_data_makespan = assignments_makespan[loc][dept]
                row["Assigned_Team_onlyConsiderMakespan"] = dept_data_makespan["Team"]
                row["TravelTime_onlyConsiderMakespan"] = dept_data_makespan["Travel Time"]
                row["Makespan_onlyConsiderMakespan"] = makespan_only


            solution_excel_data.append(row)

solution_excel_file_path = "fjsp_solutionsGenetic Algorithm.xlsx"
pd.DataFrame(solution_excel_data).to_excel(solution_excel_file_path, index=False, engine='openpyxl')


print("Solution JSON file saved at:", solution_json_file_path)
print("Solution Excel file saved at:", solution_excel_file_path)
