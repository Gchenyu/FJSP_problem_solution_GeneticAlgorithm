import os
import pandas as pd
import json
from tqdm import tqdm
import numpy as np
import random
from deap import base, creator, tools, algorithms


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
problem_json_path = "fjsp_problemsGenetic_Algorithm.json"
problem_excel_path = "fjsp_problemsGenetic_Algorithm.xlsx"

if os.path.exists(problem_json_path):
    try:
        with open(problem_json_path, "r") as file:
            fjsp_problems = json.load(file)
        print("Existing problem file found. Skipping problem generation.")
    except json.JSONDecodeError:
        print("JSON file corrupted. Regenerating...")
        fjsp_problems = [generate_random_fjsp() for _ in range(num_problems)]
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
    num_locations = problem["num_locations"]
    num_departments = problem["num_departments"]
    travel_times = problem["travel_times"]
    departments = problem["departments"]
    num_teams = len(travel_times[departments[0]])
    num_variables = num_locations * num_departments


    if "FitnessMulti" in creator.__dict__:
        del creator.FitnessMulti
    if "IndividualMulti" in creator.__dict__:
        del creator.IndividualMulti

    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
    creator.create("IndividualMulti", list, fitness=creator.FitnessMulti)
    toolbox = base.Toolbox()


    def init_individual():
        individual = []
        assigned_teams = {dept: set() for dept in departments}  # Keep track of the teams assigned to each department

        for loc in range(num_locations):
            for dept in departments:
                available_teams = list(set(range(num_teams)) - assigned_teams[dept])  # Select only unused teams
                if not available_teams:
                    available_teams = list(range(num_teams))
                team = random.choice(available_teams)
                assigned_teams[dept].add(team)
                individual.append(team)

        return creator.IndividualMulti(individual)

    toolbox.register("individual", init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # **Fitness Function**
    def evaluate(individual):
        arrival_times = []
        team_usage = {dept: set() for dept in departments}
        penalty = 0

        for loc in range(num_locations):
            loc_times = []
            for dept_idx, dept in enumerate(departments):
                team_idx = individual[loc * num_departments + dept_idx]

                # **Check for reused teams**
                if team_idx in team_usage[dept]:
                    penalty += 100  # Punish duplicate teams
                else:
                    team_usage[dept].add(team_idx)

                loc_times.append(travel_times[dept][team_idx][loc])
            arrival_times.append(loc_times)


        makespan = max((max(times) for times in arrival_times), default=0)
        sync_cost = sum(max(timesMulti) - min(timesMulti) for timesMulti in arrival_times)


        return makespan + penalty, sync_cost

    toolbox.register("mate", tools.cxUniform, indpb=0.5)

    # **Improved mutation: swap mutation, ensuring uniqueness**
    def mutate_swap(individual):

        if num_locations > 1:
            loc1, loc2 = random.sample(range(num_locations), 2)
        else:
            loc1, loc2 = 0, 0
        dept_idx = random.randint(0, num_departments - 1)
        idx1, idx2 = loc1 * num_departments + dept_idx, loc2 * num_departments + dept_idx
        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
        return individual,

    toolbox.register("mutate", mutate_swap)
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("evaluate", evaluate)

    # GENETIC ALGORITHMS
    population = toolbox.population(n=100)
    algorithms.eaMuPlusLambda(
        population, toolbox, mu=100, lambda_=200, cxpb=0.7, mutpb=0.2, ngen=100, verbose=False
    )

    # **Selecting the best individuals**
    best_individual = tools.selBest(population, k=1)[0]
    makespan, sync_cost = toolbox.evaluate(best_individual)

    # **Analysis of final allocation**
    assignments = {}
    for loc in range(num_locations):
        assignments[f"Location {loc + 1}"] = {}
        for dept_idx, dept in enumerate(departments):
            team_idx = best_individual[loc * num_departments + dept_idx]
            assignments[f"Location {loc + 1}"][dept] = {
                "Assigned_TeamMulti": f"Team_{team_idx + 1}",
                "Travel_TimeMulti": travel_times[dept][team_idx][loc]
            }

    return {
        "Assignments": assignments,
        "Makespan": makespan,
        "Sync Cost": sync_cost
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

    # **Optimize individual generation**: Ensure that teams in each department are not reused
    def generate_valid_individual():
        individual = [-1] * num_variables
        used_teams = {dept: set() for dept in departments}  # Record the teams used by each department
        for loc in range(num_locations):
            for dept_idx, dept in enumerate(departments):
                available_teams = list(set(range(num_teams)) - used_teams[dept])
                if available_teams:
                    team = np.random.choice(available_teams)
                else:
                    team = random.choice(available_teams) if available_teams else random.randint(0, num_teams - 1)  # 这里应该改成随机选一个新团队
                individual[loc * num_departments + dept_idx] = team
                used_teams[dept].add(team)
        return creator.IndividualMin(individual)

    toolbox.register("individual", generate_valid_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # **Optimizing Fitness Evaluation**
    def evaluate_makespan(individual):
        arrival_times = []
        team_usage = {dept: set() for dept in departments}
        penalty = 0

        for loc in range(num_locations):
            loc_times = []
            for dept_idx, dept in enumerate(departments):
                team_idx = individual[loc * num_departments + dept_idx]
                loc_times.append(travel_times[dept][team_idx][loc])

                # **Optimize constraint checking**: increase penalty if team is already used
                if team_idx in team_usage[dept]:
                    penalty += 1000  # Increase the penalty appropriately to force the algorithm to find a better solution
                else:
                    team_usage[dept].add(team_idx)

            arrival_times.append(max(loc_times))

        makespan = max(arrival_times) + penalty
        return makespan,

    # **Optimizing mutation operators**
    def custom_mutate(individual, indpb=0.2):

        for loc in range(num_locations):
            for dept_idx, dept in enumerate(departments):
                if random.random() < indpb:
                    available_teams = list(set(range(num_teams)) - {individual[loc * num_departments + i] for i in range(num_departments)})
                    if available_teams:
                        individual[loc * num_departments + dept_idx] = random.choice(available_teams)
        return individual,

    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", custom_mutate)

    # **Optimize selection operator**
    def tournament_selection(population, k):
        tournsize = min(len(population), 3)
        selected = tools.selTournament(population, k, tournsize=tournsize)
        random.shuffle(selected)
        return selected

    toolbox.register("select", tournament_selection)

    toolbox.register("evaluate", evaluate_makespan)


    population = toolbox.population(n=100)
    result_population, _ = algorithms.eaSimple(
        population, toolbox, cxpb=0.7, mutpb=0.2, ngen=100, verbose=False
    )


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
# for idx, problem in tqdm(enumerate(fjsp_problems[:10]), total=num_problems, desc="Solving FJSP problems"):  #test
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

    assignmentsMulti = solution["SolutionMulti"]["Assignments"]
    MakespanMulti = solution["SolutionMulti"]["Makespan"]
    SyncCostMulti = solution["SolutionMulti"]["Sync Cost"]


    assignments_makespan = solution["Travel_Time_onlyConsiderMakespan"]
    makespan_only = solution["Makespan_onlyConsiderMakespan"]


    for loc, loc_data in assignmentsMulti.items():
        for dept, dept_data in loc_data.items():
            row = {
                "Problem_ID": problem_id,
                "Location": loc,
                "Department": dept,
                "Assigned_TeamMulti": dept_data.get("Assigned_TeamMulti", "N/A"),
                "Travel_TimeMulti": dept_data.get("Travel_TimeMulti", "N/A"),
                "MakespanMulti": MakespanMulti,
                "SyncCostMulti": SyncCostMulti
            }


            if loc in assignments_makespan and dept in assignments_makespan[loc]:
                dept_data_makespan = assignments_makespan[loc][dept]
                row["Assigned_Team_onlyConsiderMakespan"] = dept_data_makespan["Team"]
                row["TravelTime_onlyConsiderMakespan"] = dept_data_makespan["Travel Time"]
                row["Makespan_onlyConsiderMakespan"] = makespan_only


            solution_excel_data.append(row)

solution_excel_file_path = "fjsp_solutionsGenetic_Algorithm.xlsx"
df = pd.DataFrame(solution_excel_data)
try:
    with pd.ExcelWriter(solution_excel_file_path, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
except PermissionError:
    print(f"无法写入 {solution_excel_file_path}，请关闭 Excel 文件并重试！")



print("Solution JSON file saved at:", solution_json_file_path)
print("Solution Excel file saved at:", solution_excel_file_path)