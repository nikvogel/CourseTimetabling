import gurobipy as gp
from gurobipy import GRB
from tools import read_instance_file
from tools import *
from output import *

# Compute required sets and graphs
def preprocess_data(instance, courses, rooms, curricula):
    # Compute eligible rooms for each course
    for course in courses.values():
        course.compute_eligible_rooms(rooms)

    # Compute conflict graph
    conflict_graph = generate_conflict_graph(instance, courses, curricula)
    conflict_graph.print_graph()

    # Compute period graphs
    schedule_graph = ScheduleGraph(instance, courses)

    return conflict_graph, schedule_graph

# Build the optimization model
def model_builder(instance, courses, rooms, curricula, conflict_graph, schedule_graph):
    # Create a new model
    model = gp.Model("CourseTimetabling")

    # Create periods
    periods = range(instance.num_periods)

    # Add binary variables x[c, p] for each course c and period p
    x = model.addVars(courses, periods, vtype=GRB.BINARY, name="x")

    # Retrieve neighborhood sizes
    neighborhood_sizes = schedule_graph.get_neighborhood_sizes_for_all_subsets()

    # Enforce that each course is scheduled exactly as often as required number of lectures
    for course in courses:
        available_periods = list(set(periods) - set(courses[course].unavailability))
        model.addConstr(sum(x[course, period] for period in periods) == courses[course].num_lectures)

    # Add matching constraints
    for subset, period in neighborhood_sizes:
        model.addConstr(sum(x[c.name, period] for c in subset) <= neighborhood_sizes[(subset, period)])

    # Add conflict constraints
    conflict_edges = conflict_graph.get_edges()

    for edge in conflict_edges:
        # Unpack edge to get courses and periods
        ((c1, p1), (c2, p2)) = edge

        # Add constraint
        model.addConstr(x[c1.name, p1] + x[c2.name, p2] <= 1)

    return model, x

# Set the objective function
def set_objective_function(model, x, conflict_graph):
    # Set the objective function for the problem
    V_conf = conflict_graph.get_nodes()

    # Define the objective function
    # obj = gp.quicksum(prio(c, p) * x[c, p] for (c, p) in V_conf)
    # TODO - Add priority function
    obj = gp.quicksum(1 * x[c.name, p] for (c, p) in V_conf)

    # Set the objective function to the model
    model.setObjective(obj, GRB.MINIMIZE)

    # Update the model to include the objective
    model.update()

    return model

# Solve the model and print the results
def solve_model_and_print_results(model):
    # Solve the model and print the results
    try:
        # Optimize the model
        model.optimize()
        
        # Check if an optimal solution was found
        if model.status == GRB.OPTIMAL:
            print('Optimal solution found')

            # Print solution
            for v in model.getVars():
                print('%s %g' % (v.varName, v.x))

            # Print objective value
            print('Obj: %g' % model.objVal)
            
        # If no optimal solution was found
        else:
            print('No optimal solution found. Status code: ' + str(model.status))

    # Catch any Gurobi errors
    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))

    # Catch any non-Gurobi errors
    except Exception as e:
        print(e)

    pass


def main():
    # Read input data
    instance, courses, rooms, curricula = read_instance_file("toy.txt")

    # Compute required sets and graphs
    conflict_graph, schedule_graph = preprocess_data(instance, courses, rooms, curricula)

    # Create a new model
    model, x = model_builder(instance, courses, rooms, curricula, conflict_graph, schedule_graph)

    # Set the objective function
    model = set_objective_function(model, x, conflict_graph)

    # Solve the model and print the results
    solve_model_and_print_results(model)

     # Call the visualization functions
    # Extract the solution into a DataFrame
    sol_df = extract_solution(x, courses, range(instance.num_periods), instance.periods_per_day)

    # Define your days of the week
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'][0:instance.days]

    print(sol_df)

    # Create timetables for all teachers
    timetables_teachers = create_timetables(sol_df, 'Teacher', days_of_week, instance.periods_per_day)

    # Create timetables for all courses
    timetables_courses = create_timetables(sol_df, 'Course', days_of_week, instance.periods_per_day)

    # Create timetables for all curricula
    curricula_timetables = create_curricula_timetables(sol_df, curricula, days_of_week, instance.periods_per_day)


if __name__ == "__main__":
    main()
