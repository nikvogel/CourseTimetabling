import gurobipy as gp
from gurobipy import GRB
from tools import read_instance_file
from tools import *

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
def model_builder(instance, courses, rooms, curricula):
    # Create a new model
    model = gp.Model("CourseTimetabling")

    return model

# Set the objective function
def set_objective_function(model):
    # Set the objective function for the problem
    pass

# Add constraints
def add_constraints(model, input_data, decision_variables):
    # Add the constraints required for the problem
    pass

# Solve the model and print the results
def solve_model_and_print_results(model, input_data, decision_variables):
    # Solve the model and print the results
    pass


def main():
    # Read input data
    instance, courses, rooms, curricula = read_instance_file("toy.txt")

    # Compute required sets and graphs
    conflict_graph, schedule_graph = preprocess_data(instance, courses, rooms, curricula)

    # Create a new model
    model = model_builder(instance, courses, rooms, curricula)

    # Set the objective function
    set_objective_function(model)

    # Add constraints
    add_constraints(model)

    # Solve the model and print the results
    solve_model_and_print_results(model)


if __name__ == "__main__":
    main()
