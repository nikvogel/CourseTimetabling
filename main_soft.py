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
def model_builder_first_stage(instance, courses, rooms, curricula, conflict_graph, schedule_graph):
    # Create a new model
    model = gp.Model("CourseTimetabling")

    # Create periods
    periods = range(instance.num_periods)

    # Add binary variables x[c, p] for each course c and period p
    x = model.addVars(courses, periods, vtype=GRB.BINARY, name="x")

    # Enforce that each course is scheduled exactly as often as required number of lectures
    for course in courses:
        available_periods = list(set(periods) - set(courses[course].unavailability))
        model.addConstr(sum(x[course, period] for period in available_periods) == courses[course].num_lectures)
    print(f'Done: Course lectures')

    ### Constraints for room capacities ###

    # Number of courses that can take place at p is less than or equal to the number of (available) rooms
    for period in periods:
        model.addConstr(sum(x[course, period] for course in courses) <= 
                        instance.num_rooms,  name=f"room_availability_{period}")
    
    # Compute list of room capacities
    room_capacities = [room.capacity for room in rooms]
    # Remove duplicates to get a set of room capacities
    room_capacities = list(set(room_capacities))
    room_capacities.sort()

    y = {}
    for s in room_capacities[:-1]: # For each s ∈ S , except the biggest (see IP model)
        C_s = [course.name for course in courses.values() if course.num_students >= s] 
        for c in C_s:
            for p in periods:
                # Add binary variables y[s, c, p] for each course c, period p and room capacity s
                y[s, c, p] = model.addVar(vtype=GRB.BINARY, name=f"y_{s}_{c}_{p}")
                # Add constraint 
                # TODO - Explain this constraint
                model.addConstr(x[c, p] - y[s, c, p] >= 0, 
                            name=f"room_capacity_{c}_{p}_{s}")

    # 3. sum over c∈C≥s(x_c,p − y_s,c,p) ≤ |R≥s|
    # TODO - Explain this constraint
    for s in room_capacities:
        C_s = [c.name for c in courses.values() if c.num_students >= s] 
        R_s = [r for r in rooms if r.capacity >= s]
        for p in periods:
            model.addConstr(gp.quicksum(x[c, p] - y[s, c, p] for c in C_s) <= len(R_s), 
                            name=f"large_room_availability_{p}_{s}")

    ### Min working days constraints ###

    # Create z variables
    z = model.addVars(courses, range(instance.days), vtype=GRB.BINARY, name="z")

    # Add constraint: sum over p∈d (x_c,p) - z_c,d ≥0 ∀c∈C, d∈D
    for c in courses.keys():
        for d in range(instance.days):
            periods_in_day = [p for p in periods if p // instance.periods_per_day == d]
            model.addConstr(gp.quicksum(x[c, p] for p in periods_in_day) - z[c, d] >= 0, f"min_days1_{c}_{d}")

    # create w variables
    w = model.addVars(courses.keys(), vtype=GRB.INTEGER, name="w")

    # Add constraint: sum over d (z_c,d) + w_c ≥ mnd(c) ∀c∈C
    for c, course in courses.items():
        model.addConstr(gp.quicksum(z[c, d] for d in range(instance.days)) + w[c] >= course.min_days, f"min_days2_{c}")

    ### Curriculum compactness constraints ###

    # Add new binary variables r and v
    r = model.addVars(periods, range(len(curricula)), vtype=GRB.BINARY, name="r")
    v = model.addVars(periods, range(len(curricula)), vtype=GRB.BINARY, name="v")

    for cu_index, cu in enumerate(curricula):
        for p in periods:
            model.addConstr(gp.quicksum(x[c.name, p] for c in cu.courses) - r[p, cu_index] == 0, f"curriculum_period_{cu_index}_{p}")

    for cu_index in range(len(curricula)):
        for day in range(instance.days):
            for period in range(instance.periods_per_day):
                p = day * instance.periods_per_day + period  # calculate the absolute period
                if period != 0:  # not the first period of the day
                    if period != instance.periods_per_day - 1:  # not the last period of the day
                        model.addConstr(-r[p - 1, cu_index] + r[p, cu_index] - r[p + 1, cu_index] - v[p, cu_index] <= 0,
                                        f"compactness_{cu_index}_{day}_{period}")
                    else:  # for the last period of the day
                        model.addConstr(-r[p - 1, cu_index] + r[p, cu_index] - v[p, cu_index] <= 0,
                                        f"compactness_{cu_index}_{day}_{period}_last")
                else:  # for the first period of the day
                    model.addConstr(r[p, cu_index] - r[p + 1, cu_index] - v[p, cu_index] <= 0,
                                    f"compactness_{cu_index}_{day}_{period}_first")
    print(f'Done: Conflict constraints')

    ### Teacher constraint ###

    # Define teachers' sets of courses
    teachers = set(course.teacher for course in courses.values())
    teachers_courses = {teacher: [course for course in courses.values() if course.teacher == teacher] for teacher in teachers}

    # For each period and each teacher, the sum of x[c, p] for all courses taught by the teacher should be <= 1.
    for teacher in teachers:
        for p in periods:
            model.addConstr(
                gp.quicksum(x[c.name, p] for c in teachers_courses[teacher]) <= 1,
                f"teacher_conflicts_{teacher}_{p}")
    return model, x, y, w, v

# Set the objective function
def set_objective_function_first_stage(model, x, y, w, v, instance, courses, rooms, curricula, conflict_graph, penalty_wc, penalty_v):
    # Create periods
    periods = range(instance.num_periods)

    # Set the objective function for the problem
    V_conf = conflict_graph.get_nodes()

    # Compute list of room capacities
    room_capacities = [room.capacity for room in rooms]
    # Remove duplicates to get a set of room capacities
    room_capacities = list(set(room_capacities))
    room_capacities.sort()
    

    # Create dictionary holding obj, relecting the difference between demanded and scheduled room capacity
    obj_s_c_p = {}
    C_s_dict = {}
    for s in room_capacities[:-1]:
        C_s = [c for c in courses.values() if c.num_students >= s]
        C_s_dict[s] = C_s
        for c in C_s:
            for p in periods:
                obj_s_c_p[s, c.name, p] = min(c.num_students - s, s + 1 - s)

    # Define the objective function
    # obj = gp.quicksum(prio(c, p) * x[c, p] for (c, p) in V_conf)
    # TODO - Add priority function
    objective_1 = gp.quicksum(1 * x[c.name, p] for (c, p) in V_conf)
    #objective_2 = gp.quicksum(obj_s_c_p[s, c.name, p] * y[s, c.name, p] for s in room_capacities[:-1] for c in C_s[s] for p in periods)
    objective_2 = 0
    for s in room_capacities[:-1]:
        for c in C_s_dict[s]:
            for p in periods:
                objective_2 += obj_s_c_p[s, c.name, p] * y[s, c.name, p]
    objective_3 = gp.quicksum(penalty_wc * w[c] for c in courses.keys())
    objective_4 = gp.quicksum(penalty_v * v[p, cu_index] for p in periods for cu_index in range(len(curricula)))
    
    # Set the objective function to the model
    model.setObjective(objective_1 + objective_2 + objective_3 + objective_4, GRB.MINIMIZE)

    # Update the model to include the objective
    model.update()

    return model

# Solve the model and print the results
def solve_model_and_print_results_first_stage(model):
    # Solve the model and print the results
    try:
        # Optimize the model
        model.optimize()
        
        # Check if an optimal solution was found
        if model.status == GRB.OPTIMAL:
            print('Optimal solution found')

            # Print solution
            # for v in model.getVars():
             #   print('%s %g' % (v.varName, v.x))

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
    instance, courses, rooms, curricula = read_instance_file("Instances/comp01.txt")

    # Compute required sets and graphs
    conflict_graph, schedule_graph = preprocess_data(instance, courses, rooms, curricula)

    ### First stage ###

    # Create a new model
    model, x, y, w, v = model_builder_first_stage(instance, courses, rooms, curricula, conflict_graph, schedule_graph)

    # Set the objective function
    model = set_objective_function_first_stage(model, x, y, w, v, instance, courses, rooms, curricula, conflict_graph, penalty_wc=5, penalty_v=2)

    model.write("model_first_stage.lp")

    # Solve the model and print the results
    solve_model_and_print_results_first_stage(model)

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


    ### Second stage ###

    #TODO - Add second stage


if __name__ == "__main__":
    main()
