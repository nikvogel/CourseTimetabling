import gurobipy as gp
from gurobipy import GRB
import datetime
from tools import read_instance_file
from tools import *
from output_tools import *

# Compute required sets and graphs
def preprocess_data(instance, courses, rooms, curricula):
    '''Only needed for base model - not the actual model'''
    # Compute eligible rooms for each course
    for course in courses.values():
        course.compute_eligible_rooms(rooms)

    # Compute conflict graph
    conflict_graph = generate_conflict_graph(instance, courses, curricula)

    # Compute period graphs
    schedule_graph = ScheduleGraph(instance, courses)

    return conflict_graph, schedule_graph

# Build the optimization model
def model_builder_first_stage(instance, courses, rooms, curricula, conflict_graph, schedule_graph):
    # Create a new model
    model = gp.Model("Course_Timetabling")

    # Create periods
    periods = range(instance.num_periods)

    # Add binary variables x[c, p] for each course c and period p
    x = model.addVars(courses, periods, vtype=GRB.BINARY, name="x")

    # Enforce that each course is scheduled exactly as often as required number of lectures
    for course in courses:
        available_periods = list(set(periods) - set(courses[course].unavailability))
        model.addConstr(sum(x[course, period] for period in available_periods) == courses[course].num_lectures)

    ### Constraints for room capacities ###

    # Number of courses that can take place at p is less than or equal to the number of (available) rooms
    for period in periods:
        model.addConstr(sum(x[course, period] for course in courses) <= 
                        instance.num_rooms,  name=f"room_availability_{period}")
    
    # Compute list of room capacities
    room_capacities = [room.capacity for room in rooms.values()]
    # Remove duplicates and sort
    room_capacities = list(set(room_capacities))
    room_capacities.sort()

    y = {}
    for s in room_capacities[:-1]: # For each s ∈ S , except the biggest (see IP model)
        C_s = [course.name for course in courses.values() if course.num_students > s] 
        for c in C_s:
            for p in periods:
                # Add binary variables y[s, c, p] for course c, period p and room capacity s
                y[s, c, p] = model.addVar(vtype=GRB.BINARY, name=f"y_{s}_{c}_{p}")
                # Add constraint 
                # Enforce that y[s, c, p] can only be one if course c is scheduled at period p
                model.addConstr(x[c, p] - y[s, c, p] >= 0, 
                            name=f"room_capacity_{c}_{p}_{s}")

    # Course scheduled in rooms with capacity at least s is less than or equal to number of scheduled courses with demand of at least s
    # By including y[s, c, p] in the constraint, we make this a soft constraint
    for s in room_capacities[:-1]:
        C_s = [c.name for c in courses.values() if c.num_students > s] 
        R_s = [r.name for r in rooms.values() if r.capacity > s]
        for p in periods:
            model.addConstr(gp.quicksum(x[c, p] - y[s, c, p] for c in C_s) <= len(R_s), 
                            name=f"large_room_availability_{p}_{s}")

    ### Min working days constraints ###

    # Create z variables for courses and days
    z = model.addVars(courses, range(instance.days), vtype=GRB.BINARY, name="z")

    # If a course is scheduled at least once on a day, z[c, d] is 1
    for c in courses.keys():
        for d in range(instance.days):
            periods_in_day = [p for p in periods if p // instance.periods_per_day == d]
            model.addConstr(gp.quicksum(x[c, p] for p in periods_in_day) - z[c, d] >= 0, f"min_days1_{c}_{d}")

    # create w variables
    w = model.addVars(courses.keys(), vtype=GRB.INTEGER, name="w")

    # Enforce that course is scheduled at least min_days times
    # We make this a soft constraint by including w[c] in the constraint
    for c, course in courses.items():
        model.addConstr(gp.quicksum(z[c, d] for d in range(instance.days)) + w[c] >= course.min_days, f"min_days2_{c}")

    ### Curriculum compactness constraints ###

    # Add new binary variables r and v
    r = model.addVars(periods, curricula, vtype=GRB.BINARY, name="r")
    v = model.addVars(periods, curricula, vtype=GRB.BINARY, name="v")

    for curriculum in curricula:
        for p in periods:
            model.addConstr(gp.quicksum(x[c.name, p] for c in curricula[curriculum].courses) - r[p, curriculum] == 0, f"curriculum_period_{curriculum}_{p}")

    for curriculum in curricula:
        for day in range(instance.days):
            for period in range(instance.periods_per_day):
                p = day * instance.periods_per_day + period  # calculate the absolute period
                if period != 0:  # not the first period of the day
                    if period != instance.periods_per_day - 1:  # not the last period of the day
                        model.addConstr(-r[p - 1, curriculum] + r[p, curriculum] - r[p + 1, curriculum] - v[p, curriculum] <= 0,
                                        f"compactness_{curriculum}_{day}_{period}")
                    else:  # for the last period of the day
                        model.addConstr(-r[p - 1, curriculum] + r[p, curriculum] - v[p, curriculum] <= 0,
                                        f"compactness_{curriculum}_{day}_{period}_last")
                else:  # for the first period of the day
                    model.addConstr(r[p, curriculum] - r[p + 1, curriculum] - v[p, curriculum] <= 0,
                                    f"compactness_{curriculum}_{day}_{period}_first")
    print(f'Done: Conflict constraints')

    ### Teacher constraint ###

    # Define teachers' sets of courses
    teachers = set(course.teacher for course in courses.values())
    teachers_courses = {teacher: [course for course in courses.values() if course.teacher == teacher] for teacher in teachers}

    # Teacher can only teach one course at a time (in one period)
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

    # Compute list of room capacities
    room_capacities = [room.capacity for room in rooms.values()]
    # Remove duplicates to get a set of room capacities
    room_capacities = list(set(room_capacities))
    room_capacities.sort()
    
    # Create dictionary holding penalty term for not scheduling demaned room capcity
    obj_s_c_p = {}
    C_s_dict = {}
    for index, s in enumerate(room_capacities[:-1]):
        C_s = [c for c in courses.values() if c.num_students > s]
        C_s_dict[s] = C_s
        for c in C_s:
            for p in periods:
                obj_s_c_p[s, c.name, p] = min(c.num_students - s, room_capacities[index+1] - s)

    ### Set the objective function ###
    
    # Penalty for not scheduling demaneded room capacity
    objective_1 = 0
    for s in room_capacities[:-1]:
        for c in C_s_dict[s]:
            for p in periods:
                objective_1 += obj_s_c_p[s, c.name, p] * y[s, c.name, p]
    # Penalty for not meeting min working days
    objective_2 = gp.quicksum(penalty_wc * w[c] for c in courses.keys())
    # Penalty for not meeting curriculum compactness
    objective_3 = gp.quicksum(penalty_v * v[p, curriculum] for p in periods for curriculum in curricula)
    
    # Set the objective function to the model
    model.setObjective(objective_1 + objective_2 + objective_3, GRB.MINIMIZE)

    # Update the model to include the objective
    model.update()

    return model

# Solve the model and print the results
def solve_model_and_print_results(model, filename):
    try:
        # Optimize the model
        model.Params.LogFile = filename + '_log_' + str(datetime.datetime.now()) + '.txt'
        model.optimize()
        
        # Check if an optimal solution was found
        if model.status == GRB.OPTIMAL:
            print('Optimal solution found')

            # Print solution to file
            sol = ['']
            for v in model.getVars():
                if v.x > 0.5:
                    sol += '%s %g' % (v.varName, v.x) + '\n'
            
            with open(filename + '_solution.txt', 'w') as f:
                f.write(''.join(sol))

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

def preprocess_second_stage(instance, courses, rooms, x, y):
    # Define U = {(course, period) : x[course, period] == 1, for all courses and periods}
    U = [(c, p) for c in courses for p in range(instance.num_periods) if x[c, p].X == 1]

    # Define V = {(room, period) for all rooms and periods}
    V = [(r, p) for r in rooms for p in range(instance.num_periods)]

    # Compute list of room capacities
    room_capacities = [room.capacity for room in rooms.values()]
    # Remove duplicates to get a set of room capacities
    room_capacities = list(set(room_capacities))
    room_capacities.sort()

    # Define set of Edges between U and V
    E = []
    for s in room_capacities[:-1]:
        C_s = [course.name for course in courses.values() if course.num_students > s] 
        for (c, p) in U:
            for (r, period) in V:
                if p == period:
                    #if y[s, c, p].X == 0 and courses[c].num_students <= r.capacity:
                    if courses[c].num_students <= rooms[r].capacity:
                        E.append((c, r, p))
                    if c in C_s:
                        if (y[s, c, p].X == 1 and courses[c].num_students > rooms[r].capacity and 
                            rooms[r].capacity == max_smaller_value(room_capacities, courses[c].num_students)):
                            print(f'I was here: {s},{c},{p}')
                            E.append((c, r, p))
                        else:
                            pass
                    else:
                        pass

    # Remove duplicates            
    E = list(set(E))

    return U, V, E

def model_builder_second_stage(instance, courses, rooms, U, V, E):
    # Define the second-stage model
    stage2_model = gp.Model("Course_Room_Assignment")

    # Define decision variables
    y = stage2_model.addVars(courses, rooms, vtype=GRB.BINARY, name="y")  # y_c,r
    uv = stage2_model.addVars(E, vtype=GRB.BINARY, name="u_v")  # u_c,p v_r,p

    # Objective function - Minimize sum over all courses: Number of rooms used per course (minus the number of courses)
    print(f'Number of courses: {instance.num_courses}')
    stage2_model.setObjective(gp.quicksum(y[c, r] for c in courses for r in rooms) - instance.num_courses , GRB.MINIMIZE)

    # y_c,r is set to 1 if course c is assigned to room r in any period
    for c in courses:
        for r in rooms:
            stage2_model.addConstr(uv.sum(c,r,'*') - 
                                   instance.num_periods * y[c, r] <= 0)
                        
    # Courses have to be assigned to suitable rooms for scheduled periods
    for c, p in U:
        stage2_model.addConstr(gp.quicksum(uv[c, r, p] 
                                for r in [e[1] for e in E if (e[0],e[2]) == (c,p)])== 1)

    # Rooms can only be used once per period
    for r, p in V:
        stage2_model.addConstr(gp.quicksum(uv[c, r, p] 
                                            for c in [e[0] for e in E if (e[1],e[2]) == (r,p)]) <= 1)

    return stage2_model, y, uv


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

    model.write("Output/model_first_stage.lp")

    # Solve the model and print the results
    solve_model_and_print_results(model, 'Output/first_stage')

    # Call the visualization functions
    # Extract the solution into a DataFrame
    sol_df = extract_solution_first_stage(x, courses, range(instance.num_periods), instance.periods_per_day)
    sol_df.to_csv("Output/solution_first_stage.csv")
    

    # Define your days of the week
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][0:instance.days]

    # Create timetables for all teachers
    # timetables_teachers = create_timetables(sol_df, 'Teacher', days_of_week, instance.periods_per_day)

    # Create timetables for all courses
    timetables_courses = create_timetables(sol_df, 'Course', days_of_week, instance.periods_per_day)
    master_timetable_courses = merge_df_cells(list(timetables_courses.values()))
    master_timetable_courses.to_csv("Output/master_timetable_courses.csv")

    # Create timetables for all curricula
    # curricula_timetables = create_curricula_timetables(sol_df, curricula, days_of_week, instance.periods_per_day)

    ### Second stage ###

    U, V, E = preprocess_second_stage(instance, courses, rooms, x, y)

    # Build the second stage model
    model_second_stage, y_2, uv = model_builder_second_stage(instance, courses, rooms, U, V, E)
    model_second_stage.write("Output/model_second_stage.lp")

    # Solve the second stage model
    solve_model_and_print_results(model_second_stage, 'Output/second_stage')

    sol_df = extract_solution_second_stage(uv, instance, sol_df)
    sol_df.to_csv("Output/solution_final.csv")
    
    # timetables_rooms = create_timetables(sol_df, 'Room', days_of_week, instance.periods_per_day)
    # print(timetables_rooms)

if __name__ == "__main__":
    main()
