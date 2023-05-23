import gurobipy as gp
from gurobipy import GRB
import datetime
import os
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
def model_builder(instance, courses, rooms, curricula):
    # Create a new model
    model = gp.Model("Course_Timetabling")

    # Create periods
    periods = range(instance.num_periods)

    # Create binary variables x[c, p, r] for each course c, period p and room r
    x = model.addVars(courses, periods, rooms, vtype=GRB.BINARY, name="x")

    # Lectures constrait
    for c, course in courses.items():
        model.addConstr(sum(x[c, p, r] for p in periods for r in rooms) == course.num_lectures)

    # Room occupancy constraint
    for p in periods:
        for r in rooms:
            model.addConstr(sum(x[c, p, r] for c in courses) <= 1)

    # Conflict constraints

    teachers = set(course.teacher for course in courses.values())
    teachers_courses = {teacher: [course for course in courses.values() if course.teacher == teacher] for teacher in teachers}

    for p in periods:
        for cu, curriculum in curricula.items():
            model.addConstr(sum(x[c.name, p, r] for c in curriculum.courses for r in rooms) <= 1)
        for t in teachers:
            # TODO - Calculate the courses of a teacher
            model.addConstr(sum(x[c.name, p, r] for c in teachers_courses[t] for r in rooms) <= 1)

    # Unavailability constraints
    for c, course in courses.items():
        for p in course.unavailability:
            model.addConstr(sum(x[c, p, r] for r in rooms) == 0)

    # # Curriculum compactness constraints
    # r = model.addVars(curricula, periods, vtype=GRB.BINARY, name="z")
    # v = model.addVars(curricula, periods, vtype=GRB.BINARY, name="v")

    # for cu, curriculum in curricula.items():
    #     for p in periods:
    #         model.addConstr(r[cu, p] == sum(x[c.name, p, r] for c in curriculum.courses for r in rooms))
    #     for p in periods[1:-1]:
    #         model.addConstr(v[cu, p] >= r[cu, p-1] - r[cu, p] + r[cu, p+1])

        ### Curriculum compactness constraints ###

    # Add new binary variables r and v
    r = model.addVars(periods, curricula, vtype=GRB.BINARY, name="r")
    v = model.addVars(periods, curricula, vtype=GRB.BINARY, name="v")

    for curriculum in curricula:
        for p in periods:
            model.addConstr(gp.quicksum(x[c.name, p, room] for room in rooms for c in curricula[curriculum].courses) - r[p, curriculum] == 0, f"curriculum_period_{curriculum}_{p}")

    for curriculum in curricula:
        for day in range(instance.days):
            for period in range(instance.periods_per_day):
                p = day * instance.periods_per_day + period  # calculate the absolute period
                if period != 0:  # not the first period of the day
                    if period != (instance.periods_per_day - 1):  # not the last period of the day
                        model.addConstr(-r[p - 1, curriculum] + r[p, curriculum] - r[p + 1, curriculum] - v[p, curriculum] <= 0,
                                        f"compactness_{curriculum}_{day}_{period}")
                    else:  # for the last period of the day
                        model.addConstr(-r[p - 1, curriculum] + r[p, curriculum] - v[p, curriculum] <= 0,
                                        f"compactness_{curriculum}_{day}_{period}_last")
                else:  # for the first period of the day
                    model.addConstr(r[p, curriculum] - r[p + 1, curriculum] - v[p, curriculum] <= 0,
                                    f"compactness_{curriculum}_{day}_{period}_first")

    # Room stability constraints
    y = model.addVars(courses, rooms, vtype=GRB.BINARY, name="y")

    for c, course in courses.items():
        for r in rooms:
            model.addConstr(y[c, r] >= sum(x[c, p, r] for p in periods) / course.num_lectures)

    # Min work days constraints
        ### Min working days constraints ###

    # Create z variables for courses and days
    z = model.addVars(courses, range(instance.days), vtype=GRB.BINARY, name="z")

    # z_c,d can be set to one only if course c takes place at some period of day d
    for c in courses:
        for d in range(instance.days):
            periods_in_day = [p for p in periods if p // instance.periods_per_day == d]
            model.addConstr(gp.quicksum(x[c, p, r] for r in rooms for p in periods_in_day) - z[c, d] >= 0, f"min_days1_{c}_{d}")

    # create w variables
    w = model.addVars(courses, vtype=GRB.INTEGER, name="w")

    # Enforce that course is scheduled at least min_days times
    # We make this a soft constraint by including w[c] in the constraint
    for c, course in courses.items():
        model.addConstr(gp.quicksum(z[c, d] for d in range(instance.days)) + w[c] >= course.min_days, f"min_days2_{c}")

    # Objective function
    room_capacity_penalty = sum((course.num_students - room.capacity)*x[c, p, r] for c, course in courses.items() for p in periods for r, room in rooms.items() 
                                if course.num_students > room.capacity)
    min_work_days_penalty = sum(5 * w[c] for c in courses)
    curriculum_compactness_penalty = 2 * sum(v[p, cu] for cu in curricula for p in periods)
    room_stability_penalty = sum((sum(y[c, r] for r in rooms) - 1) for c in courses)

    model.setObjective(room_capacity_penalty + min_work_days_penalty + curriculum_compactness_penalty + room_stability_penalty, GRB.MINIMIZE)

    model.update()

    return model, x


# Solve the model and print the results
def solve_model_and_print_results(model, timelimit, filename):
    try:
        # Optimize the model
        model.Params.LogFile = filename + '_log' + '.txt'
        model.setParam('TimeLimit', timelimit)
        model.optimize()

        # model.computeIIS()
        # model.write("model.ilp")
        
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
            obj = model.objVal
            lb = model.ObjBound
            gap = model.MIPGap
            time = model.Runtime
            optimality = True
            
        # If no optimal solution was found
        else:
            print('No optimal solution found. Status code: ' + str(model.status))

            # Print solution to file
            sol = ['']
            for v in model.getVars():
                if v.x > 0.5:
                    sol += '%s %g' % (v.varName, v.x) + '\n'
            
            with open(filename + '_best_solution.txt', 'w') as f:
                f.write(''.join(sol))

            # Print objective value
            print('Best Obj: %g' % model.objVal)
            obj = model.objVal
            lb = model.ObjBound
            gap = model.MIPGap
            time = model.Runtime
            optimality = False

    # Catch any Gurobi errors
    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))

    # Catch any non-Gurobi errors
    except Exception as e:
        print(e)

    return [obj, lb, gap, time, optimality]


def run_optimization(instance_name, time_limit_tuple, run_df):
    # Read input data
    instance, courses, rooms, curricula = read_instance_file(f"Instances/{instance_name}.txt")

    # Compute required sets and graphs
    conflict_graph, schedule_graph = preprocess_data(instance, courses, rooms, curricula)

    # Make output directory
    output_dir = f'Output_hard/{time_limit_tuple}/{instance_name}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create a new model
    model, x = model_builder(instance, courses, rooms, curricula)

    model.write(f"{output_dir}/model.lp")

    # Solve the model and print the results
    first_stage_list = solve_model_and_print_results(model, time_limit_tuple[0], f'{output_dir}/')

    # Call the visualization functions
    # Extract the solution into a DataFrame
    sol_df = extract_solution(x,  courses, range(instance.num_periods), rooms, instance.periods_per_day)
    sol_df.to_csv(f"{output_dir}/solution.csv")

    print(f'Solution is feasible: {verify_feasibility(sol_df, instance, courses, curricula, rooms)}')

    penalties = [calculate_penalty_room_constraint(sol_df, instance, courses, rooms), 
                 calculate_penalty_curriculum_compactness(sol_df, instance, curricula),
                 calculate_penalty_min_days_constraint(sol_df, courses),
                 calculate_penalty_room_stability(sol_df, courses)]

    print(f'Total room constraint penalty: {penalties[0]}')
    print(f'Total curriculum constraint penalty: {penalties[1]}')
    print(f'Total min days penalty: {penalties[2]}')
    print(f'Total room stability penalty: {penalties[3]}')

    # Define your days of the week
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][0:instance.days]

    # Create timetables for all teachers
    timetables_teachers = create_timetables(sol_df, 'Teacher', days_of_week, instance.periods_per_day)

    # Create timetables for all courses
    timetables_courses = create_timetables(sol_df, 'Course', days_of_week, instance.periods_per_day)
    # master_timetable_courses = merge_df_cells(list(timetables_courses.values()))
    # master_timetable_courses.to_csv(f"{output_dir}/master_timetable_courses.csv")

    # Create timetables for all curricula
    curricula_timetables = create_curricula_timetables(sol_df, curricula, days_of_week, instance.periods_per_day)


    # timetables_rooms = create_timetables(sol_df, 'Room', days_of_week, instance.periods_per_day)
    # print(timetables_rooms)

    return [instance_name] + [time_limit_tuple] + first_stage_list + penalties + [first_stage_list[0]]

def main(): 
    instance_names = []
    for i in range(1,15):
        if i < 10:
            instance_names.append(f'comp0{i}')
        else:
            instance_names.append(f'comp{i}')
    
    time_limits = [(300, 80), (3300, 500)]

    run_df = pd.DataFrame(columns=['instance', 'time_limit', 'first_stage obj', 'first_stage LB', 'first_stage gap', 'first_stage time', 'first_stage optimality', 'room_capacity penalty', 'curriculum_compactness penalty', 'min_days penalty', 'room_stability penalty', 'total'])

    for time_limit in time_limits[:1]:
        for instance_name in instance_names[:14]:
            run_info = run_optimization(instance_name, time_limit, run_df)
            run_df = pd.concat([run_df, pd.DataFrame([run_info], columns=run_df.columns)], ignore_index=True)
            run_df.to_csv(f'Output_hard/{time_limit}/run_df.csv')

if __name__ == "__main__":
    main()
 