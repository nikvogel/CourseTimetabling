import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from gurobipy import GRB

def extract_solution_first_stage(x, courses, periods, periods_per_day):
    """Extracts the solution from the Gurobi model."""
    sol = []
    for c in courses.values():
        for p in periods:
            if x[c.name, p].x > 0.5:  # if this course is scheduled at this period
                sol.append({'Course': c.name, 'Teacher': c.teacher,  
                            'Day': p // periods_per_day, 'Period': p % periods_per_day})
    return pd.DataFrame(sol)

def create_timetables(df, entity, days, periods_per_day):
    """Creates timetables for the given entity."""
    entities = df[entity].unique()
    timetables = {}
    for e in entities:
        # Create a new timetable for the entity with days as columns and periods as rows
        sub_df = pd.DataFrame(columns=days, index=range(1, periods_per_day+1))

        # Fill the timetable
        for entry in df[df[entity] == e].itertuples():
            if pd.isna(sub_df.loc[getattr(entry, "Period")+1 , days[getattr(entry, "Day")]]):
                sub_df.at[getattr(entry, "Period")+1 , days[getattr(entry, "Day")]] = getattr(entry, "Course")
            else:
                print(f'Two courses are scheduled in the same period for {entity}: {e} \n')

        # Print the timetable
        # print(f'Timetable {entity}: {e} \n')
        # print(sub_df)
        # print('\n')

        # Store the timetable
        timetables[e] = sub_df
    return timetables

def create_curricula_timetables(df, curricula, days, periods_per_day):
    """Creates a timetable for curricula."""
    timetables = {}
    for curriculum in curricula:
        # Create a new timetable for the entity with days as columns and periods as rows
        sub_df = pd.DataFrame(columns=days, index=range(1, periods_per_day+1))
        # Course names
        course_names = []
        for course in curricula[curriculum].courses:
            course_names.append(course.name)
        # Fill the timetable
        for entry in df[df['Course'].isin(course_names)].itertuples():
            if pd.isna(sub_df.loc[getattr(entry, "Period")+1 , days[getattr(entry, "Day")]]):
                sub_df.at[getattr(entry, "Period")+1 , days[getattr(entry, "Day")]] = getattr(entry, "Course")
            else:
                print(f'Two courses are scheduled in the same period for curriculum: {curriculum.name} \n')

        # Print the timetable
        # print(f'Timetable {curriculum} \n')
        # print(sub_df)
        # print('\n')
        
        # Store the timetable
        timetables[curriculum] = sub_df
    
    return timetables

def merge_df_cells(dfs):
    return pd.DataFrame(
        data=[
            [
                [df.iloc[i, j] for df in dfs if pd.notnull(df.iloc[i, j])] 
                for j in range(dfs[0].shape[1])
            ] 
            for i in range(dfs[0].shape[0])
        ],
        index=dfs[0].index,
        columns=dfs[0].columns
    )

def extract_solution_second_stage(uv, instance, sol_df):
    """Extracts the solution from the Gurobi model."""
    for var in uv:
        if uv[var].x > 0.5:
            day = var[2] // instance.periods_per_day
            period_on_day = var[2] % instance.periods_per_day
            row_ix = (sol_df[(sol_df['Course']  == var[0]) & (sol_df['Day'] == day) & (sol_df['Period'] == period_on_day)].index.tolist())
            if len(row_ix) == 1:
                sol_df.at[row_ix[0], 'Room'] = var[1]
            else:
                print('No course is scheduled for the course or more than one course is scheduled in the same period \n')
    return sol_df

def calculate_penalty_room_constraint(df, instance, courses, rooms):

    penalty_room_constraint = 0
    room_capacities = [r.capacity for r in rooms.values()]
    room_capacities.sort(reverse=True)

    for day in range(instance.days):
        for period in range(instance.periods_per_day):
            courses_in_period = df[(df['Day'] == day) & (df['Period'] == period)]['Course'].unique()
            demand_in_period = [c.num_students for c in courses.values() if c.name in courses_in_period]
            demand_in_period.sort(reverse=True)
            for i in range(len(demand_in_period)):
                if demand_in_period[i] > room_capacities[i]:
                    penalty_room_constraint += demand_in_period[i] - room_capacities[i]
                    print(f'Room capacity exceeded for day {day} and period {period} \n')

    return penalty_room_constraint

def calculate_penalty_curriculum_compactness(df, instance, curricula):

    # TODO - Impelemnt curriculum compactness

    penalty_curriculum_compactness = 0

    print('Curriculum compactness not implemented yet \n')

    for cu, curriculum in curricula.items():
        courses_in_curriculum = [course.name for course in curriculum.courses]
        df_curriculum = df[df['Course'].isin(courses_in_curriculum)]
        days_and_periods = df_curriculum[['Day', 'Period']]
        days_and_periods = list(days_and_periods.itertuples(index=False, name=None))
        days_and_periods.sort()

        for index, d_p in enumerate(days_and_periods):
            isolated = True
            if index > 0:
                if days_and_periods[index-1][1] + 1 == d_p[1] and days_and_periods[index-1][0] == d_p[0]: # has predecessor
                    isolated = False
            if index < len(days_and_periods) - 1:
                if days_and_periods[index+1][1] - 1 == d_p[1] and days_and_periods[index+1][0] == d_p[0]: # has successor
                    isolated = False
            if isolated:
                penalty_curriculum_compactness += 1

    return penalty_curriculum_compactness * 2

def calculate_penalty_min_days_constraint(df, courses):

    penalty_min_day = 0

    for c, course in courses.items():
        num_days = len(df[df['Course'] == c]['Day'].unique())
        if num_days < course.min_days:
            penalty_min_day += course.min_days - num_days
            print(f'Course {c} is only scheduled {num_days} days and not {course.min_days} days \n')

    return penalty_min_day * 5

def calculate_penalty_room_stability(df, courses):

    penalty_room_stability = 0

    for c, course in courses.items():
        num_rooms = len(df[df['Course'] == c]['Room'].unique())
        if num_rooms > 1:
            penalty_room_stability += num_rooms - 1
            print(f'Course {c} is scheduled in {num_rooms} rooms')

    return penalty_room_stability

def verify_feasibility(df, instance, courses, curricula, rooms=None):

    # TODO - code here

    feasible = True

    # Lectures
    '''All lectures of a course must be scheduled, and they must be assigned to
    distinct periods. A violation oc- curs if a lecture is not scheduled.'''

    for c, course in courses.items():
        if len(df[df['Course'] == c]) != course.num_lectures:
            print(f'Course {c} is not scheduled exacly {course.num_lectures} times ({len(df[df["Course"] == c])}\n')
            feasible = False

    # Conflicts
    '''Lectures of courses in the same curriculum or taught by the same teacher
    must be all scheduled in different periods. Two conflicting lectures in
    the same period represent one violation. Three conflicting lectures count as
    3 violations: one for each pair.'''

    # Curriculum Conflicts
    for cu, curriculum in curricula.items():
        courses_in_curriculum = [course.name for course in curriculum.courses]
        df_curriculum = df[df['Course'].isin(courses_in_curriculum)]
        days_and_periods = df_curriculum[['Day', 'Period']]
        days_and_periods = list(days_and_periods.itertuples(index=False, name=None))
        duplicates = [p_d for p_d in days_and_periods if days_and_periods.count(p_d) > 1]
        if len(duplicates) > 0:
            print(f'Two courses are scheduled in the same period for curriculum: {curriculum.name} \n')
            print(f'Period {duplicates.unique()} \n')
            feasible = False

    # Teacher Conflicts
    # Define teachers' sets of courses
    teachers = set(course.teacher for course in courses.values())
    teachers_courses = {teacher: [course for course in courses.values() if course.teacher == teacher] for teacher in teachers}

    for teacher, teached_courses in teachers_courses.items():
        df_teacher = df[df['Course'].isin([course.name for course in teached_courses])]
        days_and_periods = df_teacher[['Day', 'Period']]
        days_and_periods = list(days_and_periods.itertuples(index=False, name=None))
        duplicates = [p_d for p_d in days_and_periods if days_and_periods.count(p_d) > 1]
        if len(duplicates) > 0:
            print(f'Two courses are scheduled in the same period for teacher: {teacher} \n')
            print(f'Period {duplicates.unique()} \n')
            feasible = False

    # Availabilities
    '''If the teacher of the course is not available to teach that course at a
    given period, then no lectures of the course can be scheduled at that
    period. Each lecture in a period unavailable for that course is one
    violation.'''

    for c, course in courses.items():
        df_course = df[df['Course'] == c][['Day', 'Period']]
        scheduled_day_periods = list(df_course.itertuples(index=False, name=None))
        unavailable_periods = course.unavailability
        unavailable_day_periods = []
        for p in unavailable_periods:
            day = p// instance.periods_per_day
            period_on_day = p % instance.periods_per_day
            unavailable_day_periods.append((day, period_on_day))
        for unavailable_day_period in unavailable_day_periods:
            if unavailable_day_period in scheduled_day_periods:
                print(f'Course {c} is scheduled in an unavailable period: {unavailable_day_period}\n')
                feasible = False
        
     # RoomOccupancy
    '''Two lectures cannot take place in the same room in the same period. Two
    lectures in the same room at the same period represent one violation . Any
    extra lecture in the same period and room counts as one more violation.'''

    if rooms is not None: # Only checks room occupancy if rooms are provided
        for r, room in rooms.items():
            df_room = df[df['Room'] == r]
            days_and_periods = df_room[['Day', 'Period']]
            days_and_periods = list(days_and_periods.itertuples(index=False, name=None))
            duplicates = [d_p for d_p in days_and_periods if days_and_periods.count(d_p) > 1]
            if len(duplicates) > 0:
                print(f'Two courses are scheduled in the same period for room: {r} \n')
                print(f'Period {duplicates.unique()} \n')
                feasible = False

    return feasible