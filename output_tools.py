import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
        print(f'Timetable {entity}: {e} \n')
        print(sub_df)
        print('\n')

        # Store the timetable
        timetables[e] = sub_df
    return timetables

def create_curricula_timetables(df, curicula, days, periods_per_day):
    """Creates a timetable for curricula."""
    timetables = {}
    for curiculum in curicula:
        # Create a new timetable for the entity with days as columns and periods as rows
        sub_df = pd.DataFrame(columns=days, index=range(1, periods_per_day+1))
        # Course names
        course_names = []
        for course in curiculum.courses:
            course_names.append(course.name)
        # Fill the timetable
        for entry in df[df['Course'].isin(course_names)].itertuples():
            if pd.isna(sub_df.loc[getattr(entry, "Period")+1 , days[getattr(entry, "Day")]]):
                sub_df.at[getattr(entry, "Period")+1 , days[getattr(entry, "Day")]] = getattr(entry, "Course")
            else:
                print(f'Two courses are scheduled in the same period for curriculum: {curiculum.name} \n')

        # Print the timetable
        print(f'Timetable {curiculum.name} \n')
        print(sub_df)
        print('\n')
        
        # Store the timetable
        timetables[curiculum.name] = sub_df
    
    return timetables