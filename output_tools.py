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