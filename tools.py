class Course:
    def __init__(self, name, teacher, num_lectures, min_days, num_students):
        self.name = name
        self.teacher = teacher
        self.num_lectures = num_lectures
        self.min_days = min_days
        self.num_students = num_students
        self.unavailability = []
        self.eligible_rooms = []

    def add_unavailability(self, period):
        self.unavailability.append(period)

    def compute_eligible_rooms(self, rooms):
        self.eligible_rooms = [room.name for room in rooms.values() if room.capacity >= self.num_students]

    # Define the "__lt__" method
    def __lt__(self, other):
        if not isinstance(other, Course):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return self.name < other.name


class Room:
    def __init__(self, name, capacity):
        self.name = name
        self.capacity = capacity


class Curriculum:
    def __init__(self, name, num_courses, courses):
        self.name = name
        self.num_courses = num_courses
        self.courses = courses


class Instance:
    def __init__(self, name, num_courses, num_rooms, days, periods_per_day, num_curricula, num_constraints):
        self.name = name
        self.num_courses = num_courses
        self.num_rooms = num_rooms
        self.days = days
        self.periods_per_day = periods_per_day
        self.num_curricula = num_curricula
        self.num_constraints = num_constraints
        self.num_periods = days * periods_per_day


''' Note that the conflict graph only considers eligible combinations of course c and period p 
    as the eligibility of the combination is already taken care of by scheduling courses only in 
    eligible periods.'''
class ConflictGraph:
    def __init__(self):
        self.nodes = {}

    def add_conflict(self, c1, p1, c2, p2):
        if (c1, p1) not in self.nodes:
            self.nodes[(c1, p1)] = set()
        if (c2, p2) not in self.nodes:
            self.nodes[(c2, p2)] = set()

        self.nodes[(c1, p1)].add((c2, p2))
        self.nodes[(c2, p2)].add((c1, p1))

    def print_graph(self):
        for node, conflicts in self.nodes.items():
            print(f"Node ({node[0].name}, {node[1]}):")
            for conflict in conflicts:
                print(f"  Conflict with ({conflict[0].name}, {conflict[1]})")

    def get_edges(self):
        edges = set()
        for node, conflicts in self.nodes.items():
            for conflict in conflicts:
                # Ensure that each edge is represented as a tuple of nodes in sorted order
                edge = tuple(sorted([node, conflict]))
                edges.add(edge)
        return list(edges)

    def get_nodes(self):
        return list(self.nodes.keys())

class PeriodGraph:
    def __init__(self, period):
        self.period = period
        self.courses = set()
        self.rooms = set()
        self.edges = set()

    def add_course(self, course):
        self.courses.add(course)

    def add_room(self, room):
        self.rooms.add(room)

    def add_edge(self, course, room):
        if course in self.courses and room in self.rooms:
            self.edges.add((course, room))


class ScheduleGraph:
    def __init__(self, instance, courses):
        # Initialize a PeriodGraph for each period
        self.period_graphs = [PeriodGraph(period) for period in range(instance.num_periods)]
        
        # Generate the schedule graph for each period
        self.generate_schedule_graph(instance, courses)

    def generate_schedule_graph(self, instance, courses):
        # For each period and course
        for period in range(instance.num_periods):
            for course in courses.values():
                # If the course is not unavailable at this period
                if (course, period) not in course.unavailability:
                    # Add the course to the PeriodGraph for this period
                    self.period_graphs[period].courses.add(course)
                    
                    # For each room eligible for this course
                    for room in course.eligible_rooms:
                        # Add the room to the PeriodGraph for this period
                        self.period_graphs[period].rooms.add(room)
                        
                        # If both the course and room are in the PeriodGraph for this period
                        if course in self.period_graphs[period].courses and room in self.period_graphs[period].rooms:
                            # Add an edge between the course and room in the PeriodGraph for this period
                            self.period_graphs[period].edges.add((course, room))
    
    def verify_halls_condition(self):
        # Loop over all periods
        for period in range(len(self.period_graphs)):
            # Retrieve all subsets of courses for this period
            subsets_courses = self.get_all_subsets(self.period_graphs[period].courses)
            
            # For each subset of courses
            for subset in subsets_courses:
                # Get the neighborhood of this subset
                neighborhood = self.get_neighborhood(subset, period)
                
                # If the size of the neighborhood is less than the size of the subset
                if len(neighborhood) < len(subset):
                    # Return False - Hall's condition is not satisfied
                    return False

        # If we've made it this far, then Hall's condition is satisfied for all periods
        return True

    def get_all_subsets(self, courses):
        # Use Python's built-in combinations function to generate all subsets of courses
        from itertools import chain, combinations
        return list(chain.from_iterable(combinations(courses, r) for r in range(len(courses)+1)))

    def get_neighborhood(self, subset, period):
        # Initialize an empty set to store the neighborhood
        neighborhood = set()

        # For each course-room pair in the edges of the period
        for course, room in self.period_graphs[period].edges:
            # If the course is in the subset
            if course in subset:
                # Add the room to the neighborhood
                neighborhood.add(room)

        # Return the neighborhood
        return neighborhood
    
    def get_neighborhood_sizes_for_all_subsets(self):
        # Initialize the dictionary to store neighborhood sizes
        neighborhood_sizes = {}

        # Loop over all periods
        for period in range(len(self.period_graphs)):
            # Retrieve all subsets of courses for this period
            subsets_courses = self.get_all_subsets(self.period_graphs[period].courses)
            
            # For each subset of courses
            for subset in subsets_courses:
                # Get the neighborhood of this subset
                neighborhood = self.get_neighborhood(subset, period)
                
                # Store the size of this neighborhood indexed by the subset and period
                neighborhood_sizes[(frozenset(subset), period)] = len(neighborhood)

        # Return the dictionary of neighborhood sizes
        return neighborhood_sizes


def read_instance_file(file_path):
    rooms = {}
    curricula = {}
    instance = None
    courses = {}

    with open(file_path, 'r') as file:
        lines = file.readlines()

    line_idx = 0
    while line_idx < len(lines):
        line = lines[line_idx].strip()

        if line.startswith('Name:'):
            instance_data = {}
            while not lines[line_idx].strip() == "":
                key, value = lines[line_idx].strip().split(": ")
                instance_data[key] = value
                line_idx += 1
            instance = Instance(instance_data['Name'], int(instance_data['Courses']), int(instance_data['Rooms']),
                                int(instance_data['Days']), int(instance_data['Periods_per_day']),
                                int(instance_data['Curricula']),
                                int(instance_data['Constraints']))

        elif line.startswith('COURSES:'):
            line_idx += 1
            while not lines[line_idx].strip() == "":
                course_data = lines[line_idx].strip().split()
                course = Course(course_data[0], course_data[1], int(course_data[2]), int(course_data[3]), int(course_data[4]))
                courses[course_data[0]] = course
                line_idx += 1

        elif line.startswith('ROOMS:'):
            line_idx += 1
            while not lines[line_idx].strip() == "":
                room_data = lines[line_idx].strip().split()
                room = Room(room_data[0], int(room_data[1]))
                rooms[room_data[0]] = room
                line_idx += 1

        elif line.startswith('CURRICULA:'):
            line_idx += 1
            while not lines[line_idx].strip() == "":
                curriculum_data = lines[line_idx].strip().split()
                num_courses = int(curriculum_data[1])
                curriculum_courses = [courses[course_name] for course_name in curriculum_data[2:2 + num_courses]]
                curriculum = Curriculum(curriculum_data[0], num_courses, curriculum_courses)
                curricula[curriculum_data[0]] = curriculum
                line_idx += 1

        elif line.startswith('UNAVAILABILITY_CONSTRAINTS:'):
            line_idx += 1
            while line_idx < len(lines) and not lines[line_idx].strip().startswith('END') and not lines[line_idx].strip() == "":
                constraint_data = lines[line_idx].strip().split()
                if constraint_data[0] in courses:
                    # convert tuple of room and period to period index
                    period = int(constraint_data[1])*instance.periods_per_day + int(constraint_data[2])
                    # add unavailability to course
                    courses[constraint_data[0]].add_unavailability(period)
                line_idx += 1

        line_idx += 1

    # Print the read data
    print("Instance:")
    print(f"Name: {instance.name}, Courses: {instance.num_courses}, Rooms: {instance.num_rooms}, Days: {instance.days}, "
          f"Periods per day: {instance.periods_per_day}, Curricula: {instance.num_curricula}, Constraints: {instance.num_constraints}")

    print("\nCourses:")
    for course in courses.values():
        print(f"{course.name} {course.teacher} {course.num_lectures} {course.min_days} {course.num_students}")
        for unavailability in course.unavailability:
            print(f"  Unavailability: Period {unavailability}")

    print("\nRooms:")
    for room in rooms.values():
        print(f"{room.name} {room.capacity}")

    print("\nCurricula:")
    for curriculum in curricula.values():
        print(f"{curriculum.name} {curriculum.num_courses}")
        for course in curriculum.courses:
            print(f"  {course.name}")

    return instance, courses, rooms, curricula

def generate_conflict_graph(instance, courses, curricula):
        graph = ConflictGraph()

        # Courses in the same curriculum are in conflict
        for curriculum in curricula.values():
            for period in range(instance.num_periods):
                for i in range(len(curriculum.courses)):
                    # Only include periods that are available for the course
                    if period in curriculum.courses[i].unavailability:
                        pass
                    for j in range(i + 1, len(curriculum.courses)):
                        # Only include periods that are available for the course
                        if period in curriculum.courses[j].unavailability:
                            pass
                        graph.add_conflict(curriculum.courses[i], period, curriculum.courses[j], period)

        return graph

def max_smaller_value(lst, value):
    # Filter list to get only values smaller than the target value
    smaller_values = [i for i in lst if i < value]

    # If no smaller value exists, return None
    if not smaller_values:
        return None

    # Return the maximum value among the smaller values
    return max(smaller_values)


