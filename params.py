FOLDER_NAME = 'final'
ROBOT_RADIUS = 0.2 # Robot is modeled as a cylinder for collision avoidance
ROBOT_HEIGHT = 1.575
GRID_SIZE = 0.25
MANIP_COST = 5.00
H_LEN = 2

IGNORE_OBJS = ["Floor", "Window", "Blinds", "Curtains"]

gifs_path = f'gifs/{FOLDER_NAME}/'
sg_path = f'scene_graphs/{FOLDER_NAME}/'
scene_path = 'scene_data/'
results_path = f"results/{FOLDER_NAME}/"

OPENABLE_RECEPTACLES = ["Box", "Cabinet", "Drawer", "Fridge", "Microwave", "Safe", "Toilet", "Dresser", "Desk", "SideTable"]
