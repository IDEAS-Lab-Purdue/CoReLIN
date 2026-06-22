import os
import json
import prior
import time
import numpy as np
import networkx as nx
from itertools import product
from shapely.geometry import Point, Polygon
from ai2thor.controller import Controller
from PIL import Image, ImageDraw, ImageFont

from params import *
from classes.get_floorplan import build_skeleton_and_doors
from classes.representation import llm_repr
from classes.grid_graph import reachable_graph
from classes.price_of_clutter import poc
from classes.receptacle_info import VALID_RECEPTACLES



class mr_env:
    def __init__(self, episode_num:int, env_config_path:str, save_img:bool=False, seed:int=None) -> None:
        if seed is not None:
            np.random.seed(seed)
            print(f"INFO: Seed set to {seed}")

        self.ep_num = episode_num
        with open(env_config_path, 'r') as file:
            data = json.load(file)
            self.env_config = data
        self.save_img = save_img
        self.ep_done = False

        self.action_route = [] # list of strings of actions from high-level planner
        self.route = [] # list of robot pose history
        self.achieved_targets = {} # dict of targets the robot successfully achieved
        self.frame_files = []
        self.floorplan_files = []
        self.disabled_objs = []
        self.obs_encountered = 0
        self.fpv_files = []
        self.viz_files = []

    def reset(self) -> None:
        os.makedirs(scene_path, exist_ok=True)
        self.floorplan_id = self.env_config["house_id"]
        self.dataset = prior.load_dataset("procthor-10k")
        self.dataset = self.dataset["test"]
        self.house = self.dataset[self.floorplan_id]

        self.simulator = Controller(
            scene=self.house,
            agentMode="default",
            visibilityDistance=8.5,
            interactableDistance=1.5,
            width=1024,
            height=768,
            fieldOfView=90,
            snapToGrid=False,
        )
        self.simulator.reset(self.house)
        self.print_error_message("Reset")

        event = self.simulator.step(action="GetReachablePositions", gridSize=GRID_SIZE)
        reachable_positions = event.metadata["actionReturn"]
        self.available_pos = [[round(pos['x'], 2), round(pos['z'], 2)] for pos in reachable_positions]
        self.grapher = reachable_graph(self.available_pos, GRID_SIZE)
        self.graph, self.node_coords = self.grapher.build_graph()

        self.total_nodes = len(reachable_positions)
        bw_cen = nx.betweenness_centrality(self.graph)
        self.max_avg_bw = float(sum(bw_cen.values())) / self.total_nodes

        all_env_objects = []
        spawn_objects = self.env_config["spawn_objects"]
        for i in range(int(len(spawn_objects)/2.00)):
            new_obj = spawn_objects[i]
            attrs = {}
            attrs["objectName"] = new_obj["objectName"]
            attrs["position"] = new_obj["position"]
            attrs["rotation"] = new_obj["rotation"]
            all_env_objects.append(attrs)

        env_objects = self.simulator.last_event.metadata["objects"]
        for obj in env_objects:
            if obj["pickupable"] or obj["moveable"]:
                attrs = {}
                attrs["objectType"] = obj["objectType"]
                attrs["objectName"] = obj["name"]
                attrs["position"] = obj["position"]
                attrs["rotation"] = obj["rotation"]
                all_env_objects.append(attrs)

        self.simulator.step(action='SetObjectPoses', objectPoses=all_env_objects)
        self.print_error_message("SetObjectPoses")

        self.start = np.array(self.env_config["start_pose"])
        self.curr_coord = [self.start[0], self.start[2]]

        self.simulator.step(
            action="TeleportFull",
            x=self.start[0],
            y=self.start[1],
            z=self.start[2],
            rotation={"x": 0.0, "y": self.env_config["initial_orientation"], "z": 0.0},
            standing=True,
            horizon=30.0,
            forceAction=True
        )
        self.print_error_message("TeleportFull")

        self.goals = self.env_config["goals"]
        self.curr_pose = self.get_current_pose()
        self.goals_achieved = 0 #
        self.current_goal = self.goals[str(self.goals_achieved+1)]
        self.str_goal = self.assign_goal()
        self.str_goal = "Bring object type {} to receptacle type {}".format(self.current_goal[0], self.current_goal[1])
        self.pick_ID = None
        self.place_ID = None
        self.reward = 0.0

        self.SR = 0 # Success rate
        self.PL = 0 # Path length
        self.OI = 0 # #object_interactions
        self.NI = 0 # #navigation_actions
        self.cost = 0.0 # Cost of existence

        self.scene_data = llm_repr(self.house, GRID_SIZE, self.available_pos, None, MANIP_COST)

        env_objs = self.simulator.last_event.metadata["objects"]

        objs = [obj for obj in env_objs if obj["objectType"] not in IGNORE_OBJS]
        robot_pose = [self.start[0], self.start[1], self.start[2], self.env_config["initial_orientation"]]
        inv_obj = self.simulator.last_event.metadata["inventoryObjects"]
        self.get_nav_metrics()
        event = self.simulator.step(action="GetReachablePositions", gridSize=GRID_SIZE)
        reachable_positions = event.metadata["actionReturn"]
        available_pos = [[round(pos['x'], 2), round(pos['z'], 2)] for pos in reachable_positions]
        self.goal_san, self.all_paths = self.scene_data.get_san(0.0, self.reward, self.goals_achieved, self.current_goal, self.route, available_pos, robot_pose, objs, inv_obj, self.simulator, True, scene_path + 'scene_action_nav_0.json')
        self.steps = 0
        self.achieved_goal = False

        self.get_camera_pos()
        self.init_overhead_view()
        if self.save_img:
            self.get_overhead_view()
            self.get_floorplan_view()
            self.get_fpv()
        return self.str_goal, self.goal_san 

    def get_nav_metrics(self):
        event = self.simulator.step(action="GetReachablePositions", gridSize=GRID_SIZE)
        reachable_positions = event.metadata["actionReturn"]
        available_pos = [[round(pos['x'], 2), round(pos['z'], 2)] for pos in reachable_positions]
        self.graph2, self.node_coords2 = self.grapher.build_pruned_graph(available_pos)
        bw_cen = nx.betweenness_centrality(self.graph2)
        self.rel_bw = (float(sum(bw_cen.values())) / self.total_nodes) / self.max_avg_bw
        self.overall_poc_value = poc(self.graph, self.graph2, dict(zip(self.graph.nodes(), self.node_coords)), dict(zip(self.graph2.nodes(), self.node_coords2)), return_details=False)

    def get_target_recep(self, goal_ID:str) -> list:
        env_objs = self.simulator.last_event.metadata["objects"]
        return self.scene_data.get_target_recep(self.current_goal[1], goal_ID, env_objs)

    def get_path_data(self, go_to_oid:str) -> dict:
        data = self.scene_data.get_path_rep(go_to_oid, self.goals_achieved, self.cost)
        for each_obj in data:
            oid = list(each_obj.keys())[0]
            recep_data = self.get_obj_recep_data(go_to_oid, [oid])
            receps = recep_data[0][oid]
            each_obj[oid]["valid_receptacles"] = receps
        return data

    def get_actually_reachable_pts(self):
        event = self.simulator.step(action="GetReachablePositions", gridSize=GRID_SIZE)
        reachable_positions = event.metadata["actionReturn"]
        available_pos = [[round(pos['x'], 2), round(pos['z'], 2)] for pos in reachable_positions]
        return available_pos

    def assign_goal(self):
        inv_obj = self.simulator.last_event.metadata["inventoryObjects"]
        if len(inv_obj) == 0:
            return "Reach {} object.".format(self.current_goal[0])
        else:
            if inv_obj[0]["objectType"] != self.current_goal[0]:
                return "Reach {} object.".format(self.current_goal[0])
            else:
                return "Place {} object on {}.".format(self.current_goal[0], self.current_goal[1])

    def is_goal_achieved(self) -> bool:
        env_objs = self.simulator.last_event.metadata["objects"]
        recep_objs = [obj for obj in env_objs if obj["objectId"].split('|')[0] == self.current_goal[1]]
        for recep in recep_objs:
            contains = recep["receptacleObjectIds"]
            if contains is not None:
                for obj_ID in contains:
                    obj_type = obj_ID.split('|')[0]
                    if obj_type == self.current_goal[0]:
                        return True
        if self.pick_ID is not None and self.place_ID is not None:
            pick_type = self.pick_ID.split('|')[0]
            place_type = self.place_ID.split('|')[0]
            if pick_type == self.current_goal[0] and place_type == self.current_goal[1]: return True
        return False

    def execute_plan(self, move_to_obj_ID:str, plan:dict):
        seq = []
        interacts = plan["interact"]
        detours = plan["avoid"]
        self.obs_encountered += len(interacts) + len(detours)

        obj_recep_asgn = plan["object_receptacle_pairs"]
        for item in obj_recep_asgn:
            obj_ID = item["object_ID"]
            recep_ID = item["receptacle_ID"]
            seq.append({"oid": obj_ID, "type": "interact", "move_to": recep_ID})

        exec_plan = []
        prev_was_detour = False
        for item in seq:
            if item["type"] == "interact": # Pick up the object and place on the receptacle
                if prev_was_detour:
                    exec_plan.append({"action": "NavigateTo", "oid": item["oid"], "type": "detour"})
                    prev_was_detour = False # reset state
                else:
                    exec_plan.append({"action": "NavigateTo", "oid": item["oid"], "type": "receptacle"})
                exec_plan.append({"action": "PickObject", "oid": item["oid"]})
                if item["oid"].split('|')[0] != self.current_goal[0] and item["move_to"].split('|')[0] != self.current_goal[1]:
                    self.OI += 1.0
                exec_plan.append({"action": "NavigateTo", "oid": item["move_to"], "type": "receptacle"})
                exec_plan.append({"action": "PutObject", "oid": item["move_to"]})
            if item["type"] == "detour":
                prev_was_detour = True # Detours around this object to reach next decision point
        if prev_was_detour:
            exec_plan.append({"action": "NavigateTo", "oid": move_to_obj_ID, "type": "detour"})
        else:
            exec_plan.append({"action": "NavigateTo", "oid": move_to_obj_ID, "type": "receptacle"})
        print(exec_plan)
        a, b = self.execute_actions(exec_plan)
        for objID in self.disabled_objs:
            self.simulator.step(action="EnableObject", objectId=objID)
        return a, b

    def execute_actions(self, plan:list[dict]) -> None:
        last = False
        for item in plan:
            act = item["action"]
            oid = item["oid"]
            try: option = item["type"]
            except: option = None
            print("Executing {} for {} by {}".format(act, oid, option))
            if item == plan[-1]: last = True
            else: False
            
            if act == "NavigateTo":
                a, b, c, d = self.navigate_to(oid, option)
            else:
                a, b, c, d = self.execute_step(act, oid, last_step=last)
        self.print_progress()
        return self.str_goal, self.goal_san #a, b

    def get_graph(self, obj:dict) -> nx.Graph:
        pts = self.get_actually_reachable_pts()
        obj_pos = [obj["position"]["x"], obj["position"]["z"]]
        obj_idx = self.scene_data.validator.find_coord_idx(obj_pos)
        obj_pt = self.node_coords[obj_idx]
        pts.append([float(obj_pt[0]), float(obj_pt[1])])
        step = GRID_SIZE
        directions = [(step, 0.0), (-step, 0.0), (0.0, step), (0.0, -step)]
        for dx, dy in directions:
            nbr = [round(obj_pos[0] + dx, 2), round(obj_pos[1] + dy, 2)]
            pts.append(nbr)

        graph, _ = self.grapher.build_pruned_graph(pts)
        return graph

    def get_obj_recep_data(self, goal_oid:str, blocker_objs:list) -> list:
        data = []
        env_objs = self.simulator.last_event.metadata["objects"]
        goal_obj = [g_obj for g_obj in env_objs if g_obj["objectId"] == goal_oid][0]
        goal_pos = [goal_obj["position"]["x"], goal_obj["position"]["z"]]
        goal_idx = self.scene_data.validator.find_coord_idx(goal_pos)

        for blocker in blocker_objs:
            obj = [obje for obje in env_objs if obje["objectId"] == blocker][0]
            obj_pos = [obj["position"]["x"], obj["position"]["z"]]
            obj_idx = self.scene_data.validator.find_coord_idx(obj_pos)
            graph = self.get_graph(obj)
            receps = []
            valid_receps = [recep for recep in env_objs if 
                            recep["objectType"] in VALID_RECEPTACLES[obj["objectType"]] and 
                            not recep["openable"] and 
                            recep["objectId"] in self.scene_data.ever_visible_objs and
                            recep["objectId"].split('|')[0] not in ["wall", "door", "room"]]
            
            for each_recep in valid_receps:
                recep_attrs = {}
                recep_pos = [each_recep["position"]["x"], each_recep["position"]["z"]]
                recep_idx = self.scene_data.validator.find_coord_idx(recep_pos)
                # blocker to recep
                try:
                    path = nx.shortest_path(graph, str(obj_idx), str(recep_idx))
                except:
                    continue # No path to receptacle, skip 
                recep_attrs["recep_object_ID"] = each_recep["objectId"]
                recep_attrs["cost_to_bring_obj_to_recep"] = len(path)
                # recep to goal
                blocking_obstacles = []
                try:
                    path = nx.shortest_path(self.graph, str(recep_idx), str(goal_idx))
                    for idx in path[:-1]:
                        xn, zn = self.node_coords[int(idx)]
                        b_obj, is_free = self.scene_data.validator._is_free(each_recep["objectId"], xn, zn, env_objs, 0.2, None)
                        for b in b_obj:
                            if b not in blocking_obstacles:
                                blocking_obstacles.append(b)
                except:
                    continue # No path to receptacle, skip 
                recep_attrs["minimum_cost_to_reach_goal_from_recep"] = len(path)
                recep_attrs["obstacles_between_goal_and_recep"] = len(blocking_obstacles)
                receps.append(recep_attrs)
            data.append({obj["objectId"]: receps})
        return data

    def execute_step(self, action:str, args=None, last_step:bool=True) -> None:
        self.steps += 1
        is_act_success = False 
        error_msg = "Error"
        self.update_metrics(action)

        if action in ["MoveAhead", "MoveBack", "MoveRight", "MoveLeft"]:
            self.simulator.step(action=action, moveMagnitude=GRID_SIZE, forceAction=True)
            self.route.append({action: GRID_SIZE})
            is_act_success = self.simulator.last_event.metadata["lastActionSuccess"]
            error_msg = self.simulator.last_event.metadata["errorMessage"]
        if action == "RotateRight":
            self.simulator.step("RotateRight", forceAction=True)
            self.route.append({action: 90})
            is_act_success = self.simulator.last_event.metadata["lastActionSuccess"]
            error_msg = self.simulator.last_event.metadata["errorMessage"]
        if action == "RotateLeft":
            self.simulator.step("RotateLeft", forceAction=True)
            self.route.append({action: 90})
            is_act_success = self.simulator.last_event.metadata["lastActionSuccess"]
            error_msg = self.simulator.last_event.metadata["errorMessage"]

        if action == "PickObject":
            env_objs = self.simulator.last_event.metadata["objects"]
            relevant_obj = [obj for obj in env_objs if obj["objectId"] == args][0]
            self.pick_ID = args
            if relevant_obj["distance"] < 1.75:
                self.simulator.step(action="PickupObject", objectId=args, forceAction=True)
            else:
                self.simulator.step(action="PickupObject", objectId=args, forceAction=False)
            self.print_error_message(action)
            is_act_success = self.simulator.last_event.metadata["lastActionSuccess"]
            error_msg = self.simulator.last_event.metadata["errorMessage"]
            self.route.append({action: args})
            self.get_nav_metrics()
        if action == "PutObject":
            self.place_ID = args
            recep_type = args.split('|')[0]
            if recep_type in OPENABLE_RECEPTACLES:
                self.simulator.step("OpenObject", objectId=args)
            self.simulator.step(action, objectId=args, forceAction=True)
            self.print_error_message(action)
            is_act_success = self.simulator.last_event.metadata["lastActionSuccess"]
            error_msg = self.simulator.last_event.metadata["errorMessage"]
            if not is_act_success:
                try:
                    inv_obj = self.simulator.last_event.metadata["inventoryObjects"][0]
                    self.simulator.step(action="DropHandObject", forceAction=True)
                    self.simulator.step(action="DisableObject", objectId=inv_obj["objectId"])
                    self.disabled_objs.append(args)
                except:
                    inv_obj = ''
            if recep_type in OPENABLE_RECEPTACLES:
                self.simulator.step("CloseObject", objectId=args)
            self.route.append({action: args})
        self.print_error_message(action)
        env_objs = self.simulator.last_event.metadata["objects"]
        for obj in env_objs:
            if obj["visible"] and obj["objectId"] not in self.scene_data.ever_visible_objs:
                self.scene_data.ever_visible_objs.append(obj["objectId"])
        if self.save_img:
            self.get_overhead_view(action, args)
            self.get_fpv()
            # self.get_floorplan_view()
        self.cost = self.NI + MANIP_COST*self.OI
        self.write_results()
        for _ in range(20):
            self.get_next_goal()
        env_objs = self.simulator.last_event.metadata["objects"]
        self.curr_pose = self.get_current_pose()
        inv_obj = self.simulator.last_event.metadata["inventoryObjects"]
        if len(inv_obj) > 0:
            objs1 = [obj for obj in env_objs if obj["objectType"] not in IGNORE_OBJS]
            objs = [obj for obj in objs1 if obj["objectId"] != inv_obj[0]["objectId"]]
        else:
            objs = [obj for obj in env_objs if obj["objectType"] not in IGNORE_OBJS]
        self.str_goal = "Bring object type {} to receptacle type {}".format(self.current_goal[0], self.current_goal[1])
        if action in ["PickObject", "PutObject", "SpecialToken"]:
            event = self.simulator.step(action="GetReachablePositions", gridSize=GRID_SIZE)
            reachable_positions = event.metadata["actionReturn"]
            available_pos = [[round(pos['x'], 2), round(pos['z'], 2)] for pos in reachable_positions]
            self.goal_san, self.all_paths = self.scene_data.get_san(self.cost, self.reward, self.goals_achieved, self.current_goal, self.route, available_pos, self.curr_pose, objs, inv_obj, self.simulator, True, scene_path + 'scene_action_nav_0.json')
        return self.str_goal, self.goal_san, is_act_success, error_msg

    def write_results(self):
        if self.obs_encountered == 0:
            div_by = 1.00
        else:
            div_by = self.obs_encountered
        perc_interacted = self.OI / div_by

        os.makedirs(results_path + str(len(self.house["rooms"])), exist_ok=True)

        with open(results_path + "/{}/fp_{}.csv".format(len(self.house["rooms"]), self.floorplan_id), "a") as f:
            f.write("{}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(self.SR, self.PL, self.OI, self.NI, perc_interacted, self.rel_bw, self.overall_poc_value, self.cost, self.steps))
            f.close()

    def update_metrics(self, action:str) -> None:
        if action in ["MoveAhead", "MoveBack", "MoveRight", "MoveLeft"]:
            self.PL += GRID_SIZE
            self.NI += 1
        if action in ["RotateRight", "RotateLeft"]:
            self.NI += 1
        if action in ["PickObject", "PutObject"]:
            pass

    def get_next_goal(self):
        if self.is_goal_achieved():
            self.scene_data.goal_found = False
            self.scene_data.target_found = False
            self.achieved_goal = True
            self.SR += 1
            self.goals_achieved += 1
            self.reward += 10.0
            try:
                self.current_goal = self.goals[str(self.goals_achieved+1)]
                print("New goal is - ", self.current_goal)
            except:
                self.current_goal = [None, None]
                if self.SR > 10:
                    self.ep_done = True
                self.print_progress()
                pass
            self.str_goal = self.assign_goal()
            self.pick_ID = None
            self.place_ID = None
            self.print_progress()
        else:
            pass

    def update_graph(self):
        event = self.simulator.step(action="GetReachablePositions", gridSize=2*GRID_SIZE)
        r_pos = event.metadata["actionReturn"]
        available_pos = [[round(pos['x'], 2), round(pos['z'], 2)] for pos in r_pos]

        self.grapher = reachable_graph(available_pos, GRID_SIZE)
        self.graph, self.node_coords = self.grapher.build_graph()

    def get_point_room_id(self, pt:list):
        p = Point(pt[0], pt[1])
        for room in self.house["rooms"]:
            poly = Polygon([(v["x"], v["z"]) for v in room["floorPolygon"]])
            if poly.contains(p):
                return room["id"]
        return None

    def find_nearest_coord(self, coord:np.ndarray) -> int:
        curr_room_id = self.get_point_room_id(coord)
        sorted_nodes = sorted(self.node_coords, key=lambda n: np.linalg.norm(n - coord))
        for node in sorted_nodes:
            room_id = self.get_point_room_id(node)
            if room_id == curr_room_id:
                return node

    def navigate_to(self, oid:str, option:str):
        env_objs = self.simulator.last_event.metadata["objects"]
        try:
            path = self.all_paths[oid][option]
        except: # Receptacle, get either shortest or detour, whichever is available
            path_chars = self.scene_data.san[-1]["robot"]["valid_navigation_actions"]["NavigateTo"][oid]
            if path_chars["shortest_path_exists"]:
                path = self.all_paths[oid]["shortest"]
            else:
                path = self.all_paths[oid]["detour"]
        relevant_obj = [obj for obj in env_objs if obj["objectId"] == oid]
        relevant_obj = relevant_obj[0]
        obj_pos = [relevant_obj["position"]["x"], relevant_obj["position"]["z"]]
        if len(path) > 1: # At least 2 coords to find the movement action between them
            actions = []
            yaw = np.radians(self.curr_pose[3])
            ahead_dx, ahead_dz = np.sin(yaw), np.cos(yaw)       # right-handed: yaw about +Y
            right_dx, right_dz =  np.cos(yaw), -np.sin(yaw)     # 90° clockwise from ahead
            for i in range(1, len(path)):
                x0, z0 = self.node_coords[int(path[i-1])][0], self.node_coords[int(path[i-1])][1]
                nav_candidates = {
                    "MoveAhead": (x0 + GRID_SIZE * ahead_dx, z0 + GRID_SIZE * ahead_dz),
                    "MoveBack": (x0 - GRID_SIZE * ahead_dx, z0 - GRID_SIZE * ahead_dz),
                    "MoveRight": (x0 + GRID_SIZE * right_dx, z0 + GRID_SIZE * right_dz),
                    "MoveLeft": (x0 - GRID_SIZE * right_dx, z0 - GRID_SIZE * right_dz),
                }
                new_x, new_z = self.node_coords[int(path[i])][0], self.node_coords[int(path[i])][1]
                for act, (xn, zn) in nav_candidates.items():
                    if round(xn, 2) == round(new_x, 2) and round(zn, 2) == round(new_z, 2):
                        actions.append(act)
                        break
            for act in actions[:-1]:
                self.execute_step(act, last_step=False)
            a, b, c, d = self.execute_step(actions[-1])
            rotates = 0
            while True:
                env_objs = self.simulator.last_event.metadata["objects"]
                relevant_obj = [obj for obj in env_objs if obj["objectId"] == oid]
                relevant_obj = relevant_obj[0]

                if relevant_obj["visible"]: break

                a, b, c, d = self.execute_step("RotateRight")
                rotates += 1
                if rotates == 5:
                    break
            return a, b, c, d
        else:
            is_act_success = True
            error_msg = "You are already at {}. Rotate to face the object.".format(oid)
            return self.str_goal, self.goal_san, is_act_success, error_msg
        
    def print_error_message(self, source:str) -> None:
        pass
        # if not self.simulator.last_event.metadata["lastActionSuccess"]:
        #     print("ERROR: {} - {}".format(source, self.simulator.last_event.metadata["errorMessage"]))

    def get_current_pose(self) -> np.ndarray:
        event = self.simulator.last_event
        position = event.metadata["agent"]["position"]
        rotation = event.metadata["agent"]["rotation"]
        pose = np.array([position['x'], position['y'], position['z'], rotation['y']])
        return pose

    def get_camera_pos(self) -> None:
        x_mins, x_maxs, z_mins, z_maxs, y_tops = [], [], [], [], []
        for obj in self.simulator.last_event.metadata["objects"]:
            if not obj.get("axisAlignedBoundingBox"):
                continue

            aabb = obj["axisAlignedBoundingBox"]
            center = aabb["center"]
            size = aabb["size"]

            x_mins.append(center["x"] - size["x"] / 2)
            x_maxs.append(center["x"] + size["x"] / 2)
            z_mins.append(center["z"] - size["z"] / 2)
            z_maxs.append(center["z"] + size["z"] / 2)
            y_tops.append(center["y"] + size["y"] / 2)
        
        x_center = (min(x_mins) + max(x_maxs)) / 2
        z_center = (min(z_mins) + max(z_maxs)) / 2
        y_camera = max(y_tops)

        self.cam_pose = [x_center, y_camera, z_center]

    def init_overhead_view(self) -> None:
        if not os.path.exists(gifs_path):
            os.makedirs(gifs_path, exist_ok=True)

        bounds = self.simulator.last_event.metadata["sceneBounds"]
        size   = bounds["size"]
        half_extent = max(size["x"], size["z"]) / 2.0
        margin = 0.05 * half_extent
        self.simulator.step(
            action="AddThirdPartyCamera",
            position={"x": self.cam_pose[0], "y": self.cam_pose[1], "z": self.cam_pose[2]},
            rotation={"x": 90.0, "y": 0.0, "z": 0.0},
            orthographic=True,
            orthographicSize=half_extent + margin,
            skyboxColor="white"
        )
        self.cam_size = half_extent + margin

    def plot_path(self, path:list, obj_pos:list) -> None:
        frame = self.simulator.last_event.third_party_camera_frames[0]
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)

        r = 6
        # Plot agent route
        for i in range(1, len(path)):
            from_coord = [self.node_coords[int(path[i-1])][0], 0.0, self.node_coords[int(path[i-1])][1]]
            to_coord = [self.node_coords[int(path[i])][0], 0.0, self.node_coords[int(path[i])][1]]
            from_x, from_y = self.world_to_image(from_coord, img.width, img.height)
            to_x, to_y = self.world_to_image(to_coord, img.width, img.height)
            draw.line([[from_x, from_y], [to_x, to_y]], fill=(0, 255, 0), width=2)

        # Plot end point
        ep_x, ep_y = self.world_to_image([obj_pos[0], 0.0, obj_pos[1]], img.width, img.height)
        draw.ellipse([ep_x-r, ep_y-r, ep_x+r, ep_y+r], fill="red")

        start_xy = self.world_to_image(self.start, img.width, img.height)
        draw.ellipse([start_xy[0]-r, start_xy[1]-r, start_xy[0]+r, start_xy[1]+r], fill="red")

        if self.save_img:
            save_path = gifs_path + 'ep_{}_fp_{}_path_{}.png'.format(self.ep_num, self.floorplan_id, self.steps)
            img.save(save_path)

    def get_floorplan_view(self):
        agent_pose = self.get_current_pose()
        pose = {"x":agent_pose[0], 
                "y": agent_pose[1], 
                "z": agent_pose[2], 
                "yaw": agent_pose[3]}
        path = gifs_path + 'floorplan/{}.png'.format(self.steps)
        os.makedirs(gifs_path + "floorplan/", exist_ok=True)
        build_skeleton_and_doors(self.house, pose, ppm=10, out_path=path)
        self.floorplan_files.append(path)

    def get_overhead_view(self, action=None, args=None) -> None:
        frame = self.simulator.last_event.third_party_camera_frames[0]

        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default(size=8)

        r = 2
        target_id = 0 # For the case when no target has been achieved yet
        for target_id, target_val in self.achieved_targets.items():
            obj_type, goal_coord = target_val[0], target_val[1]
            new_label = str(target_id) + ' : ' + obj_type
            x, y = self.world_to_image(goal_coord, img.width, img.height)
            draw.ellipse([x-r, y-r, x+r, y+r], fill="green")
            draw.text((x, y), new_label, fill="black", font=font)

        env_objs = self.simulator.last_event.metadata["objects"]

        goal_locations = [obj["position"] for obj in env_objs if obj["objectType"] == self.current_goal[0]]
        recep_locations = [obj["position"] for obj in env_objs if obj["objectType"] == self.current_goal[1] and obj["objectId"] != "Shelf|+00.17|+00.10|+00.20"]
        for pos in goal_locations:
            coord = [pos["x"], pos["y"], pos["z"]]
            x, y = self.world_to_image(coord, img.width, img.height)
            draw.ellipse([x-r, y-r, x+r, y+r], fill="green")
            draw.text((x, y), self.current_goal[0], fill="black", font=font)
        for pos in recep_locations:
            coord = [pos["x"], pos["y"], pos["z"]]
            x, y = self.world_to_image(coord, img.width, img.height)
            draw.ellipse([x-r, y-r, x+r, y+r], fill="purple")
            draw.text((x, y), self.current_goal[1], fill="black", font=font)

        agent_pose = self.simulator.last_event.metadata["agent"]["position"]
        coord = [agent_pose["x"], agent_pose["y"], agent_pose["z"]]
        x, y = self.world_to_image(coord, img.width, img.height)
        agent_yaw = self.simulator.last_event.metadata["agent"]["rotation"]["y"]
        arrow_len = 0.25
        agent_yaw_rad = np.radians(agent_yaw)
        fx, fy, fz = round(np.sin(agent_yaw_rad), 2), 0.0, round(np.cos(agent_yaw_rad), 2)
        tip_world = [agent_pose["x"] + arrow_len * fx, agent_pose["y"], agent_pose["z"] + arrow_len * fz]
        xt, yt = self.world_to_image(tip_world, img.width, img.height)
        theta = np.atan2(yt-y, xt-x)
        phi = np.radians(28)
        head_len = 20
        left = (xt - head_len * np.cos(theta - phi), yt - head_len * np.sin(theta - phi))
        right = (xt - head_len * np.cos(theta + phi), yt - head_len * np.sin(theta + phi))
        draw.polygon([(xt, yt), left, right], fill="red")

        font = ImageFont.load_default(size=20)
        to_print = "Bring {} to {}. SR: {}, PL: {}, Nav: {}, Manip: {} \n Current action - {}, {}".format(self.current_goal[0], self.current_goal[1], self.SR, self.PL, self.NI, self.OI, action, args)
        bbox = draw.textbbox((0, 0), to_print, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]        
        padding = 25  # space for the label
        new_width = img.width
        new_height = img.height + padding
        new_img = Image.new("RGB", (new_width, new_height), color="white")
        new_img.paste(img, (0, padding))
        draw = ImageDraw.Draw(new_img)
        x = (new_width - text_width) // 2
        y = (padding - text_height) // 2 + 10
        draw.text((x, y), to_print, font=font, fill="black")

        if self.save_img:
            save_path = gifs_path + 'overhead/{}.png'.format(self.steps)
            os.makedirs(gifs_path + "overhead/", exist_ok=True)
            new_img.save(save_path)
            self.frame_files.append(save_path)

    def get_fpv(self):
        img = self.simulator.last_event.frame
        img = Image.fromarray(img)
        save_path = gifs_path + 'fpv/{}.png'.format(self.steps)
        os.makedirs(gifs_path + "fpv/", exist_ok=True)
        img.save(save_path)

    def world_to_image(self, env_pos:list, img_width:int, img_height:int) -> tuple:
        world_height = 2 * self.cam_size
        world_width = world_height * (img_width / img_height)
        dx = env_pos[0] - self.cam_pose[0]  # x-axis (left-right)
        dz = env_pos[2] - self.cam_pose[2]  # z-axis (top-bottom)

        u = int((dx + world_width / 2) / world_width * img_width)
        v = int((-(dz - world_height / 2)) / world_height * img_height)
        return (u, v)

    def show_graph(self, given_graph=None, plot_pos=None) -> None:
        frame = self.simulator.last_event.third_party_camera_frames[0]

        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)

        font = ImageFont.load_default()

        r = 5
        if plot_pos is not None:
            x2, y2 = self.world_to_image(plot_pos, img.width, img.height)
            draw.ellipse([x2-r, y2-r, x2+r, y2+r], fill="green")

        if given_graph is not None:
            edges = given_graph.edges
        else:
            edges = self.graph.edges
        r = 2
        for edge in edges:
            from_idx, to_idx = edge
            from_coord = self.node_coords[int(from_idx)]
            to_coord = self.node_coords[int(to_idx)]
            x1, y1 = self.world_to_image([from_coord[0], 0.0, from_coord[1]], img.width, img.height)
            x2, y2 = self.world_to_image([to_coord[0], 0.0, to_coord[1]], img.width, img.height)
            draw.ellipse([x1-r, y1-r, x1+r, y1+r], fill="green")
            draw.ellipse([x2-r, y2-r, x2+r, y2+r], fill="green")
            draw.line(([x1,y1], [x2,y2]), fill='black')
            draw.text((x1, y1), '{}'.format(from_idx), fill="black", font=font)
            draw.text((x2, y2), '{}'.format(to_idx), fill="black", font=font)

        if self.save_img:
            save_path = gifs_path + 'ep_{}_graph.png'.format(self.ep_num)
            print("DEBUG: Saved at - ", save_path)
            img.save(save_path)

    def make_gif(self):
        path = gifs_path + '{}_test.gif'.format(self.floorplan_id)
        im1 = Image.open(self.frame_files[0])
        frames = []
        for img_name in self.frame_files:
            frames.append(Image.open(img_name))
        im1.save(path, save_all=True, append_images=frames, duration=100, loop=0)

        # Remove files
        for filename in self.frame_files[:-1]:
            os.remove(filename)
        for filename in self.floorplan_files[:-1]:
            os.remove(filename)
        print("GIF saved - ", path)

    def print_progress(self) -> None:
        if self.obs_encountered == 0:
            div_by = 1.00
        else:
            div_by = self.obs_encountered
        perc_interacted = self.OI / div_by
        print("SR: {} PL: {} OI: {} NI: {} PercInt: {:.2f} PoC: {:.2f} Cost:{:.2f}".format(self.SR, self.PL, self.OI, self.NI, perc_interacted, self.overall_poc_value, self.cost))

    def end_simulation(self) -> None:
        self.simulator.stop()
        print("SR: {} PL: {} OI: {} NI: {} PoC: {:.2f} Cost:{:.2f}".format(self.SR, self.PL, self.OI, self.NI, self.overall_poc_value, self.cost))
    
    def scan_room(self):
        curr_pos = self.get_current_pose()
        rid = self.scene_data.validator.get_point_room_id([curr_pos[0], curr_pos[2]])
        self.scene_data.validator.room_visits[rid] += 1 # Running scan for the room
        # First go to the middle of the room
        if self.scene_data.validator.room_visits[rid] > 1:
            path = self.all_paths[rid]["middle"]
        if self.scene_data.validator.room_visits[rid] > 2:
            path = self.all_paths[rid]["longest"]

            if len(path) > 1: # At least 2 coords to find the movement action between them
                actions = []
                yaw = np.radians(self.curr_pose[3])
                ahead_dx, ahead_dz = np.sin(yaw), np.cos(yaw)       # right-handed: yaw about +Y
                right_dx, right_dz =  np.cos(yaw), -np.sin(yaw)     # 90° clockwise from ahead
                for i in range(1, len(path)):
                    x0, z0 = self.node_coords[int(path[i-1])][0], self.node_coords[int(path[i-1])][1]
                    nav_candidates = {
                        "MoveAhead": (x0 + GRID_SIZE * ahead_dx, z0 + GRID_SIZE * ahead_dz),
                        "MoveBack": (x0 - GRID_SIZE * ahead_dx, z0 - GRID_SIZE * ahead_dz),
                        "MoveRight": (x0 + GRID_SIZE * right_dx, z0 + GRID_SIZE * right_dz),
                        "MoveLeft": (x0 - GRID_SIZE * right_dx, z0 - GRID_SIZE * right_dz),
                    }
                    new_x, new_z = self.node_coords[int(path[i])][0], self.node_coords[int(path[i])][1]
                    for act, (xn, zn) in nav_candidates.items():
                        if round(xn, 2) == round(new_x, 2) and round(zn, 2) == round(new_z, 2):
                            actions.append(act)
                            break
                # print(actions)
                for act in actions[:-1]:
                    self.execute_step(act, last_step=False)
                _, _, _, _ = self.execute_step(actions[-1])

        # 360 degree scan
        self.execute_step("RotateRight", last_step=False)
        self.execute_step("RotateRight", last_step=False)
        self.execute_step("RotateRight", last_step=False)
        self.str_goal, self.goal_san, _, _ = self.execute_step("SpecialToken")
        return self.goal_san

    def nav_to_room(self, rid:str):
        if self.scene_data.validator.room_visits[rid] > 0:
            path = self.all_paths[rid]["middle"]
        else:
            path = self.all_paths[rid]["first_time"]

        if len(path) > 1: # At least 2 coords to find the movement action between them
            actions = []
            yaw = np.radians(self.curr_pose[3])
            ahead_dx, ahead_dz = np.sin(yaw), np.cos(yaw)       # right-handed: yaw about +Y
            right_dx, right_dz =  np.cos(yaw), -np.sin(yaw)     # 90° clockwise from ahead
            for i in range(1, len(path)):
                x0, z0 = self.node_coords[int(path[i-1])][0], self.node_coords[int(path[i-1])][1]
                nav_candidates = {
                    "MoveAhead": (x0 + GRID_SIZE * ahead_dx, z0 + GRID_SIZE * ahead_dz),
                    "MoveBack": (x0 - GRID_SIZE * ahead_dx, z0 - GRID_SIZE * ahead_dz),
                    "MoveRight": (x0 + GRID_SIZE * right_dx, z0 + GRID_SIZE * right_dz),
                    "MoveLeft": (x0 - GRID_SIZE * right_dx, z0 - GRID_SIZE * right_dz),
                }
                new_x, new_z = self.node_coords[int(path[i])][0], self.node_coords[int(path[i])][1]
                for act, (xn, zn) in nav_candidates.items():
                    if round(xn, 2) == round(new_x, 2) and round(zn, 2) == round(new_z, 2):
                        actions.append(act)
                        break
            for act in actions[:-1]:
                self.execute_step(act, last_step=False)
            _, _, _, _ = self.execute_step(actions[-1])
            self.execute_step("SpecialToken")
        return self.goal_san

    def debug(self):
        pass





if __name__ == "__main__":
    env_path = "dataset/10.json"
    test = mr_env(0, env_path, True, 101)
    a, b = test.reset()
    test.debug()
    test.make_gif()
    test.end_simulation()
