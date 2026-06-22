import json
import time
from typing import Any
import numpy as np
import networkx as nx
from shapely.geometry import Point, Polygon
from ai2thor.controller import Controller
from copy import deepcopy

from .valid_actions import find_valid_actions

IGNORE_OBJS = ["Floor", "Window", "Blinds", "Curtains", "Wall", "Door"]



class llm_repr:
    def __init__(self, house_config:dict, grid_size:float, reachable_pts:list, duplicator:None, manip_cost:float):
        self.house = house_config
        self.grid_size = grid_size
        self.reachable = reachable_pts
        self.validator = find_valid_actions(grid_size, self.house, manip_cost, reachable_pts)
        self.ever_visible_objs = []
        self.seen_obj_IDs = {}
        for room in self.house["rooms"]:
            self.seen_obj_IDs[room["id"]] = []

        self.goal_found = False 
        self.target_found = False
        self.duplicator = duplicator

    def get_point_room_id(self, pt:list):
        p = Point(pt[0], pt[2])
        for room in self.house["rooms"]:
            poly = Polygon([(v["x"], v["z"]) for v in room["floorPolygon"]])
            if poly.contains(p):
                return room["id"], room["roomType"]
        return None, None

    def dump_json(self, path:str, data:Any):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def find_coord_idx(self, node_coords, coord:np.ndarray) -> int:
        dists = np.linalg.norm(node_coords - coord, axis = 1)
        min_dist = min(dists)
        return np.where(dists == min_dist)[0][0]

    def is_navigable(self, obj:dict, graph:nx.Graph, node_coords:np.ndarray, curr_pose:list) -> bool:
        obj_pos = [obj["position"]["x"], obj["position"]["z"]]
        nearest_node = self.find_coord_idx(node_coords, obj_pos)
        curr_node = self.find_coord_idx(node_coords, [curr_pose[0], curr_pose[2]])
        try:
            path = nx.shortest_path(graph, str(curr_node), str(nearest_node))
            return True
        except nx.NetworkXNoPath:
            return False

    def get_blockers(self, oid_list:list[str]) -> list:
        all_blockers = []
        # print(oid_list)
        for oid in oid_list:
            oid_blockers = self.san[-1]["robot"]["valid_navigation_actions"]["NavigateTo"][oid]["path_blockers"]
            if len(oid_blockers) != 0:
                for id in oid_blockers:
                    if id not in all_blockers:
                        all_blockers.append(id)
                other_blockers = self.get_blockers(oid_blockers)
                for id in other_blockers:
                    if id not in all_blockers:
                        all_blockers.append(id)
        return all_blockers

    def get_goal_rep(self, env_objs:list, env_goal:list) -> list[dict]:
        objs_desc = []
        for obj in env_objs:
            if obj["objectId"].split('|')[0] == env_goal[0] and obj["objectId"] in self.ever_visible_objs: # pickup
                obj_desc = deepcopy(self.san[-1]["robot"]["valid_navigation_actions"]["NavigateTo"][obj["objectId"]])
                obj_desc["path_blockers"] = len(obj_desc["path_blockers"])
                objs_desc.append({obj["objectId"]: obj_desc})
        return objs_desc

    def get_target_recep(self, target_recep_type:str, goal_oid:str, env_objs:list) -> list: # Stop returning None
        potential_receps = [recep for recep in env_objs if recep["objectId"].split('|')[0] == target_recep_type]
        goal_obj = [obj for obj in env_objs if obj["objectId"] == goal_oid][0]
        goal_pos = [goal_obj["position"]["x"], goal_obj["position"]["z"]]
        closest_recep_ID = None
        closest_recep_cost = float('inf')
        for recep in potential_receps:
            if recep["objectId"] in self.ever_visible_objs:
                recep_paths = self.san[-1]["robot"]["valid_navigation_actions"]["NavigateTo"][recep["objectId"]] #EDIT HERE TO GET PATH CHARS
                if recep_paths["shortest_path_exists"]:
                    path = self.all_paths[recep["objectId"]]["shortest"]
                elif recep_paths["detour_path_exists"]:
                    path = self.all_paths[recep["objectId"]]["detour"]
                else:
                    continue
                recep_cost = len(path)
                if recep_cost < closest_recep_cost:
                    closest_recep_ID = recep["objectId"]
                    closest_recep_cost = recep_cost
        return closest_recep_ID

    def get_path_rep(self, oid:str, attained_goals:int, cost:float) -> list[dict]:
        rep = []
        progress = {}
        progress["progress"] = "completed {} out of total 20 tasks".format(attained_goals)
        rep.append(progress)

        cost = {}
        cost["cost_so_far"] = cost
        rep.append(cost)

        all_blockers = []
        obj_desc = []
        all_blockers = self.get_blockers([oid])
        all_blockers.append(oid)
        for obj_ID in all_blockers:
            obj_rep = deepcopy(self.san[-1]["robot"]["valid_navigation_actions"]["NavigateTo"][obj_ID])
            if obj_ID == oid:
                obj_rep["is_obstacle_object"] = False 
                obj_rep["reward_upon_removing_this_object"] = 100.0
            obj_desc.append({obj_ID: obj_rep})
        return obj_desc

    def get_san(self, exec_cost, reward, attained_goals:int, env_goal, prev_steps:dict, detour_pts:list, robot_pose:list, env_objs:list, inv_obj:str, cntrlr:Controller, save_json:bool=False, save_path:str="") -> None:
        curr_room_id, curr_room_type = self.get_point_room_id(robot_pose)

        goal_rep = []

        san = []
        curr_task = {}
        curr_task["goal"] = "Bring object type {} to receptacle type {}".format(env_goal[0], env_goal[1])
        san.append(curr_task)
        goal_rep.append(curr_task)

        progress = {}
        progress["progress"] = "completed {} out of total 20 tasks".format(attained_goals)
        san.append(progress)
        goal_rep.append(progress)

        rew = {}
        rew["reward_gained"] = reward
        san.append(rew)
        goal_rep.append(rew)

        cost = {}
        cost["total_cost"] = exec_cost
        san.append(cost)
        goal_rep.append(cost)

        tracker = {}
        tracker["room_visit_counter"] = self.validator.room_visits
        san.append(tracker)

        for obj in env_objs:
            if obj["objectType"] not in IGNORE_OBJS:
                try: obj["visible"] 
                except KeyError: continue # Ignore inventory objects
                if obj["visible"] and obj["objectId"] not in self.ever_visible_objs:
                    self.ever_visible_objs.append(obj["objectId"])

        if len(inv_obj) > 0:
            objs_affecting_action = [obj for obj in env_objs if obj["objectId"] != inv_obj[0]["objectId"]]
            env_objs = objs_affecting_action

        robot_pos_2 = {"x": round(robot_pose[0], 2), "z": round(robot_pose[2], 2)}
        t1 = time.time()
        if len(inv_obj) > 0:
            cntrlr_copy = None
            navigate_to, path_to_rooms = self.validator.valid_grid_actions(robot_pos_2, detour_pts, env_objs, self.ever_visible_objs, inv_obj, cntrlr_copy)
        else:
            navigate_to, path_to_rooms = self.validator.valid_grid_actions(robot_pos_2, detour_pts, env_objs, self.ever_visible_objs, inv_obj, None)
        t2 = time.time()

        robot = {}
        robot_attrs = {}
        robot_attrs["inventory"] = inv_obj
        robot_attrs["currently_in_room"] = curr_room_id
        robot_attrs["current_room_type"] = curr_room_type
        robot_attrs["valid_navigation_actions"] = navigate_to

        robot["robot"] = robot_attrs
        san.append(robot)

        if save_json:
            self.dump_json(save_path, san)
        self.san = san
        self.all_paths = path_to_rooms
        visible_desc = self.get_visible_obj_description(env_objs, env_goal)

        return visible_desc, path_to_rooms

    def get_visible_obj_description(self, env_objs:list, env_goal:list):
        all_seen_objs = []
        room_exp_info = {}
        for room in self.house["rooms"]:
            r_id = room["id"]
            room_info = {}
            SAN = deepcopy(self.san[-1]["robot"]["valid_navigation_actions"]["NavigateTo"][r_id])
            room_info["details"] = SAN
            room_info["times_scanned"] = self.validator.room_visits[r_id] #> 0
            room_exp_info[r_id] = room_info

        all_seen_objs.append({"room_exploration_status": room_exp_info})
        goal_obj = []
        target_obj = []
        for oid in self.ever_visible_objs: # Build description for every object discovered
            o_type = oid.split('|')[0]
            if o_type == "room":
                continue
            if o_type not in ["wall", "floor"]:
                if o_type == env_goal[0]:
                    goal_obj.append(oid)
                elif o_type == env_goal[1]:
                    target_obj.append(oid)

        if len(goal_obj) > 0:
            for oid in goal_obj:
                try:
                    self.get_blockers([oid])
                    SAN = self.san[-1]["robot"]["valid_navigation_actions"]["NavigateTo"][oid]
                    all_seen_objs.append({oid: {'discovered': True, "scene_description": SAN}})
                    self.goal_found = True
                except:
                    pass
        else:
            all_seen_objs.append({env_goal[0]: "Not found. Keep exploring"})

        if len(target_obj) > 0:
            for oid in target_obj:
                try:
                    self.get_blockers([oid])
                    SAN = self.san[-1]["robot"]["valid_navigation_actions"]["NavigateTo"][oid]
                    all_seen_objs.append({oid: {'discovered': True, "scene_description": SAN}})
                    self.target_found = True
                except:
                    pass
        else:
            all_seen_objs.append({env_goal[1]: "Not found. Keep exploring"})

        return all_seen_objs