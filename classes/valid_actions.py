import numpy as np
import json
import time
import networkx as nx
from shapely.geometry import Point, Polygon
from typing import Dict, List, Tuple, Optional
from ai2thor.controller import Controller

from .grid_graph import reachable_graph

INTERACTION_DIST = 1.5



class find_valid_actions:
    def __init__(self, step_size:float, house_config:dict, manip_cost:float, free_pts:list) -> None:
        self.step_size = step_size
        self.house = house_config
        self.manipulation_cost = manip_cost
        self.grapher = reachable_graph(free_pts, self.step_size)
        self.graph, self.node_coords = self.grapher.build_graph()
        self.bw_cen = nx.betweenness_centrality(self.graph)
        self.room_visits = {}
        for room in self.house["rooms"]:
            self.room_visits[room["id"]] = 0

    def _aabb_xz(self, aabb: Dict) -> Tuple[float, float, float, float]:
        c = aabb["center"]; s = aabb["size"]
        hx, hz = s["x"] * 0.5, s["z"] * 0.5
        xmin, xmax = c["x"] - hx, c["x"] + hx
        zmin, zmax = c["z"] - hz, c["z"] + hz
        return xmin, xmax, zmin, zmax

    def _inflate_interval(self, minv: float, maxv: float, padding: float) -> Tuple[float, float]:
        return minv - padding, maxv + padding

    def _point_inside_rect(self, x: float, z: float, rect: Tuple[float, float, float, float]) -> bool:
        xmin, xmax, zmin, zmax = rect
        return (xmin <= x <= xmax) and (zmin <= z <= zmax)

    def _is_free(self, self_oid:str, x: float, z: float, objects: List[Dict], agent_radius: float, scene_bounds: Optional[Dict], ignore_obj_ID:str=None) -> bool:
        padding = 0.15 # 1e-4
        occupying_objects = []
        objects = [obj for obj in objects if obj["objectId"] != self_oid]
        for obj in objects:
            if obj["objectId"].split('|')[0] not in ["wall", "door"]:
                if ignore_obj_ID is not None:
                    if obj["objectId"] == ignore_obj_ID: continue
                if obj["pickupable"]:
                    aabb = obj.get("axisAlignedBoundingBox")
                    xmin, xmax, zmin, zmax = self._aabb_xz(aabb)
                    pad = agent_radius + padding
                    xmin, xmax = self._inflate_interval(xmin, xmax, pad)
                    zmin, zmax = self._inflate_interval(zmin, zmax, pad)
                    if self._point_inside_rect(x, z, (xmin, xmax, zmin, zmax)):
                        occupying_objects.append(obj)

        if len(occupying_objects) != 0:
            if len(occupying_objects) > 1:
                for obj in occupying_objects:
                    if not obj["pickupable"]:
                        return [obj["objectId"]], False
            return [occupying_objects[0]["objectId"]], False

        return occupying_objects, True

    def get_point_room_id(self, pt:list):
        p = Point(pt[0], pt[1])
        for room in self.house["rooms"]:
            poly = Polygon([(v["x"], v["z"]) for v in room["floorPolygon"]])
            if poly.contains(p):
                return room["id"]
        return None

    def find_detour_coord_idx(self, coord:np.ndarray, robot_rid:str=None) -> int:
        obj_rid = self.get_point_room_id(coord)
        dists = np.linalg.norm(self.detour_coords - coord, axis = 1)
        sorted_indices = np.argsort(dists)
        for each_idx in sorted_indices:
            each_coord = self.detour_coords[each_idx]
            coord_rid = self.get_point_room_id(each_coord)
            if coord_rid == obj_rid: # Found coord must be in the same room as the object
                return self.find_coord_idx(each_coord)

    def find_coord_idx(self, coord:np.ndarray, robot_rid:str=None) -> int:
        obj_rid = self.get_point_room_id(coord)
        dists = np.linalg.norm(self.node_coords - coord, axis = 1)
        sorted_indices = np.argsort(dists)
        for each_idx in sorted_indices:
            each_coord = self.node_coords[each_idx]
            coord_rid = self.get_point_room_id(each_coord)
            if coord_rid == obj_rid: # Found coord must be in the same room as the object
                return each_idx

    def find_path_to_room(self, curr_pose:list, obj_pos:dict, env_objs:dict) -> list:
        rooms_crossed = []
        curr_node = self.find_coord_idx([curr_pose["x"], curr_pose["z"]])
        pt_idx = self.find_coord_idx(obj_pos)
        path = nx.shortest_path(self.graph, str(curr_node), str(pt_idx))
        for idx in path[:-1]:
            r_id = self.get_point_room_id(self.node_coords[int(idx)])
            if r_id not in rooms_crossed:
                rooms_crossed.append(r_id)
        return rooms_crossed

    def find_path_to_recep(self, ignore_obj_ID:str, curr_pose:list, obj_pos:dict, env_objs:dict, verbose:bool=False) -> tuple:
        curr_node = self.find_coord_idx([curr_pose["x"], curr_pose["z"]], None)
        robot_rid = self.get_point_room_id([curr_pose["x"], curr_pose["z"]])
        pt_idx = self.find_coord_idx(obj_pos, robot_rid)
        shortest_path = nx.shortest_path(self.graph, str(curr_node), str(pt_idx))

        try:
            detour_path = nx.shortest_path(self.detour_graph, str(curr_node), str(pt_idx))
            dp_exists = True
        except:
            dp_exists = False
            detour_path = []

        free_sp = []
        blockers = []
        sp_exists = True
        for idx in shortest_path[:-3]:
            xn, zn = self.node_coords[int(idx)]
            _, is_free = self._is_free(xn, zn, env_objs, 0.2, None)
            if is_free:
                free_sp.append(idx)
            else:
                sp_exists = False
                break

        for idx in shortest_path[:-3]:
            xn, zn = self.node_coords[int(idx)]
            sp_blockers, is_free = self._is_free(xn, zn, env_objs, 0.2, None, ignore_obj_ID)
            if len(sp_blockers) > 0:
                for oid in sp_blockers:
                    if oid not in blockers:
                        blockers.append(oid)

        return sp_exists, free_sp, blockers, dp_exists, detour_path

    def find_path_to(self, oid, curr_pose:dict, obj_pos:list, env_objs:dict, verbose:bool=False) -> tuple:
        curr_node = self.find_coord_idx([curr_pose["x"], curr_pose["z"]], None)
        robot_rid = self.get_point_room_id([curr_pose["x"], curr_pose["z"]])
        pt_idx = self.find_coord_idx(obj_pos, robot_rid)
        shortest_path = nx.shortest_path(self.graph, str(curr_node), str(pt_idx))

        try:
            pt_idx = self.find_detour_coord_idx(obj_pos, robot_rid)
            detour_path = nx.shortest_path(self.detour_graph, str(curr_node), str(pt_idx))
            dp_exists = True
        except:
            dp_exists = False
            detour_path = []

        free_sp = []
        blockers = []
        sp_exists = True
        if oid != "room":
            for idx in shortest_path[:-2]:
                xn, zn = self.node_coords[int(idx)]
                _, is_free = self._is_free(oid, xn, zn, env_objs, 0.2, None)
                if is_free:
                    free_sp.append(idx)
                else:
                    sp_exists = False
                    break
        else:
            for idx in shortest_path:
                xn, zn = self.node_coords[int(idx)]
                _, is_free = self._is_free(oid, xn, zn, env_objs, 0.2, None)
                if is_free:
                    free_sp.append(idx)
                else:
                    sp_exists = False
                    break

        if oid != 'room':
            for idx in shortest_path[:-2]:
                xn, zn = self.node_coords[int(idx)]
                sp_blockers, is_free = self._is_free(oid, xn, zn, env_objs, 0.2, None)
                if len(sp_blockers) > 0:
                    for obj_id in sp_blockers:
                        if obj_id not in blockers:
                            blockers.append(obj_id)
        else:
            for idx in shortest_path:
                xn, zn = self.node_coords[int(idx)]
                sp_blockers, is_free = self._is_free(oid, xn, zn, env_objs, 0.2, None)
                if len(sp_blockers) > 0:
                    for obj_id in sp_blockers:
                        if obj_id not in blockers:
                            blockers.append(obj_id)
        return sp_exists, free_sp, blockers, dp_exists, detour_path

    def find_coord_to_room_via_door(self, curr_rid, rid, curr_pose:dict, env_objs:dict, verbose:bool=False) -> tuple:
        needed_door = None
        if curr_rid == rid: return None
        for door in self.house["doors"]:
            r1 = door["room0"]
            r2 = door["room1"]
            if curr_rid == r1 and rid == r2:
                needed_door = door 
                break
            if curr_rid == r2 and rid == r1:
                needed_door = door 
                break

        if needed_door is not None:
            door_pos = [round(needed_door["assetPosition"]["x"], 2), round(needed_door["assetPosition"]["z"], 2)]
        else:
            door_pos = None

        return door_pos
    
    def get_self_bw_value(self, obj_pos:list, robot_rid:str=None):
        obj_idx = self.find_coord_idx(obj_pos, robot_rid)
        return self.bw_cen[str(obj_idx)]

    def valid_grid_actions(self, curr_pos:dict, detour_pts:list, env_objs:list, seen_obj_IDs:list, inv_obj:list, cntrlr:Controller) -> List[str]:
        all_paths = {}
        self.detour_graph, self.detour_coords = self.grapher.build_pruned_graph(detour_pts)

        valid_nav = {}
        nav_dict = {}

        curr_coord = [curr_pos["x"], curr_pos["z"]]
        robot_rid = self.get_point_room_id(curr_coord)

        # Needs to be here coz objects are moved around
        room_coords = {}
        for room in self.house["rooms"]:
            all_paths[room["id"]] = {}
            room_coords[room["id"]] = []

        for coord in self.node_coords:
            coord_rid = self.get_point_room_id(coord)
            blocker_id, is_free = self._is_free(coord_rid, coord[0], coord[1], env_objs, 0.2, None)
            if is_free:
                room_coords[coord_rid].append(coord)

        nav_dict[robot_rid] = {"can_navigate": True,
                               "reason": "You are currently here!",
                                "type": room["roomType"],
                               }

        for room in self.house["rooms"]:
            coords_in_room = np.array(room_coords[room["id"]]).reshape(-1, 2)
            avg_x, avg_z = coords_in_room.mean(axis=0)
            sp_exists, free_sp, blockers, dp_exists, detour_path = self.find_path_to("room", curr_pos, [avg_x, avg_z], env_objs)
            if sp_exists: all_paths[room["id"]]["middle"] = free_sp 
            elif dp_exists: all_paths[room["id"]]["middle"] = detour_path

            rooms_crossed = self.find_path_to_room(curr_pos, [avg_x, avg_z], env_objs)
            if self.room_visits[room["id"]] == 0:
                if len(rooms_crossed) >= 3:
                    nav_dict[room["id"]] = {"can_navigate": False, 
                                            "reason": "Never discovered this room! Visit and explore {} to enable navigation!".format(rooms_crossed[1]),
                                            "type": room["roomType"],
                                            }
                    continue
            else:
                if len(rooms_crossed) >= 3:
                    nav_dict[room["id"]] = {"can_navigate": False, 
                                            "reason": "Visit {} to enable navigation to this room.".format(rooms_crossed[1]),
                                            "type": room["roomType"],
                                            }
                    continue

            door_coord = self.find_coord_to_room_via_door(robot_rid, room["id"], curr_pos, env_objs)
            if door_coord is not None:
                dists = np.linalg.norm(coords_in_room - np.array(door_coord).reshape(1, 2), axis = 1)
            else:
                dists = np.linalg.norm(coords_in_room - np.array(curr_coord).reshape(1, 2), axis = 1)
            sorted_indices = np.argsort(dists)

            for i in range(1, len(sorted_indices)):
                idx = sorted_indices[-i]
                coord_ = coords_in_room[idx]
                sp_exists, free_sp, blockers, dp_exists, detour_path = self.find_path_to("room", curr_pos, coord_, env_objs)
                if sp_exists: 
                    all_paths[room["id"]]["longest"] = free_sp
                    break
                elif dp_exists:
                    all_paths[room["id"]]["longest"] = detour_path
                    break

            for idx in sorted_indices:
                coord_ = coords_in_room[idx]
                sp_exists, free_sp, blockers, dp_exists, detour_path = self.find_path_to("room", curr_pos, coord_, env_objs)
                if sp_exists: 
                    all_paths[room["id"]]["first_time"] = free_sp
                    break
                elif dp_exists:
                    all_paths[room["id"]]["first_time"] = detour_path
                    break

            if room["id"] != robot_rid:
                rid = room["id"]
                closest_coord = coord_
                sp_exists, free_sp, blockers, dp_exists, detour_path = self.find_path_to("room", curr_pos, closest_coord, env_objs)
                rooms_crossed = self.find_path_to_room(curr_pos, closest_coord, env_objs)

                if sp_exists:
                    nav_dict[room["id"]] = {"can_navigate": True, 
                                            "reason": "Path is unobstructed",
                                            "type": room["roomType"],
                                            }
                    all_paths[room["id"]]["shortest"] = free_sp
                else:
                    sp_blocked_by_discovered_objs = True
                    for blocker in blockers:
                        if blocker not in seen_obj_IDs:
                            sp_blocked_by_discovered_objs = False
                            break # No need to check further
                    if sp_blocked_by_discovered_objs:
                        if dp_exists:
                            nav_dict[rid] = {"can_navigate": True,
                                            "reason": "Detour path exists and shortest path can be cleared",
                                            "type": room["roomType"],
                                            }
                            all_paths[room["id"]]["detour"] = detour_path
                        else:
                            nav_dict[rid] = {"can_navigate": False,
                                            "reason": "Room not navigable.",
                                            "type": room["roomType"],
                                            }
                    else: # No idea of the blockers. Needs further exploration
                        nav_dict[rid] = {"can_navigate": False,
                                        "type": room["roomType"],
                                        "reason": "Need to explore the {} further to discover a path to the room".format(rooms_crossed[0]),
                                        }
        with open('all_paths.json', "w", encoding="utf-8") as f:
            json.dump(all_paths, f, indent=2)





        for oid in seen_obj_IDs:
            if len(inv_obj) > 0 and oid == inv_obj[0]["objectId"]:
                continue
            if oid.split('|')[0] in ["wall", "window", "door", "room"]:
                continue

            all_paths[oid] = {}
            rel_obj = [obj for obj in env_objs if obj["objectId"] == oid]
            obj = rel_obj[0]
            obj_pos = [obj["position"]["x"], obj["position"]["z"]]

            try:
                sp_exists, free_sp, sp_blockers, dp_exists, detour_path = self.find_path_to(oid, curr_pos, obj_pos, env_objs, False)
            except:
                continue

            all_paths[oid]["shortest"] = free_sp
            all_paths[oid]["detour"] = detour_path
            self_bw = self.get_self_bw_value(obj_pos, robot_rid)

            if sp_exists:
                nav_dict[oid] = {"shortest_path_exists": True,
                                 "path_blockers": [],
                                 "shortest_path_cost": len(free_sp),
                                 "is_obstacle_object": True if obj["parentReceptacles"] is not None and len(obj["parentReceptacles"]) == 1 and obj["parentReceptacles"][0].split('|')[0] == "Floor" and not obj["receptacle"] else False,
                                 "cost_to_remove_this_object": len(free_sp) + 2*self.manipulation_cost,
                                 "percentage_paths_freed_upon_removing_this_object": 100*round(self_bw, 2),
                                 }
            elif dp_exists:
                nav_dict[oid] = {"shortest_path_exists": False,
                                 "path_blockers": sp_blockers,
                                 "shortest_path_cost": len(free_sp),
                                 "is_obstacle_object": True if obj["parentReceptacles"] is not None and len(obj["parentReceptacles"]) == 1 and obj["parentReceptacles"][0].split('|')[0] == "Floor" and not obj["receptacle"] else False,
                                 "cost_to_remove_this_object": len(free_sp) + 2*self.manipulation_cost,
                                 "percentage_paths_freed_upon_removing_this_object": 100*round(self_bw, 2),
                                 "detour_path_exists": True,
                                 "detour_path_cost": len(detour_path)}
            else:
                nav_dict[oid] = {"shortest_path_exists": False,
                                 "path_blockers": sp_blockers,
                                 "shortest_path_cost": len(free_sp),
                                 "is_obstacle_object": True if obj["parentReceptacles"] is not None and len(obj["parentReceptacles"]) == 1 and obj["parentReceptacles"][0].split('|')[0] == "Floor" and not obj["receptacle"] else False,
                                 "cost_to_remove_this_object": len(free_sp) + 2*self.manipulation_cost,
                                 "percentage_paths_freed_upon_removing_this_object": 100*round(self_bw, 2),
                                 "detour_path_exists": False}

        valid_nav["NavigateTo"] = nav_dict
        return valid_nav, all_paths
