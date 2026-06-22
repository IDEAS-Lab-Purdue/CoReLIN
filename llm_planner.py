import json 
from openai import OpenAI
from pathlib import Path

from env import mr_env
from params import H_LEN, results_path

API_KEY = Path("api_key.txt").read_text(encoding="utf-8", errors="replace")



class llm:
    def __init__(self, model_name:str, env_config_path:str, save_img:bool=False) -> None:
        self.llm_name = model_name
        self.i = 0
        self.client = OpenAI(api_key=API_KEY)
        self.env = mr_env(1, env_config_path, save_img, None)
        self.goal, self.goal_san = self.env.reset()
        task_desc = Path("prompts/system.txt").read_text(encoding="utf-8", errors="replace")
        self.task_msg = [{"role": "system", "content": task_desc}]
        output_rules = Path("prompts/developer.txt").read_text(encoding="utf-8", errors="replace")
        self.rule_msg = [{"role": "developer", "content": output_rules}]
        self.extra_inputs = []
        self.tools = [
            {
                "type": "function",
                "name": "scan_room",
                "description": "Do a 360 degree scan from your current position.",
                "parameters": {"type": "object",
                               "properties": {"reason": {"type": "string",
                                                         "description": "Why did you choose to call this tool?"}},
                               "required": ["reason"]
                              }
            },
            {
                "type": "function",
                "name": "go_to_room",
                "description": "Go to a room.",
                "parameters": {"type": "object",
                               "properties": {"room_ID": {"type": "string",
                                                          "description": "ID of the room you want to navigate into. Give the full ID, for example - as 'room|9'"},
                                              "reason": {"type": "string",
                                                         "description": "Why did you choose to call this tool?"}},
                               "required": ["room_ID", "reason"]
                              }
            },
            {
                "type": "function",
                "name": "get_path_data",
                "description": "Retrieve the characteristics of the path that leads to the specified goal object. This can only be the small pickupable object ID. Can be called only for one object. Please give ONLY the exact ID of the object, such as 'RemoteControl|surface|3|14_copy_0'",
                "parameters": {"type": "object",
                               "properties": {"goal_id": {"type": "string",
                                              "description": "Object ID of the goal object you want to move to."},
                                            },
                               "required": ["goal_id"]
                              }
            },
            {
                "type": "function",
                "name": "execute_plan",
                "description": "From the given object description, choose which objects to move out of your way and which objects to detour around in order to complete the given task. Also assign receptacles to each object that you want to move out of your way.",
                "parameters": {"type": "object",
                            "properties": {"avoid": {"type": "array",
                                                    "description": "List of object IDs of the objects you want to avoid interacting with.",
                                                    "items": {"type": "string"}},
                                            "interact": {"type": "array",
                                                         "description": "List of object IDs of the objects you want to interact with and remove from your path.",
                                                         "items": {"type": "string"}},
                                            "object_receptacle_pairs": {"type": "array",
                                                                    "description": "List of object IDs corresponding to each provided object. First entry of this list corresponds to the receptacle that the first object of provided object desciption should go to. Also include the target receptacle for the goal object that would lead to completion of the task.",
                                                                    "items": {"type": "object",
                                                                              "properties": {"object_ID": {"type": "string", "description": "Object ID of the blocker object"},
                                                                                             "receptacle_ID": {"type": "string", "description": "Object ID of the receptacle assigned to the blocker object"}
                                                                                             }
                                                                             },
                                                                      },
                                            "reason": {"type": "string",
                                                       "description": "Describe why you chose this specific assignment based on the costs involved."}
                                            },
                                "required": ["object_receptacle_pairs"]
                            }
            },
        ]
        self.prev_tool = None

    def record_output(self, to_write):
        with open("results/llm_outputs.txt", "a") as f:
            f.write(str(self.i))
            f.write("\n")
            f.write(str(to_write))
            f.write("\n")
            f.write("\n")

    def run_episode(self) -> None:
        extra_msg = [{"role": "user",
                        "content": [{"type": "input_text", "text": "Your next function calls can be - scan_room OR go_to_room OR get_obj_path_data"}]}]

        while not self.env.ep_done:
            self.i += 1
            if self.env.SR == 20: break

            try:
                msg = [{"role": "user", 
                        "content": [{"type": "input_text", "text": "Your task is - " + str(self.goal) + ". You have completed {} tasks out of total 20. The scene description of known parts of the scene is as follows - ".format(self.env.goals_achieved) + json.dumps(self.env.goal_san)},]
                        }]
                input_messages = self.task_msg + self.rule_msg + self.extra_inputs[-3*H_LEN:] + msg + extra_msg
                response = self.client.responses.create(model=self.llm_name,
                                                        tools=self.tools,
                                                        temperature=1.0,
                                                        input=input_messages,
                                                        parallel_tool_calls=False,
                                                        )
                self.extra_inputs += response.output

                for item in response.output:
                    if item.type == "function_call": 
                        if item.name == "scan_room":
                            args_dict = json.loads(item.arguments)
                            print("Executing - scan_room")
                            print(args_dict)
                            new_san = self.env.scan_room()
                            self.extra_inputs.append({
                                "type": "function_call_output",
                                "call_id": item.call_id,
                                "output": json.dumps(new_san)
                                })
                            if self.env.scene_data.goal_found and self.env.scene_data.target_found:
                                extra_msg = [{"role": "user",
                                            "content": [{"type": "input_text", "text": "Your next function calls can be - get_path_data"}]}]
                            else:
                                extra_msg = [{"role": "user",
                                            "content": [{"type": "input_text", "text": "Your next function calls can be - scan_room OR go_to_room"}]}]

                        if item.name == "go_to_room":
                            args_dict = json.loads(item.arguments)
                            print("Executing - go_to_room for ", args_dict["room_ID"])
                            print(args_dict)
                            try:
                                new_san = self.env.nav_to_room(args_dict["room_ID"])
                                self.extra_inputs.append({
                                    "type": "function_call_output",
                                    "call_id": item.call_id,
                                    "output": json.dumps(new_san)
                                    })
                                if self.env.scene_data.goal_found and self.env.scene_data.target_found:
                                    extra_msg = [{"role": "user",
                                                "content": [{"type": "input_text", "text": "Relevant objects found! Your next function calls can be - get_path_data"}]}]
                                else:
                                    extra_msg = [{"role": "user",
                                                "content": [{"type": "input_text", "text": "Your next function calls can be - scan_room OR go_to_room"}]}]
                            except:
                                self.extra_inputs.append({
                                    "type": "function_call_output",
                                    "call_id": item.call_id,
                                    "output": "Action failed. Please refer to the scene description to retry - " + json.dumps(self.env.goal_san)
                                    })

                        if item.name == "get_path_data":
                            self.consecutive_scans = 0
                            print("Executing - get_path_data")
                            args_dict = json.loads(item.arguments)
                            print(args_dict)
                            self.record_output(args_dict)
                            try:
                                obj_path_data = self.env.get_path_data(args_dict["goal_id"])
                                go_to_oid = args_dict["goal_id"]
                                self.extra_inputs.append({
                                    "type": "function_call_output",
                                    "call_id": item.call_id,
                                    "output": json.dumps(obj_path_data)
                                    })
                                if self.env.scene_data.goal_found and self.env.scene_data.target_found:
                                    extra_msg = [{"role": "user",
                                                "content": [{"type": "input_text", "text": "Your next function calls can be - execute_plan"}]}]
                                else:
                                    extra_msg = [{"role": "user",
                                                "content": [{"type": "input_text", "text": "You have not found the objects required to complete the task. Keep exploring! Your next function calls can be - go_to_room OR scan_room"}]}]
                            except:
                                self.extra_inputs.append({
                                    "type": "function_call_output",
                                    "call_id": item.call_id,
                                    "output": "Action failed. Please refer to the scene description to retry - " + json.dumps(self.env.goal_san)
                                    })

                        if item.name == "execute_plan":
                            print("Executing - execute_plan")
                            args_dict2 = json.loads(item.arguments)
                            print("Receps asgn - ", args_dict2)
                            self.record_output(args_dict2)
                            self.goal, self.goal_san = self.env.execute_plan(go_to_oid, args_dict2)
                            self.extra_inputs.append({
                                "type": "function_call_output",
                                "call_id": item.call_id,
                                "output": "Success: True, Message: TASK SUCCESSFUL!"})
                            extra_msg = [{"role": "user",
                                          "content": [{"type": "input_text", "text": "Your next function calls can be - scan_room OR go_to_room OR get_path_data"}]}]
                        self.prev_tool = item.name

            except KeyboardInterrupt:
                break 

        self.env.make_gif()
        self.env.write_results()
        self.env.end_simulation()





if __name__ == "__main__":
    dataset_path = "dataset/test/5/3.json"
    print(f"Running test of example 5-room environment!")
    test = llm("gpt-5-mini-2025-08-07", dataset_path, True) # Change to False if do not want to save images
    test.run_episode()

    with open(results_path + "/r_{}.csv".format(len(test.env.house["rooms"])), "a") as f:
        if test.env.obs_encountered == 0:
            div_by = 1.00
        else:
            div_by = test.env.obs_encountered
        perc_interacted = test.env.OI / div_by
        f.write("{:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}\n".format(test.env.SR, test.env.PL, test.env.OI, test.env.NI, perc_interacted, test.env.rel_bw, test.env.overall_poc_value, test.env.cost, test.env.steps))
        f.close()

    print("Done!")