import numpy as np
from win32com.client import Dispatch
import pandas as pd
import time


n_lanes = 8
n_cells = 60
frames = 2


class VisEnv():
    """docstring for VisEnv."""
    def __init__(self):
        self.dir = "./dqn.inp"

        self.n_cross = 4
        self.n_lanes = 8
        self.n_queued = 0

        self.reward_func= "dif_of_global_v"
        # self.reward_func = "mixed_q_v"
        self.mode = "fixed flow"
        self.flow_rate = 450

        self.error_summary = 0
        self.error_action = 0


        self.all_red = True
        self.reset()

    def reset(self):
        self.Vissim = Dispatch("Vissim.Vissim")
        self.Vissim.LoadNet(self.dir)
        self.Net = self.Vissim.Net
        self.links = self.Net.Links
        self.inputs = self.Net.VehicleInputs
        self.vehicles = self.Net.Vehicles
        self.signal_groups = [self.Net.SignalControllers(1).SignalGroups.GetSignalGroupByNumber(i+1) for i in range(2 * self.n_cross)]
        self.s_groups = self.reconstruct_signal_groups()

        self.test = False
        self.pre_v = 50
        self.pre_n_queued = 0
        self.pre_action = np.zeros(self.n_cross)
        self.summary = [[],[],[]]
        self.state = np.zeros([n_lanes,n_cells,frames])

    def reconstruct_signal_groups(self):
        return np.array(self.signal_groups).reshape(-1,2)

    def set_flow_rate(self, rate):
        for fr in self.inputs:
            fr.SetAttValue('VOLUME', rate)
        print("Set Successfully, Rate=",rate)

    def set_flow_mode(self):
        if self.mode == "fixed flow":
            self.set_flow_rate(self.flow_rate)
        else:
            pass

    def step(self, actions, steps=5):
        """"""
        if self.all_red:
            all_red_actions = np.zeros(self.n_cross)
            for i in range(self.n_cross):
                if self.pre_action[i] == actions[i]:
                    all_red_actions[i] = actions[i]
                else:
                    "all red time"
                    all_red_actions[i] = 2

            self.pre_action = actions

            self.action(all_red_actions)
            for _ in range(6):
                self.Vissim.Simulation.RunSingleStep()

        if self.action(actions):
            for _ in range(steps):
                self.Vissim.Simulation.RunSingleStep()
            self.next_state, self.current_v = self.get_state()
        else:
            print("sigal set failed!")

        self.n_queued = self.get_queued()
        reward = self.cal_reward()
        self.pre_n_queued = self.n_queued

        if self.test:
            self.trav_time = self.travel_time()
            self.record()

        return self.next_state, reward, self.down()

    def action(self, actions):
        try:
            for i in range(self.n_cross):
                self.turn_cross_signal(self.s_groups[i], actions[i])
            return True
        except:
            self.error_action += 1
            return False

    def turn_cross_signal(self, group, action):
        if action == 1:
            group[0].SetAttValue('type', 2)
            group[1].SetAttValue('type', 3)
        elif action == 0:
            group[1].SetAttValue('type', 2)
            group[0].SetAttValue('type', 3)

            "all red time"
        else:
            group[1].SetAttValue('type', 3)
            group[0].SetAttValue('type', 3)

    def down(self):
        if self.pre_n_queued >= 60:
            return True
        else:
            return False

    def cal_reward(self):
        if self.reward_func == "dif_of_global_v":
            reward = self.current_v - self.pre_v
            self.pre_v = self.current_v
            return reward + 1
        elif self.reward_func == "mixed_q_v":
            reward = (self.current_v - self.pre_v) + (self.n_queued - self.pre_n_queued)
            self.pre_v = self.current_v
            return reward

    def get_state(self):
        cell_length = 10
        state = np.zeros([n_lanes,n_cells,frames])
        lane = 0
        speed = []
        for link in self.links:
            vehicles = link.GetVehicles()
            for v in vehicles:
                try:
                    i = int(v.AttValue("TOTALDISTANCE")/cell_length)
                    if i < n_cells:
                        state[lane, i, 0] += 1
                        sp = v.AttValue("SPEED")
                        state[lane, i, 1] += (sp/10)
                        speed.append(sp)
                except:
                    print("ERROR---TOTALDISTANCE")
            lane += 1
        speed = np.array(speed).mean()
        return state, speed

    def get_queued(self):
        queued_vehicles = self.vehicles.GetQueued()
        return queued_vehicles.Count

    def travel_time(self):
        summary = []
        for i in range(self.n_lanes):
            tr = self.Net.TravelTimes(i+1)
            summary.append(tr.GetResult(600, "TRAVELTIME", "", 0))
        return np.array(summary).mean()

    def record(self):
        self.summary[0].append(self.pre_v)
        self.summary[1].append(self.pre_n_queued)
        self.summary[2].append(self.trav_time)

    def write_summary(self, episode, dir):
        summary = pd.DataFrame(self.summary).T
        # print(summary.shape)
        print("Test Phase performed, episode:", episode)
        print(summary.describe())
        summary.to_csv(dir + "/summary_epi_%d.csv" % episode, index=False)
