import numpy as np
from win32com.client import Dispatch
import pandas as pd
import random

n_lanes = 8
n_cells = 60
frames = 2
# from PIL import Image


class VisEnv():
    """docstring for VisEnv."""
    def __init__(self):
        self.dir = "D:\\Vissim\\Example\\dqn.inp"

        self.n_cross = 4
        self.n_lanes = 8
        self.n_queued = 0

        # self.reward_func= "dif_of_global_v"
        # self.reward_func = "mixed_q_v"
        # self.reward_func = "absolute_mix_q_v"
        # self.reward_func = "simple"
        self.reward_func = "sparse"
        self.mode = "fixed flow"

        self.flow_rate = 400

        self.counter = 0
        self.error_action = 0


        self.all_red = True
        self.reset()

    def reset(self):
        self.Vissim = Dispatch("Vissim.Vissim")
        self.Vissim.LoadNet(self.dir)
        self.simulation = self.Vissim.Simulation
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
        # self.state = np.zeros([n_lanes,n_cells,frames])
        '''Return last 4 frames'''
        self.state = [np.zeros([84,84]) for _ in range(4)]
        self.init_state = np.zeros([84,84,4])
        self.random_seed()

    def random_seed(self, seed = 7):
        self.simulation.RandomSeed = seed

    def reconstruct_signal_groups(self):
        return np.array(self.signal_groups).reshape(-1,2)

    def set_flow_rate(self, rate):
        for fr in self.inputs:
            fr.SetAttValue('VOLUME', rate)
        print("Set Successfully, Rate =",rate)

    def set_flow_mode(self):
        if self.mode == "fixed flow":
            self.set_flow_rate(self.flow_rate)
        else:
            pass

    def all_red_time(self, actions, sec=5):
        all_red_actions = np.zeros(self.n_cross)
        for i in range(self.n_cross):
            if self.pre_action[i] == actions[i]:
                all_red_actions[i] = actions[i]
            else:
                "all red time"
                all_red_actions[i] = 2

        self.pre_action = actions

        self.action(all_red_actions)
        for _ in range(sec):
            self.Vissim.Simulation.RunSingleStep()

    def step(self, actions, steps=5):
        """All red time for clear the conflict area"""
        if self.all_red:
            self.all_red_time(actions)

        if self.action(actions):
            for _ in range(steps):
                self.Vissim.Simulation.RunSingleStep()

            # self.next_state, self.current_v, valid = self.get_state()
            self.next_state, self.current_v, valid = self.get_state_atari_style()

            # unstable Vissim will give nan data sometimes.
            if not valid:
                print("\nWarning: unexpected break!!!\n")
                self.current_v = -1

        else:
            print("sigal set failed!")

        reward = self.cal_reward()

        if self.test:
            self.n_queued = self.get_queued()
            self.pre_n_queued = self.n_queued
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

    def action_sample(self):
        return [random.randint(0,1) for i in range(self.n_cross)]

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
        if self.pre_v <= 20:
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
        elif self.reward_func == "absolute_mix_q_v":
            ad_q = - self.n_queued
            self.pre_v = self.current_v
            reward = ad_q
            if np.isnan(reward):reward = -30
            return float(reward)
        elif self.reward_func == 'simple':
            self.pre_v = self.current_v
            return self.pre_v / 100
        elif self.reward_func == 'sparse':
            self.pre_v = self.current_v
            return 0.

    def get_state(self):
        cell_length = 10
        state = np.zeros([n_lanes,n_cells,frames])
        lane = 0
        speed = []
        error = 0
        valid = True
        for link in self.links:
            vehicles = link.GetVehicles()

            for v in vehicles:
                # vissim is shit.
                try:
                    poi, s = v.AttValue("TOTALDISTANCE"), v.AttValue("SPEED")
                except:
                    error += 1
                    break
                if np.isnan(poi) or np.isnan(s):
                    break
                else:
                    i = int(poi/cell_length)
                    if i < n_cells and i >= 0:
                        state[lane, i, 0] += 1
                        state[lane, i, 1] += (s/10)
                        speed.append(s)

            lane += 1

        if len(speed) == 0:
            speed = 0
        else:
            speed = np.array(speed).mean()
            # avoid strange big value.
            if speed > 60 or speed < 0:
                speed = 0
                error += 5
        if error >= 5:
            valid = False
        return state, speed, valid

    def get_state_atari_style(self):
        new_frame, speed = self.get_new_frame_atari_style()
        # img = Image.fromarray(new_frame * 50)
        # img = img.convert('L')
        # img.save('./logs/'+str(self.counter)+'.png',"PNG")
        self.state.append(new_frame)
        self.state.pop(0)
        valid = True
        state = np.array(self.state).transpose((1,2,0))
        return state, speed, valid

    def get_new_frame_atari_style(self):
        new_frame = np.zeros([84,84])
        temp = np.zeros([8,75])
        cell_length = 10
        n_cells = 75
        lane = 0
        speed = []
        self.counter += 1


        for link in self.links:
            vehicles = link.GetVehicles()

            for v in vehicles:
                # vissim is shit.
                try:
                    poi, s = v.AttValue("TOTALDISTANCE"), v.AttValue("SPEED")
                except:
                    break
                if np.isnan(poi) or np.isnan(s):
                    break
                else:
                    i = int(poi/cell_length)
                    if i < n_cells and i >= 0:
                        if lane in [1,7,3,5]:
                            i = 74 - i
                        temp[lane, i] += 1
                        speed.append(s)
            lane += 1
        '''Avg of speed'''
        if len(speed) == 0:
            speed = 0
        else:
            speed = np.array(speed).mean()
            # avoid strange big value.
            if speed > 60 or speed < 0:
                speed = 0

        '''Rebuild the state frame'''

        for lane, target in zip([1,0,7,6],[20,21,50,51]):
            new_frame[target][:75] = temp[lane]
        new_frame = new_frame.T
        for lane, target in zip([2,3,4,5],[20,21,50,51]):
            new_frame[target][:75] = temp[lane]
        new_frame = new_frame.T
        return new_frame, speed

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
