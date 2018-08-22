
     
    
## Intelligent Traffic Signal Control

This Project is a traffic control system based on DQN [(arxiv:1312.5602)](https://arxiv.org/abs/1312.5602) on [Vissim](http://vision-traffic.ptvgroup.com/en-us/products/ptv-vissim/). It's an original implement that intelligent traffic signal control via deep reinforment learning on partial urban traffic net. Choosing fine hyper-parameters, agent could learn to how to improve the performance of global net in a long  term.

#### Dependencies
- Vissim 4.3.0
- Python 3.5
- Tensorflow 1.2.0
- other common packages like pandas numpy matplotlib pywin32
- ...

#### About Vissim
VisEnv.py wrapped the orignal api into the open.ai style. For now, speed, travel time, queued vehicles count interfaces are provided.
Use this like:
``` python
fron vis_env import * 
env = VisEnv()
...
for epi in range(episodes):
    env.reset()
    env.test = True
    for _ in range(steps):
        next_state, reward, done = env.step(action)
    env.write_summary(epi, dir)
```
#### Experiments
The performance of DQN is not so good among the series of reinforcement learning algorithm, but agent are still capble to act appropriately in our traffic enviroment.

![Queue](https://github.com/Linging/Traffic-Signal-Control/blob/master/images/Q_Mix%20Q.png)
![Travel Time](https://github.com/Linging/Traffic-Signal-Control/blob/master/images/T_Mix%20Q.png)

#### TODO
More reinforment learning models like dueling-DQN, DDPG, to further improve the performance of agent, and to solve the large discrete actions space problem.
Intelligent Traffic Signal Control



