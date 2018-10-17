# Navigation Project

Note : Reward was capped at 14 for all algorithms

## 1) Baseline : Vanilla DQN

| Reward Progression                                           |                                                              |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![](/home/shashank/deep-reinforcement-learning/p1_navigation/Report/dqn_tau_2.png) | ![](/home/shashank/deep-reinforcement-learning/p1_navigation/Report/dqn_tau_1.png) |

DQN Architecture:

Performance in Testing mode: 

## 2) Double DQN

| Reward Progression                                           |                                                              |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![](/home/shashank/deep-reinforcement-learning/p1_navigation/Report/DDQN.png) | ![DDQN_2](/home/shashank/deep-reinforcement-learning/p1_navigation/Report/DDQN_2.png) |

| DQN without soft update             | Double DQN without soft update         |
| ----------------------------------- | -------------------------------------- |
| ![](/home/shashank/Desktop/DQN.png) | ![](/home/shashank/Desktop/DDQN_1.png) |



**<u>What's different</u>**? : The action chosen for target fixed Q value and the evaluation of Q-Values for that action are done **<u>separately</u>** by the Local and Target network , unlike in vanilla DQN where only the Target Net is used.

We can see that in the initial stages of training, Double DQN is more stable and gradual unlike Vanilla DQN.

The difference is hardly appreciable. But, if we take $\tau​$ =1 and perform hard update on target network every 10 episodes, ie; disabling soft-update, the performance difference is noticeable. Double DQN reaches +14 average reward first, about 100 epsiodes before DQN. 

**<u>In my opinion</u>**, Soft updating takes away the effect of Double DQN in case of this environment. 

**<u>Further study</u>** can  be done to check the effect of Double DQN on custom or much more environments. Maybe it has more effect in more complex environments.

## 3)Dueling DQN

​	

## 4)Prioritized Experience Replay