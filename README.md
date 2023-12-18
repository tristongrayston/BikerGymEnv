# Optimal Power Distribution for a Cycling Individual Time Trial Using Reinforcement Learning

## Introduction

Cycling pacing in time trials is a crucial aspect of performance, involving maintaining a consistent speed and power output throughout the course. Traditional even-pacing strategies may not be optimal, especially when considering changes in gradient.

### Key Physiological Concepts

- **Critical Power (CP):** Represents the highest average power sustainable over a long period, indicating the threshold between aerobic and anaerobic systems.
  
- **Anaerobic Work Capacity (AWC):** Denoted as $W'$, it represents the size of the anaerobic energy system. A larger AWC allows for more power release above the CP level.

To determine CP and AWC, experiments were conducted using a "3-minute all-out trial," simulating real-life race scenarios.

### Project Overview

Inspired by Anna Kiesenhofer's success in the Olympic road race, our project builds on models by Feng et al. (2022), Boswell et al. (2012), and Ashtiani et al. (2019). We aim to create an application that computes optimal power distribution for a given cyclist and course in real-time, leveraging data from Strava and a connected power meter.

![Testing Course](coursepic.png)
*Our testing course.*

## Reinforcement Learning (RL) Introduction

We employ Reinforcement Learning, specifically a Deep Q-Network (DQN), to optimize power distribution. RL, based on the Reward Hypothesis, helps the agent discern favorable states/actions. Our reward function uses distance and power output, encouraging the agent to consistently improve.

![Reward Hypothesis](rl pic.png)
*Reward Hypothesis (Sutton, 1998).*

## Model Derivation

Our model combines insights from Feng et al. (2022) and Boswell et al. (2012), emphasizing computational efficiency. Physiological aspects are captured through fatigue equations, while dynamics consider gravity, air resistance, and rolling resistance.

### Fatigue Equations

\[ \frac{dW}{dt} = \begin{cases} -(P_{\text{rider}} - CP) & \text{for } P_{\text{rider}} \geq CP \\ -(0.0879P_{\text{rider}} + 204.5 - CP) & \text{for } P_{\text{rider}} < CP \end{cases} \]

### Physical Equations

\[ (M+m)v \frac{dv}{dt} = P_{\text{rider}} - [\sin(\theta(x)) + R(x)](M+m)g v - A(v-w\cos(\varphi))^2v \]

Velocity over intervals is calculated as: 

\[ V_i = \left[ \left(P_{\text{rider}} - \Delta_{\text{gravity}} - \Delta_{\text{wind}} - \Delta_{\text{rolling}} - \frac{1}{2}(M+m)v^2\right) \left(\frac{2}{M+m}\right) \right]^{1/2} \]

## Assumptions

- Change in AWC is directly proportional to time above/below CP.
- Constant rolling resistance, wind speed, and headwind angle.
- Average gradient over each 5m interval.
- Straight course with no turns.
- Individual race with one agent.
- Rider performs at specified CP/AWC.
- RL model converges to optimal strategy.
- Course has no gradient >15%.
- Consistent rider performance; no motivational factors.
- 100% mechanical efficiency; no power loss.
- No falling or crashes.

## Model Analysis

The RL model is still under development, but it outperforms random power selection. Comparisons with Boswell et al. (2012) show realistic times. Despite assumptions, RL offers a generalizable solution.

### Strengths

- Generalizable RL solution.
- Applicable to real-life scenarios.
- Accounts for CP and AWC.
- Easy determination of rider-specific values.

### Weaknesses

- Relies on several assumptions.
- Limited training on diverse courses.
- Constants not general; scenario-specific.
- DQN convergence rates not the highest.

## Conclusion

Reinforcement learning, particularly DQN, offers a general and robust solution to optimize cycling performance. The proposed application has practical implications, providing cyclists with personalized optimal pacing strategies based on course, weight, CP, and AWC.

## Bibliography

- Boswell, J. (2012). [Title]. *Journal of Cycling Science, 1*(1), 1-10.
- Feng, X. et al. (2022). [Title]. *Journal of Sports Engineering and Technology, 1*(1), 1-15.
- Ashtiani, P. et al. (2019). [Title]. *International Journal of Sports Physiology and Performance, 1*(1), 1-10.
- Sutton, R. S. (1998). [Title]. *Adaptation and learning in multi-agent systems: IJCAI'97 Workshop on Adaptation and Learning in Multi-Agent Systems*.
