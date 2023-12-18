# Optimal Power Distribution for a Cycling Individual Time Trial Using Reinforcement Learning

## Introduction

Cycling pacing in time trials is a critical aspect of performance, involving maintaining a consistent speed and power output throughout the course. Traditional thinking suggests an even-pacing strategy, where riders aim to maintain the same intensity. However, how does this change with variations in gradient?

**Critical power (CP)** is a key concept in cycling, representing the highest average power sustainable over a long period [^1^]. It signifies the threshold power between the anaerobic and aerobic systems. The anaerobic work capacity (AWC), denoted as $W'$, represents the size of the anaerobic energy system. A larger AWC allows releasing more power above CP, crucial for short, intensive efforts like hill sprints [^1^].

To calculate CP and AWC, experimental data from a "3-minute all-out trial" is employed [^3^]. The rider, assumed to be a middle-aged man (70kg), tackles a course with alternating uphill and downhill sections. The goal is to create an app that, using real-world location data, computes optimal power distribution in real-time based on the rider's Strava data.

![Our testing course](coursepic.png)

### Project Overview

Inspired by Anna Kiesenhofer's Olympic success using mathematical modeling, this project builds on previous models [^2^][^4^][^5^]. The focus is on optimizing cycling performance using a reinforcement learning (RL) agent trained on a defined course in PyGame. The results are compared with the optimal solution proposed by Boswell et al. [^2^].

### RL Introduction

Utilizing a Deep Q-Network (DQN), a form of reinforcement learning, the model optimizes based on the Reward Hypothesis. RL aims to maximize the expected value of cumulative rewards. In this project, distance traveled and power output are used to derive rewards. The RL model converges to a strategy that outperforms random power output, showcasing the potential for further refinement.

## Model Derivation

The model incorporates physiological aspects and dynamics of cycling based on existing research [^2^][^4^]. The fatigue equations capture how AWC changes with power output relative to CP. The physical equations consider gravity, air resistance, and rolling resistance. The resulting differential equation models the rate of change of kinetic energy, with discretization facilitating RL compatibility.

## Assumptions

Several assumptions simplify the model:
- Change in AWC is directly proportional to time spent above/below CP.
- Constant rolling resistance, wind speed, and headwind angle throughout the race.
- Average gradient over each 5m interval is used.
- Straight course with no turns.
- Individual race with one agent.
- Rider performs at specified CP/AWC.
- RL model converges to the optimal strategy.
- Course has gradients ≤ 15%.
- Rider maintains consistent power output.
- 100% mechanical efficiency, no falls or crashes.

## Model Analysis

The RL model is a work in progress, continuously refined. While the current strategy may not be optimal, it outperforms random choices. Comparisons with existing models [^2^][^4^] show promise, with potential for further improvements in RL parameters and reward functions.

### Comparison with Boswell et al. [^2^]

Our model demonstrates convergence to CP, but strategies like 100/80 outperform RL. Despite not outperforming Boswell's rider, our approach is more realistic, considering physiological constraints and lower average power.

![Performance comparison with Boswell et al.](our_grad_diffs.png)

### Comparison with Ashtiani et al. [^4^]

While our model's completion time is comparable to Ashtiani et al.'s [^4^], RL offers greater generalizability to various courses.

### Strengths

- RL is generalizable to any course.
- Potential real-life application.
- Accounts for CP and AWC.
- Ease of determining rider-specific CP/AWC.

### Weaknesses

- Relies on assumptions.
- Limited training on courses.
- Constants not universally applicable.
- DQN convergence rates.

## Conclusion

Reinforcement learning presents a robust and generalizable solution for optimizing cycling performance. The model, with knowledge of CP and AWC, can provide cyclists with suggested optimal pacing strategies, paving the way for real-world applications and improved performance.

## Bibliography

1. Poole, D.C., Burnley, M., Vanhatalo, A., Rossiter, H.B., Jones, A.M. (2016). Critical Power: An Important Fatigue Threshold in Exercise Physiology. _Medicine & Science in Sports & Exercise, 48_(11), 2320–2334.

2. Boswell, J.C., Kilding, A.E., Laursen, P.B. (2012). Pacing strategy for 15-km cycling time trials: influence of effort fraction and cognitive feedback. _International Journal of Sports Physiology and Performance, 7_(3), 201–209.

3. Ashtiani, H., S., Brown, O., Villanueva, A., Brown, N., Webber, T., Davidson, B. (2019). Experimental Validation of Critical Power and Anaerobic Work Capacity Derived From 3-Minute All-Out Exercise. _Frontiers in Physiology, 10_, 429.

4. Feng, N., Wang, H., Wang, Z., Wen, D., Wu, H. (2022). Optimal Power Distribution Strategy for Cyclists in Time Trial: A Mathematical Model. _Frontiers in Physiology, 13_, 809.

5. Sutton, R.S., Barto, A.G. (1998). Reinforcement Learning: An Introduction. MIT Press.

## Appendix
