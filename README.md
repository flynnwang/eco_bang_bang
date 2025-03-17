# EcoBangBang's Approach - Yet Another RL Solution

I'd like to thank @Stone for creating yet another season of fascinating game environment in the Lux S3 competition. I enjoyed it very much in the three months! Also, I want to thank @IsaiahP for open-sourcing the RL solution three years ago. My solution is mainly based on your work. Plus, I'm glad to see some of the old faces, really feel home here.

Time to give it back.

# My journey

I decided to play with the RL algorithm in this competition the first day I joined the game. One thing I'm curious about is how well could an  RL algorithm perform in such an incomplete information game.

There are three stages in my journey:

1. In the beginning, I tried RL but not quite working
2. In the middle, I switched to a rule-based agent
3. Finally, I tried RL again and it finally worked!

In the first stage, I planned to test a move-action-only agent to have the RL algorithm up and running. I tried all kinds of reward-shaping ideas during that time, and the best agent I made was < 700 ranking score. The agent at the time had some ideas about moving to a relic cell but was not quite good at staying on it. So it failed often when competing with 800+ agents.

Because RL kept failing in the first 50+ days, I switched to a rule-based one. The rule-based agent tries to maximize the total sum of the score between each unit and cell. Every agent and cell combination is scored based on different tasks: exploration, fuel energy, visiting a relic neighbor, staying on a relic cell, or SAP at an enemy unit. It's always fun to write a rule-based agent! It climbed up to about 1500+ at that time then plateaued. As a side effect of building this rule-based agent, I have a bunch of bugs fixed for some of the feature maps.

Then I took another try with RL, and with some luck, it finally started working 16 days before the deadline. and went up to 1400+ in a few days but then plateaued. 10 days before the deadline, I decided to drop the upgo pg loss in the original implementation because of its large magnitude in the total loss, and the agent started climbing up again all the way to 1900+.

I'm glad it works before too late. Below I'll share some of my ideas and learnings.

# Implementation details

## The model, reward function, and training process

I took the convolutional ResNet as the backbone with all the feature maps as input (see below), and used the last feature vector at the specific (x, y) location, concatenated it with the unit energy at that location to produce the actions for that unit. The idea is to have all the units controlled by a single model so that they can share information and cooperate,  while still can take different actions based on different energy values despite on the same location.

The action space is (16, 230), where for each unit, it can select from 5 moves actions and 225 possible sap locations. The action space is quite large, but with proper action masking, it works.

For the training algorithm, I used vtrace with teacher KL loss. To make the model easier, I mirrored the data for the left-top corner.

On the reward function, two reward functions were used: match win-loss and game win-loss. In the warmup training phase, the match win-loss reward was used, which simply mimics the win-loss that happened at the end of each match. What makes the algorithm start working is self-play with the zero-sum output of the baseline model: [see here](https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021/blob/main/lux_ai/nns/models.py#L153). In short, it's just not enough (at least in this game) to make the reward function zero-sum.

With some feature maps cleaned up, I started a second round of training from scratch and took the model from the warmup phrase as its teacher. Only later replaced the teacher with the latest snapshot of itself. Starting from the second round, I switched to the game win-loss reward. Also, in this round of training, I dropped the UPGO loss. IIUC, The UPGO pg loss tries to only learn from the positive signal. And from my dashboard, it's an order of magnitude larger than the baseline loss and a few others. I took a guess and dropped it. It then worked pretty well, surprisingly.

I did a third round of training by adding a few more feature maps, and the last two submissions are from the snapshots in this round shortly before the submission deadline.

## Feature Maps

The full set of feature maps is [here](https://github.com/flynnwang/eco_bang_bang/blob/75cd3c1bfbf34872a5a787b7f5447d4c74a322ca/ebb/env/luxenv.py#L31). I just quickly summarise them below:

- Game parameters: these are scalar values and I simply use an entire layer of a same value
  - Game step and match step
  - All game params: e.g. unit_move_cost, unit_sensor_range, etc and a few inference ones like nebula_vision_reduction and nebula_energy_reduction etc
  - Team points and team wins: e.g. Number of team points in the last step and the number of team points higher than the enemy team
- Map features: these are the ones that are locations specific and naturally represented as a map of (24, 24) values:
  - All the map info: cell type, cell energy, nebula, visible, relic neighbor, team point layer, etc. For cell type, I further included cell type in 3 future steps
  - Team and enemy energy and location layers, which I included info from the last 2 steps

## Detect hidden relic cells

I tried two approaches to detect the hidden relic cells: a Bayes-based one and the other uses linear equations.

The [Bayes-based approach](https://github.com/flynnwang/eco_bang_bang/blob/75cd3c1bfbf34872a5a787b7f5447d4c74a322ca/ebb/env/mapmanager.py#L720) tries to model each relic neighbor cell as an independent binary random variable with an a priori probability of 0.25, and uses the observation of **m** units on relic neighbor cells and **n** team points collected from each step, to get its posterior probability:

```Bash
P(i): priori of the i-th observed relic neighbor being a relic cell 

P(n|m): prob of collecting n team points, given m units sit on relic neighbors

P(n|m, i=relc): given the i-th relic neighbor is relic position, prob of collecting n team points from m units sitting on relic neighbor cells

P(i=relic|n, m) = P(n|m, i=relc) * P(i=relic) / P(n|m)
```

The Bayes-rule approach is robust to noise given that you limit the posterior between (0, 1) (by excluding 0 and 1), but somehow slower to converge to the true values because it does not use past information directly. That's why I tried the second approach to model each observation as a linear equation with binary variables.

Now with observations at past steps, one can try to solve the set of equations and recover the relic cell positions. I believe this approach works faster, but it's sensitive to noise. E.g. If a unit collects a team point from a relic neighbor cell that hasn't been discovered yet, those equations will break. One simple workaround is to limit the equations to the last few ones.

## Energy node and energy field

Energy is important in the game, and so is the energy field. If you walk through the Lux code, you will find that only one energy node (the other one is mirrored) (or none) is used at any step. Then the solution is straightforward: compute the energy field at all possible positions and match with them. 

Another thing for the energy node is, that it is only known where it is before the next drift, since the next location is determined from a randomly selected nearby position.

## Other hidden game parameters

Nebula drift speed. This one should be important since it helps to reuse past map cell information and to not mislead the agent with obsolete map info. It could be found out by matching the drift step with each parameter.

Nebula energy reduction. This one could be easily discovered when a unit moves onto a nebula cell and must be useful since an energy reduction of 25 is huge.

Also added the energy void field factor and sap dropoff factor estimator, but these two are more noisy, not sure whether they helped or not.
