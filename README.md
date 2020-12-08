# Automated Stock Trading Using Deep Reinforcement Learning 

Profitable automated stock trading strategy is vital to investment companies and hedge funds. It is applied to optimize capital allocation and maximize investment performance, such as expected return. Return maximization can be based on the estimates of potential return and risk. However, it is challenging to design a profitable strategy in a complex and dynamic stock market. 

### Following Paper uses a Ensemble Deep Reinforcement Learning Trading Strategy 

[An Application of Deep Reinforcement Learning to Algorithmic Trading](https://arxiv.org/pdf/2004.06627.pdf)

### The strategy includes three actor-critic based algorithms:
1. Proximal Policy Optimization (PPO)
2. Advantage Actor Critic (A2C)
3. Deep Deterministic Policy Gradient (DDPG)

It combines the best features of the three algorithms, thereby robustly adjusting to different market conditions. 

The performance of the trading agent with different reinforcement learning algorithms is evaluated using Sharpe ratio and compared with both the Dow Jones Industrial Average index and the traditional min-variance portfolio allocation strategy. 



### The advantages of deep reinforcement learning
1. Deep reinforcement learning algorithms can outperform human players in many challenging games. For example, on March 2016, DeepMind’s AlphaGo program, a deep reinforcement learning algorithm, beat the world champion Lee Sedol at the game of Go.
2. Return maximization as trading goal: by defining the reward function as the change of the portfolio value, Deep Reinforcement Learning maximizes the portfolio value over time.
3. The stock market provides sequential feedback. DRL can sequentially increase the model performance during the training process.
4. The exploration-exploitation technique balances trying out different new things and taking advantage of what’s figured out. This is difference from other learning algorithms. Also, there is no requirement for a skilled human to provide training examples or labeled samples. Furthermore, during the exploration process, the agent is encouraged to explore the uncharted by human experts.
5. Experience replay: is able to overcome the correlated samples issue, since learning from a batch of consecutive samples may experience high variances, hence is inefficient. Experience replay efficiently addresses this issue by randomly sampling mini-batches of transitions from a pre-saved replay memory.
6. Multi-dimensional data: by using a continuous action space, DRL can handle large dimensional data.
7. Computational power: Q-learning is a very important RL algorithm, however, it fails to handle large space. DRL, empowered by neural networks as efficient function approximator, is powerful to handle extremely large state space and action space.
  


### Data
We track and select the Dow Jones 30 stocks (at 2016/01/01) and use historical daily data from 01/01/2009 to 05/08/2020 to train the agent and test the performance.
The whole dataset is split in the following figure. Data from 01/01/2009 to 12/31/2014 is used for training, and the data from 10/01/2015 to 12/31/2015 is used for validation and tuning of parameters. Finally, we test our agent’s performance on trading data, which is the unseen out-of-sample data from 01/01/2016 to 05/08/2020. To better exploit the trading data, we continue training our agent while in the trading stage, since this will help the agent to better adapt to the market dynamics.


### MDP model for stock trading:
•State S = [p, h, b]: a vector that includes stock prices p ∈ R+^D, the stock shares h ∈ Z+^D, and the remaining balance b ∈ R+, where D denotes the number of stocks and Z+ denotes non-negative integers.
•Action a: a vector of actions over D stocks. The allowed actions on each stock include selling, buying, or holding, which result in decreasing, increasing, and no change of the stock shares h, respectively.
•Reward r(s,a,s'):the direct reward of taking action a at state s and arriving at the new state s'.
•Policy π(s): the trading strategy at state s, which is the probability distribution of actions at state s.
•Q-value Qπ (s,a): the expected reward of taking action ? at state ? following policy π.
The state transition of our stock trading process is shown in the following figure. At each state, one of three possible actions is taken on stock d (d= 1, …, D) in the portfolio.
Selling k[d] ∈ [1,h[d]] shares results in ht+1[d] = ht[d] − k[d], where k[d] ∈ Z+ and d =1,…,D.
Holding, ht+1[d] = ht[d].
Buying k[d] shares results in ht+1[d] = ht[d] + k[d].
At time t an action is taken and the stock prices update at t+1, accordingly the portfolio values may change from “portfolio value 0” to “portfolio value 1”, “portfolio value 2”, or “portfolio value 3”, respectively, as illustrated in the below figure. Note that the portfolio value is pT h + b.


### External Constraints:
Market liquidity: The orders can be rapidly executed at the close price. We assume that stock market will not be affected by our reinforcement trading agent.
Nonnegative balance: the allowed actions should not result in a negative balance.
Transaction cost: transaction costs are incurred for each trade. There are many types of transaction costs such as exchange fees, execution fees, and SEC fees. Different brokers have different commission fees. Despite these variations in fees, we assume that our transaction costs to be 1/1000 of the value of each trade (either buy or sell).
Risk-aversion for market crash: there are sudden events that may cause stock market crash, such as wars, collapse of stock market bubbles, sovereign debt default, and financial crisis. To control the risk in a worst-case scenario like 2008 global financial crisis, we employ the financial turbulence index that measures extreme asset price movements.

### Return maximization as trading goal
Reward function is defined as the change of the portfolio value when action a is taken at state s and arriving at new state s + 1.
The goal is to design a trading strategy that maximizes the change of the portfolio value r(st,at,st+1) in the dynamic environment, and deep reinforcement learning method is used to solve this problem.


### Implementation:

Stable Baselines  Python Packaged is used to run the experiments. Stable Baselines is a set of improved implementations of Reinforcement Learning (RL) algorithms based on OpenAI Baselines

### Performance evaluations:
Quantopian’s pyfolio is used to do the backtesting. 

 


### Conclusion:
The agent was able to successfully learn the policy that gives better return than the benchmark. Also the ensamble of models helps to improve the overall performance of the portfolio.
