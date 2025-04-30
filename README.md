# NEAT and Meta-Optimisation for super_mario_bros

---

Neuro-Evolution of Augmenting Topologies is an evolutionary approach
(a genetic algorithm) that creates and evolves artificial neural networks(ANNs).
It is a black-box optimisation technique, an alternative to the more traditional reinforcement learning routes.
The mutation operator adjusts both the weights, but also the connectivity between the nodes,
making it small and efficient.


The goal here, it's to play the original super mario bros (using gym-super-mario-bros).


 ### Core features of NEAT : 
* incremental growth: start minimally and add nodes and connections only when needed;
  

* historical markings: make it possible for the system to divide the population into species based on topological similarity;


* speciation of the population: individuals compete primarily within their own niches instead of with the population at large;
topological innovations are protected and have time to optimize their structure, before they have to compete with other niches in the population.


---
### Meta-Optimisation:
NEAT has over 50 hyperparameters that can be adjusted.
Satisfactory results are delivered if only these parameters are carefully chosen.
Selecting by trial and error is a laborious task and is also susceptible to misconceptions or personal biases.

One approach for finding good parameters for an optimizer is to employ yet another overlaying optimizer, called 
meta-optimizer. In this case, the optimizer is yet another Genetic Algorithm that uses mechanisms inspired by biological
evolution (such as selection, mutation, recombination) to guide it in the large fitness landscape.

![Mario Agent](runs/winner.gif)