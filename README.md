# NEAT and Transformer architecture super_mario_bros

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
### Transformer :
A Transformer is an attention-based encoder-decoder neural network architecture. The heart of these models is the
attention mechanism.


Just as self-attention in natural language processing allows a model to connect words based on 
context, in the context of Reinforcement Learning, it enables the model to focus on different aspects of the environment
by relating various parts of the input simultaneously, enabling it to prioritise important features of the game-state
(the position of the player, enemies, and obstacles) when making decisions. 


In other words, the attention mechanism is used for learning the importance of different components
from the game state dynamically, allowing the agent to adjust its focus based on the current situation. 
By using multiple attention heads, the agent can capture various relationships and patterns simultaneously
(i.e. focusing on nearby enemies when under threat or on exploring potential paths when navigating).


* the encoder: maps an input sequence into an abstract continuous representation that holds 
all the learned information of that input


* the decoder: takes the continuous representation (got from the encoder), and step by step generates 
a single output while also being fed the previous output
  

* multi-headed attention in the encoder applies a specific attention mechanism called self-attention.
