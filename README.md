# NEAT_geometry_dash

Neuro-Evolution of Augmenting Topologies is a black-box optimization technique, that uses an evolutionary algorithm, in this case a genetic algorithm, for evolving artificial neural networks(ANNs).
The goal here, it's to evolve an ANN for playing Geometry Dash.

Core features of NEAT : 
* incremental growth: starting minimally and adding nodes and connections only when needed;
  
* historical markings:  make it possible for the system to divide the population into species based on topological similarity;
  
* speciation of the population: individuals compete primarily within their own niches instead of with the population at large;
topological innovations are protected and have time to optimize their structure, before they have to compete with other niches in the population.
---

 ### Interaction with the game
The api gd.py makes it all possible ! 
