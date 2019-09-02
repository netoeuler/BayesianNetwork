#Import required packages
import math
from pomegranate import *

#Bayesian network based on the graph of the following example: https://towardsdatascience.com/introduction-to-bayesian-networks-81031eeed94e
#Code based on the following example: https://www.edureka.co/blog/bayesian-networks/
 
cloudy =DiscreteDistribution( { 'T': 0.5, 'F': 0.5 } )

sprinkler =ConditionalProbabilityTable(
[[ 'T', 'T', 0.1 ],
[ 'T', 'F', 0.9 ],
[ 'F', 'T', 0.5 ],
[ 'F', 'F', 0.5 ]], [cloudy])

rain =ConditionalProbabilityTable(
[[ 'T', 'T', 0.8 ],
[ 'T', 'F', 0.2 ],
[ 'F', 'T', 0.2 ],
[ 'F', 'F', 0.8 ]], [cloudy])
   
wetgrass =ConditionalProbabilityTable(
[[ 'T', 'T', 'T', 0.99 ],
[ 'T', 'T', 'F', 0.01 ],
[ 'T', 'F', 'T', 0.9 ],
[ 'T', 'F', 'F', 0.1 ],
[ 'F', 'T', 'T', 0.9 ],
[ 'F', 'T', 'F', 0.1 ],
[ 'F', 'F', 'T', 0.0 ],
[ 'F', 'F', 'F', 1.0 ]], [sprinkler, rain] )
            
d1 = State( cloudy, name="cloudy" )
d2 = State( sprinkler, name="sprinkler" )
d3 = State( rain, name="rain" )
d4 = State( wetgrass, name="wetgrass" )
            
#Building the Bayesian Network
network = BayesianNetwork( "Will It rain?" )
network.add_states(d1, d2, d3, d4)
network.add_edge(d1, d2)
network.add_edge(d1, d3)
network.add_edge(d2, d4)
network.add_edge(d3, d4)
network.bake()

beliefs = network.predict_proba({ 'cloudy' : 'T' })
beliefs = map(str, beliefs)
print("n".join( "{}t{}".format( state.name, belief ) for state, belief in zip( network.states, beliefs ) ))
 
