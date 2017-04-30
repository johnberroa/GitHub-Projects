"""
This is a test space to trial the artyAI code
"""
import artilleryAI as aal

algorithm = 'ANN'
target = aal.learningMethodTarget(algorithm)
trainingSample = target.newTarget(10, "annTrain")

print(trainingSample)

arty = aal.Artillery()
v, a = arty.shoot()
env = aal.Physics()
distance = env.ballistic(v, a)

print(distance)
