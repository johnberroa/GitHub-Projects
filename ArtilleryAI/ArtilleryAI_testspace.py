"""
This is a test space to trial the artyAI code
"""
import artilleryAI as aal

algorithm = 'ANN'
# target = aal.learningMethodTarget(algorithm)
target = aal.Target()


arty = aal.Artillery()
t,tt = target.generateANNSets()
print(t)
print('new')
print(tt)
v, a = arty.shoot()
env = aal.Physics()
distance = env.ballistic(v, a)

ann = aal.ANN()

print(distance)
