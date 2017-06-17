# Regression Using Gradient Descent

## Why?
It's not the best way to do it, but it's a quick and easy way to play around with gradient descent, so why not? :D

## Learning Rate Augmentation
I have the learning rate speed up on shallower than average gradients, and slow down on steeper than average gradients.  This cuts about ~500 training steps from the training time, but makes the line jump around more. Very interesting.

### With Learning Rate Augmentation
![With Learning Rate Augmentation 1](/Gradient%20Descent/gfx/1with.png?raw=true "Regression with LRA")
![With Learning Rate Augmentation 2](/Gradient%20Descent/gfx/2with.png?raw=true "Stats with LRA")

### Without Learning Rate Augmentation
![Without Learning Rate Augmentation 1](/Gradient%20Descent/gfx/1without.png?raw=true "Regression without LRA")
![Without Learning Rate Augmentation 2](/Gradient%20Descent/gfx/2without.png?raw=true "Stats without LRA")
