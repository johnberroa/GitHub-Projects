# Button Presser

## What it does
This program asks the user to press the ```Enter``` key at specified rates.  Then it plots the results so that the user can see if they matched those rates, and also if they can maintain said rate.

## How it works
After the instructions appear, the user waits 4 seconds, and then the task begins.  It records button pressing rate every *.5* seconds.  After trying to hit the button at a rate of *8/second*, the results are plotted.  The user can then zoom in to see details if so desired with normal ```matplotlib``` functionality.

### Issues
The timing mechanism isn't exactly on time, but it's close enough to get an idea of button pressing rate.  I think to get it working perfectly would require threading.

## Example Images

![Overview](/Button%20Presser/gfx/button1.png?raw=true "Overview Image")
![Zoomed](/Button%20Presser/gfx/button2.png?raw=true "Zoomed In")
