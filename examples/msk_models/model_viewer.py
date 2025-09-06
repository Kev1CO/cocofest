"""
This example is a simple example of how to use the LiveModelAnimation class to animate a model in real-time.
The user can interact with the model by changing the joint angles using sliders.
"""

from pyorerun import LiveModelAnimation


model_path = "arm26.bioMod"
animation = LiveModelAnimation(model_path, with_q_charts=True)
animation.rerun()
