
note:
1. if you want to use model-based motion controller:set useRLCommand=False in RL_interface.py
    a.use 0 1 2 3 4 in keyboard to changed robot gait type when using model-based motion controller
    b.use ↑ ↓ ← → in keyboard to control robot velocity when using model-based motion controller
2. if you want to use your own controller for RL, set useRLCommand=True, and send your command via function sendCommandToRobot()

Author: yuzhiyou. 
wechat& phone number:18618386230.