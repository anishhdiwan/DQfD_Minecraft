# Defining actions as dictionaries (7 actions in total)

# Order of the actions dictionary
# [fwd, bwd, left, right, front-left, front-right, look left, look right, look up, look down, jump, front-jump, attack, front-attack]


# action_names is a dictionary mapping action names (which are strings) to their indices in the actions list
action_names = {
'forward':0,
'back':1,
'left':2,
'right':3,
'forward_left':4,
'left_forward':4,
'forward_right':5,
'right_forward':5,
'look-left':6,
'look-right':7,
'look-up':8,
'look-down':9,
'jump':10,
'forward_jump':11,
'jump_forward':11,
'attack':12,
'forward_attack':13,
'attack_forward':13,
'noop':14
}

# actions is a list of action dicts in the format that mineRL requires
actions = [

{
    "attack": 0,
    "back": 0,
    "camera": [0,0],
    "forward": 1,
    "jump": 0,
    "left": 0,
    "right": 0,
    "sneak": 0,
    "sprint": 0
},

{
    "attack": 0,
    "back": 1,
    "camera": [0,0],
    "forward": 0,
    "jump": 0,
    "left": 0,
    "right": 0,
    "sneak": 0,
    "sprint": 0
},

{
    "attack": 0,
    "back": 0,
    "camera": [0,0],
    "forward": 0,
    "jump": 0,
    "left": 1,
    "right": 0,
    "sneak": 0,
    "sprint": 0
},

{
    "attack": 0,
    "back": 0,
    "camera": [0,0],
    "forward": 0,
    "jump": 0,
    "left": 0,
    "right": 1,
    "sneak": 0,
    "sprint": 0
},

{
    "attack": 0,
    "back": 0,
    "camera": [0,0],
    "forward": 1,
    "jump": 0,
    "left": 1,
    "right": 0,
    "sneak": 0,
    "sprint": 0
},

{
    "attack": 0,
    "back": 0,
    "camera": [0,0],
    "forward": 1,
    "jump": 0,
    "left": 0,
    "right": 1,
    "sneak": 0,
    "sprint": 0
},

{
    "attack": 0,
    "back": 0,
    "camera": [2,0],
    "forward": 0,
    "jump": 0,
    "left": 0,
    "right": 0,
    "sneak": 0,
    "sprint": 0
},

{
    "attack": 0,
    "back": 0,
    "camera": [-2,0],
    "forward": 0,
    "jump": 0,
    "left": 0,
    "right": 0,
    "sneak": 0,
    "sprint": 0
},

{
    "attack": 0,
    "back": 0,
    "camera": [0,2],
    "forward": 0,
    "jump": 0,
    "left": 0,
    "right": 0,
    "sneak": 0,
    "sprint": 0
},

{
    "attack": 0,
    "back": 0,
    "camera": [0,-2],
    "forward": 0,
    "jump": 0,
    "left": 0,
    "right": 0,
    "sneak": 0,
    "sprint": 0
},

{
    "attack": 0,
    "back": 0,
    "camera": [0,0],
    "forward": 0,
    "jump": 1,
    "left": 0,
    "right": 0,
    "sneak": 0,
    "sprint": 0
},

{
    "attack": 0,
    "back": 0,
    "camera": [0,0],
    "forward": 1,
    "jump": 1,
    "left": 0,
    "right": 0,
    "sneak": 0,
    "sprint": 0
},

{
    "attack": 1,
    "back": 0,
    "camera": [0,0],
    "forward": 0,
    "jump": 0,
    "left": 0,
    "right": 0,
    "sneak": 0,
    "sprint": 0
},

{
    "attack": 1,
    "back": 0,
    "camera": [0,0],
    "forward": 1,
    "jump": 0,
    "left": 0,
    "right": 0,
    "sneak": 0,
    "sprint": 0
},

{
    "attack": 0,
    "back": 0,
    "camera": [0,0],
    "forward": 0,
    "jump": 0,
    "left": 0,
    "right": 0,
    "sneak": 0,
    "sprint": 0
}

]
