# DQfD Minecraft
This repo contains code and documentation to train a DQfD agent to solve the Treechop environment in mineRL v0.4

The methods and full replication details can be found at [Deep RL agents that learn behavior from human demonstrations](https://www.anishdiwan.com/post/deep-rl-in-minecraft)

The model can be run by executing main.py. The test_minerl directory contains test scripts that can be used to test individual functionalities from the main scripts. 

## How to reproduce this work?
Please refer to the appendix in the blog post for instructions on reproducing this work. The blog post goes into detail on several aspects of the project such as installation, demonstration sampling, algorithmic details, and hyperparameters. 

## Known Issues
The data iterator from the minerl package seems to be iteratively loading demonstration data to the memory whenever 'next()' is called. This causes the already overloaded RAM to fill up even faster and the Minecraft server dies with an error called "Killed" at some point during training. A fix is currently not known. This issue is being solved at present.
