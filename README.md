# DQfD Minecraft

![]([https://github.com/Your_Repository_Name/Your_GIF_Name.gif](https://github.com/anishhdiwan/DQfD_Minecraft/blob/main/E19.gif))

This repo contains code and documentation to train a DQfD agent to solve the Treechop environment in mineRL v0.4

The methods and full replication details can be found at [Deep RL agents that learn behavior from human demonstrations](https://www.anishdiwan.com/post/deep-rl-in-minecraft)

The model can be run by executing main.py. The test_minerl directory contains test scripts that can be used to test individual functionalities from the main scripts. 

## How to reproduce this work?
Please refer to the appendix in the blog post for instructions on reproducing this work. The blog post goes into detail on several aspects of the project such as installation, demonstration sampling, algorithmic details, and hyperparameters. 

## Known Fixes
The data iterator from the minerl package iteratively loads transitions from all trajectories in the demonstration dataset into a data buffer in memory. Depending on the size of the buffer, this fills up quite a lot of memory. This causes the already overloaded RAM to fill up even faster and the Minecraft server dies with an error called "Killed" at some point during training.

A fix for this was implemented by monkey patching the buffered batch iterator class to load only one transition to the data buffer at a time. This patch is implemented in `buffered_batch_iter_patches.py`.
