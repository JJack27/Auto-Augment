# UAlberta Multimedia Master Program - Reinforcement Learning for Auto Data Augmentation
An unofficial implementation of Google Brain's research in 2018 using tensorflow 1.15.0. Instead of using PPO, we use basic REINFORCE policy gradient algorithm with involving creative idea : depressed feedback.

## Requirement
- `numpy`
- `tensorflow 1.15.0`
- `keras`
- `PIL`
- `matplotlib`

## Code files
- `child_net.py`: containing python class representing child network. Can be replaced by any classifer as long as it is a **keras model**.
- `controller.py`: containing python class representing the RNN controller. Implemented in `tensorflow 1.x`.
- `data_iterator.py`: containing python class to load `cifar10` dataset. If policy is given, it will automatically apply image operations.
- `run.py`: code to run.
- `transformations.py`: contains functions of image transformations. 16 in total.

## How to run
```
python3 run.py
```
