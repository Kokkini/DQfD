# Deep Q-Learning from Demonstrations
This repo replicates the results Hester et al. obtained:
[Deep Q-Learning from Demonstraitions](https://arxiv.org/abs/1704.03732 "Deep Q-Learning from Demonstraitions")  
This repo is based on the fantastic repo from [Morikatron/DQfD](https://github.com/morikatron/DQfD)
This code is based on code from OpenAI baselines. The original code and related paper from OpenAI can be found [here](https://github.com/openai/baselines "here").
<br/>
The algorithms, hyperparameters, etc. are based on the paper as much as possible.
<br/>
Visit https://tech.morikatron.ai/entry/2020/04/15/100000 for the Morikatron's great blog post about this algorithm.
<br/>
## Setting up the environment
Required libraries
- Tensorflow 2(tensorflow-gpu when using GPU)  
- gym
- tqdm
- dill


If you don't use GPU, replace
```python:
"with tf.device('/GPU:0'):
```
in dqfd.py with
```python:
with tf.device('/CPU:0'):"
```
<br/>
### Ubuntu setup example
Clone repo:
```python:
git clone https://github.com/Kokkini/DQfD.git
```

Create and active virtual environment
```python:
conda create -n DQfDenv
conda activate DQfDenv
```

Install required libraries
```python:
pip install tensorflow-2.0
(pip install tensorflow-gpu)
pip install gym
pip install tqdm
```


## How to use
First, run make_demo.py to create a demo.
Your demo will be saved in the ./data/demo directory.
```python:
python make_demo.py --env=BreakoutNoFrameskip-v4
```
### How to collect demo episodes
・w,s,a,d：move
・SPACE: jump
・Plus (+) on numpad: increase game speed
・Minus (-) on numpad: decrease game speed
・Each episode will be automatically saved when they end (done=True)
・backspace：reset current episode without saving  
・enter: save current episode and begin another episode (use this when you want to save the episode without waiting until the end of it)
・esc：end the collection of demo episodes (the current episode will not be saved)
<br/>

After collecting demo episodes, run run_atari.py to start learning:
```python:
python run_atari.py --pre_train_timesteps=1e5 --num_timesteps=4e6
```

## Demo data
If you don't want to create your own demo data, you can download the following demo data.
My demo data for 7 episodes of Breakout (my max score is 30):  
https://drive.google.com/file/d/15pXp-kwY_wFn2Eq6XRZkgQxdXLvZwcNn/view?usp=sharing
Place the pkl file of the link in the DQfD/data/demo directory. You can now start training without collecting your own demo episodes.

## If an error occurs on MacOS
When OMP: Error appears on MacOS, add the following to the head of dqfd.py
```python:
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
```
