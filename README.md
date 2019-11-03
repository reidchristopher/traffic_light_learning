# Transfer Learning for Reinforcement Learning Based Traffic Light Control

## SUMO

We are using SUMO in Python to simulate traffic at a four way intersection. SUMO can be installed in Linux with apt using

```
sudo apt-get install sumo sumo-tools sumo-doc
```

### SUMO_HOME

If a program asks for SUMO_HOME environment variable to be set in Ubuntu, it should be /usr/share/sumo/. You can run

```
export SUMO_HOME=/usr/share/sum/
```

for every new terminal running sumo or just add that line to the end of your .bashrc.

## TensorFlow

To avoid unecessarily implementing our own neural network structure, we will be using TensorFlow for our neural networks. I am using miniconda on my machine, but it can also be installed with pip by simpy replacing <code>conda</code> with <code>pip</code>. tensorflow-gpu offers a gpu implementation that we may or may not use at this point.

```
sudo conda install tensorflow [tensorflow-gpu]
```

