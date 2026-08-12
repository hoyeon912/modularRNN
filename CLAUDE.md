# modular RNN

## Spec

- 3 layer RNN (input, 1 hidden, output layers)
	- input can be features (1, N) or images (M, N, C)
	- ouput is determined by environments
		- if an environment requries categories, make probabilities of each category.
		- if an environment requires action, make expected future reward of each action.
	- Loss funciton is also changed by environments
		- if an environment requires action, loss function is temporal difference learning $\delta = r + \gamma * V(t+1) - V(t)$
		- if an environment requires categories, use optimal loss function what you think.
- hidden layer is broken into 3 modules, which are input, intermediate, output
- nodes in the hidden layer
	- link densly in the same module.
		- input <-> input
		- intermediate <-> intermediate
		- output <-> output
	- link sparsely with the near module
		- input <-> intermediate
		- intermediate <-> output
	- don't link with the futher module.
		- input <-> output
- Use .venv folder

## Requirement

- GPU accerlation
- Pytorch
- Launch all training/long-running runs fully detached (e.g. `nohup ... & disown`) so they keep running even if the SSH connection is lost. Verify detachment (PPID 1, SIGHUP ignored) before considering a run properly launched.

## Test

- Use small size of hidden units in simple test
- Use the total 300 units for hidden layers in concrete test

### Categorical test

- MNIST dataset
- Target accuracy >= 90%

### Reinfocement learning test

- Several environments in the gymnasium package
- Gain maximum rewards of the selected envrionments before 500 episodes for 5 continuous episodes
