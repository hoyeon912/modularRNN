# modular RNN

## Spec

- 3 layer bidirecitonal RNN (input, 1 hidden, output layers)
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

## Test

- simple test with MNIST dataset
- hard test with cartpole in the gymnasium

