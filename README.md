To run:

Install stable-baselines3 and gym-multigrid, included.

Run the corresponding script to reproduce.

Statistical analysis is in figure_5_plot comments.
Comment/uncomment Frozen & ISR or ISR & Oracle datapoints and rename Frozen --> Simple, ISR --> Reward to run the respective t-tests.

# Architecture

The policy network is a convolutional neural network (CNN) followed by a multi-layer perception (MLP).
The CNN has architecture as follows, where the fourth channel is the goal:
```
nn.Sequential(
        nn.Conv2d(4, 16, kernel_size=(2, 2)),
        nn.ReLU(),
        nn.Conv2d(16, 32, kernel_size=(2, 2)),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=(2, 2)),
        nn.ReLU(),
        nn.Flatten(),
)
```
where the first parameter is the number of input channels and the second parameter is the number of output channels,
while the MLP is:
```
nn.Sequential(
    nn.Linear(n_flatten, 128), 
    nn.ReLU()
)
```
where `n\_flatten` is the number of channels resulting from the `nn.Flatten()` from the CNN.

The reward model is similar, though takes in an additional channel for the action. It consists of a CNN followed by an MLP with the CNN as follows:
```
nn.Sequential(
    nn.Conv2d(5, 16, kernel_size=(2, 2)),
    nn.ReLU(),
    nn.Conv2d(16, 32, kernel_size=(2, 2)),
    nn.ReLU(),
    nn.Flatten(),
)
```
and the MLP as follows:
```
nn.Sequential(
    nn.Linear(n_flatten, 32),
    nn.ReLU(), 
    nn.Linear(32, 32),
    nn.ReLU(), 
    nn.Linear(32, 1),
)
```
where `n\_flatten` is defined similarly. 
The MLP layer weights are initialized using the following function:
```
def layer_init(layer, std=np.sqrt(2)):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, 0.0)
        return layer
```
We initialize the MLP weights and biases by setting the biases to 0 and setting the layer weights to an orthogonal matrix with scaling factor equal to $\sqrt{2}$.

For training parameters, we use the standard settings from Stable Baselines 3.
These default settings are as follows:
```
learning_rate: 3e-4,
batch_size: 64,
n_epochs: 10,
gamma: 0.99,
gae_lambda: 0.95,
clip_range: 0.2,
clip_range_vf: None,
normalize_advantage: True,
ent_coef: 0.0,
vf_coef: 0.5,
max_grad_norm: 0.5,
use_sde: False,
sde_sample_freq: -1,
target_kl: None.
```
For figure 5, we set learning rate in the first pass equal to $10^{-4}$. 

We train for $12\cdot 10^4$ steps in the first passes and $24\cdot 10^4$ steps in the second passes.


# Citing

```
@inproceedings{rong2024value,
  title={Value Internalization: Learning and Generalizing from Social Reward},
  author={Rong, Frieda and Kleiman-Weiner, Max},
  booktitle={Proceedings of the Annual Meeting of the Cognitive Science Society},
  volume={46},
  year={2024}
}
```