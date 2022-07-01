<p align="center">
  <h3 align="center">Pavlov</h3>
  <p align="center">
    A reinforcement learning python library for training adaptive metaheuristics
  </p>
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#paper-pubblication">Paper Pubblication</a></li>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
Pavlov is a reinforcement python learning library providing the tools to optimize metaheuristics and their parameters/components, automatically managing the training phase and the information needed to test the trained models. 

**This library is still in development and many functionalities are not available yet**

### Paper Pubblication
Repository with code and experiments reported in: Reinforcement learning based adaptive metaheuristics, Michele Tessari, Giovanni Iacca.

DOI: [https://doi.org/10.48550/arxiv.2206.12233](https://doi.org/10.48550/arxiv.2206.12233)

```
@misc{https://doi.org/10.48550/arxiv.2206.12233,
  doi = {10.48550/ARXIV.2206.12233},
  url = {https://arxiv.org/abs/2206.12233},
  author = {Tessari, Michele and Iacca, Giovanni},
  keywords = {Neural and Evolutionary Computing (cs.NE), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Reinforcement learning based adaptive metaheuristics},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution Non Commercial No Derivatives 4.0 International}
}
```


### Built With

The core libraries used by Pavlov are:
* [Ray RLlib](https://docs.ray.io/en/master/rllib.html)
* [Tensorflow 2.x](https://www.tensorflow.org/)
* [OpenAI GYM](https://gym.openai.com/)

Additional RL models can be wrapped in Pavlov via the ``Agent`` interface, from libraries such as[TODO]:
* [TensorForce](https://tensorforce.readthedocs.io/)
* [TF_Agents](https://www.tensorflow.org/agents)



<!-- GETTING STARTED -->
## Getting Started
Pavlov interface is modelled after TensorFlow one, providing powerful commands to train and test the model configured. The main interface methods are still WIP.

### Prerequisites
Currently tested with Python 3.8.11, not intended to work with Pytohn 2.x.

### Installation
Create a dedicated virtualenv using the ``requirements.txt`` file (TODO), eventually adding the libraries needed for custom environments or custom metrics

- Clone the repository
```bash
git clone https://github.com/tsume82/Pavlov
```
- [TODO] Install the requirements

- Test a training
```bash
python test/test.py
```

<!-- USAGE EXAMPLES -->
## Usage
Write a configuration file, example ```config.json```:
```json
{
    "agent.algorithm": "RayProximalPolicyOptimization",
    "agent.algorithm.render_env": false,
    "agent.algorithm.num_workers": 0,
    "agent.algorithm.batch_mode": "complete_episodes",
    "agent.algorithm.lr": 1e-05,
    "agent.algorithm.train_batch_size": 200,
    "agent.algorithm.optimizer": "Adam",
    "agent.algorithm.model": {"fcnet_activation": "relu", "fcnet_hiddens": [50, 50]},
    "env.env_class": "SchedulerPolicyRayEnvironment",
    "env.env_config": {
        "solver_driver": "CMAdriver",
        "solver_driver_args": [10, 10, 11, 0.5, [-5.12, 5.12]],
        "maximize": false,
        "steps": 50,
        "state_metrics_names": ["MetricHistory", "MetricHistory", "MetricHistory", "MetricHistory", "SolverStateHistory"],
        "state_metrics_config": [
            ["IntraDeltaF", [], 40], 
            ["InterDeltaF", [], 40], 
            ["IntraDeltaX", [], 40], 
            ["InterDeltaX", [], 40], 
            [{"step_size": {"max": 3, "min": 0}}, 40]
            ],
        "reward_metric": "DeltaBest",
        "reward_metric_config": [true, true],
        "memes_no": 1,
        "action_space_config": {"step_size": {"max": 3, "min": 1e-5}}
    }
}
```

Run the training:
```
python main.py train -c config.json
```
