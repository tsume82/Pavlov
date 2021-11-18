## Comparison Table
| Function    | p(PPO with F and CR uniformly sampled < iDE) with AUC metric | p(PPO with F and CR uniformly sampled < iDE) with best of the run metric |
| :---------- | ------------------------------ | ------------------------------- |
| BentCigar | 0.4876 | **0.6024** |
| Discus | 0.4724 | **0.7104** |
| Ellipsoid | 0.4336 | **0.68** |
| Katsuura | 0.4888 | 0.4324 |
| Rastrigin | **0.5072** | **0.6192** |
| Rosenbrock | 0.326 | **0.5956** |
| Schaffers | **0.5284** | **0.6572** |
| Schwefel | 0.3416 | **0.6164** |
| Sphere | 0.1064 | 0.4524 |
| Weierstrass | **0.7704** | **0.7848** |

## Plots

##### BentCigar

![](BentCigar/PPO_with_F_and_CR_uniformly_sampled_vs_iDE:_BentCigar_comparison.png)

##### Discus

![](Discus/PPO_with_F_and_CR_uniformly_sampled_vs_iDE:_Discus_comparison.png)

##### Ellipsoid

![](Ellipsoid/PPO_with_F_and_CR_uniformly_sampled_vs_iDE:_Ellipsoid_comparison.png)

##### Katsuura

![](Katsuura/PPO_with_F_and_CR_uniformly_sampled_vs_iDE:_Katsuura_comparison.png)

##### Rastrigin

![](Rastrigin/PPO_with_F_and_CR_uniformly_sampled_vs_iDE:_Rastrigin_comparison.png)

##### Rosenbrock

![](Rosenbrock/PPO_with_F_and_CR_uniformly_sampled_vs_iDE:_Rosenbrock_comparison.png)

##### Schaffers

![](Schaffers/PPO_with_F_and_CR_uniformly_sampled_vs_iDE:_Schaffers_comparison.png)

##### Schwefel

![](Schwefel/PPO_with_F_and_CR_uniformly_sampled_vs_iDE:_Schwefel_comparison.png)

##### Sphere

![](Sphere/PPO_with_F_and_CR_uniformly_sampled_vs_iDE:_Sphere_comparison.png)

##### Weierstrass

![](Weierstrass/PPO_with_F_and_CR_uniformly_sampled_vs_iDE:_Weierstrass_comparison.png)


## Configuration

```json
{
    "agent.algorithm": "RayProximalPolicyOptimization",
    "agent.algorithm.render_env": false,
    "agent.algorithm.num_workers": 0,
    "agent.algorithm.batch_mode": "complete_episodes",
    "agent.algorithm.lr": 5e-05,
    "agent.algorithm.train_batch_size": 200,
    "agent.algorithm.optimizer": "Adam",
    "agent.algorithm.vf_clip_param": 10,
    "agent.algorithm.model": {
        "fcnet_activation": "relu",
        "fcnet_hiddens": [
            50,
            50
        ]
    },
    "env.env_class": "SchedulerPolicyRayEnvironment",
    "env.env_config": {
        "solver_driver": "DEdriver",
        "solver_driver_args": [
            10,
            10,
            12,
            "best1bin",
            "uniform"
        ],
        "maximize": false,
        "steps": 50,
        "state_metrics_names": [
            "DifferenceOfBest",
            "SolverStateHistory"
        ],
        "state_metrics_config": [
            [
                40,
                false,
                1,
                true,
                false
            ],
            [
                {
                    "F_min": {
                        "max": [
                            2
                        ],
                        "min": [
                            0
                        ]
                    },
                    "F_max": {
                        "max": [
                            2
                        ],
                        "min": [
                            0
                        ]
                    },
                    "CR_min": {
                        "max": [
                            1
                        ],
                        "min": [
                            0
                        ]
                    },
                    "CR_max": {
                        "max": [
                            1
                        ],
                        "min": [
                            0
                        ]
                    }
                },
                40
            ]
        ],
        "reward_metric": "DeltaBest",
        "reward_metric_config": [
            false,
            true,
            true
        ],
        "memes_no": 1,
        "action_space_config": {
            "F_min": {
                "max": 2,
                "min": 0
            },
            "F_max": {
                "max": 2,
                "min": 0
            },
            "CR_min": {
                "max": 1,
                "min": 0
            },
            "CR_max": {
                "max": 1,
                "min": 0
            }
        }
    }
}
```