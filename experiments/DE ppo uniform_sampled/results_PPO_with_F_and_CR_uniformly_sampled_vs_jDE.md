## Comparison Table
| Function    | p(PPO with F and CR uniformly sampled < jDE) with AUC metric | p(PPO with F and CR uniformly sampled < jDE) with best of the run metric |
| :---------- | ------------------------------ | ------------------------------- |
| BentCigar | 0.4528 | **0.5832** |
| Discus | **0.5044** | 0.3772 |
| Ellipsoid | 0.3968 | 0.478 |
| Katsuura | **0.6152** | **0.5352** |
| Rastrigin | 0.4512 | **0.528** |
| Rosenbrock | 0.3944 | **0.5196** |
| Schaffers | 0.462 | **0.5636** |
| Schwefel | 0.3316 | 0.2772 |
| Sphere | 0.0292 | 0.25 |
| Weierstrass | **0.7252** | **0.7928** |

## Plots

##### BentCigar

![](BentCigar/PPO_with_F_and_CR_uniformly_sampled_vs_jDE:_BentCigar_comparison.png)

##### Discus

![](Discus/PPO_with_F_and_CR_uniformly_sampled_vs_jDE:_Discus_comparison.png)

##### Ellipsoid

![](Ellipsoid/PPO_with_F_and_CR_uniformly_sampled_vs_jDE:_Ellipsoid_comparison.png)

##### Katsuura

![](Katsuura/PPO_with_F_and_CR_uniformly_sampled_vs_jDE:_Katsuura_comparison.png)

##### Rastrigin

![](Rastrigin/PPO_with_F_and_CR_uniformly_sampled_vs_jDE:_Rastrigin_comparison.png)

##### Rosenbrock

![](Rosenbrock/PPO_with_F_and_CR_uniformly_sampled_vs_jDE:_Rosenbrock_comparison.png)

##### Schaffers

![](Schaffers/PPO_with_F_and_CR_uniformly_sampled_vs_jDE:_Schaffers_comparison.png)

##### Schwefel

![](Schwefel/PPO_with_F_and_CR_uniformly_sampled_vs_jDE:_Schwefel_comparison.png)

##### Sphere

![](Sphere/PPO_with_F_and_CR_uniformly_sampled_vs_jDE:_Sphere_comparison.png)

##### Weierstrass

![](Weierstrass/PPO_with_F_and_CR_uniformly_sampled_vs_jDE:_Weierstrass_comparison.png)


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