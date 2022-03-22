## Comparison Table
| Function    | p(PPO mf wIntraDeltaF + IntraDeltaX < jDE) with AUC metric | p(PPO mf wIntraDeltaF + IntraDeltaX < jDE) with best of the run metric |
| :---------- | ------------------------------ | ------------------------------- |
| BentCigar | 0.304 | 0.4132 |
| Discus | **0.5052** | 0.4048 |
| Ellipsoid | 0.3696 | 0.29 |
| Katsuura | **0.542** | **0.5756** |
| Rastrigin | 0.3616 | **0.5012** |
| Rosenbrock | 0.3128 | 0.3036 |
| Schaffers | 0.3056 | 0.3336 |
| Schwefel | 0.3512 | 0.2808 |
| Sphere | 0.1192 | 0.2304 |
| Weierstrass | **0.722** | **0.7724** |
| AttractiveSector_5D | 0.4472 | 0.39 |
| AttractiveSector_10D | 0.4492 | 0.3796 |
| AttractiveSector_20D | 0.13 | 0.3176 |
| BuecheRastrigin_5D | 0.258 | 0.3804 |
| BuecheRastrigin_10D | 0.2136 | 0.3272 |
| BuecheRastrigin_20D | 0.1768 | 0.2584 |
| CompositeGR_5D | 0.3156 | 0.3696 |
| CompositeGR_10D | 0.192 | 0.4356 |
| CompositeGR_20D | 0.208 | 0.4124 |
| DifferentPowers_5D | 0.2064 | 0.4844 |
| DifferentPowers_10D | 0.2952 | 0.3348 |
| DifferentPowers_20D | 0.26 | 0.3508 |
| LinearSlope_5D | 0.3048 | 0.1804 |
| LinearSlope_10D | 0.3736 | 0.4932 |
| LinearSlope_20D | 0.168 | 0.252 |
| SharpRidge_5D | 0.2232 | 0.4608 |
| SharpRidge_10D | 0.3192 | 0.4612 |
| SharpRidge_20D | 0.1728 | 0.3076 |
| StepEllipsoidal_5D | 0.452 | **0.5208** |
| StepEllipsoidal_10D | 0.4648 | **0.6232** |
| StepEllipsoidal_20D | 0.2476 | 0.3272 |
| RosenbrockRotated_5D | 0.3072 | 0.4828 |
| RosenbrockRotated_10D | 0.2252 | 0.4144 |
| RosenbrockRotated_20D | 0.2216 | 0.2028 |
| SchaffersIllConditioned_5D | 0.3592 | 0.4276 |
| SchaffersIllConditioned_10D | 0.3616 | 0.4004 |
| SchaffersIllConditioned_20D | 0.3576 | 0.4256 |
| LunacekBiR_5D | 0.3276 | 0.4152 |
| LunacekBiR_10D | 0.336 | **0.506** |
| LunacekBiR_20D | 0.2692 | 0.3704 |
| GG101me_5D | 0.4976 | **0.6192** |
| GG101me_10D | 0.3664 | 0.4584 |
| GG101me_20D | 0.2116 | 0.254 |
| GG21hi_5D | **0.512** | 0.4816 |
| GG21hi_10D | 0.2892 | 0.3812 |
| GG21hi_20D | 0.3708 | 0.3932 |
| **Total p(PPO mf wIntraDeltaF + IntraDeltaX < jDE)** | 8.7% (4/46) | 15.2% (7/46) |

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
    "env.env_class": "SchedulerPolicyMultiRayEnvironment",
    "env.env_config": {
        "solver_driver": "DEdriver",
        "solver_driver_args": [
            [
                10,
                10,
                12,
                "best1bin",
                "normal"
            ],
            [
                10,
                10,
                11,
                "best1bin",
                "normal"
            ],
            [
                10,
                10,
                2,
                "best1bin",
                "normal"
            ],
            [
                10,
                10,
                23,
                "best1bin",
                "normal"
            ],
            [
                10,
                10,
                15,
                "best1bin",
                "normal"
            ],
            [
                10,
                10,
                8,
                "best1bin",
                "normal"
            ],
            [
                10,
                10,
                17,
                "best1bin",
                "normal"
            ],
            [
                10,
                10,
                20,
                "best1bin",
                "normal"
            ],
            [
                10,
                10,
                1,
                "best1bin",
                "normal"
            ],
            [
                10,
                10,
                16,
                "best1bin",
                "normal"
            ],
            [
                5,
                10,
                6,
                "best1bin",
                "normal"
            ],
            [
                10,
                10,
                6,
                "best1bin",
                "normal"
            ],
            [
                20,
                10,
                6,
                "best1bin",
                "normal"
            ],
            [
                5,
                10,
                4,
                "best1bin",
                "normal"
            ],
            [
                10,
                10,
                4,
                "best1bin",
                "normal"
            ],
            [
                20,
                10,
                4,
                "best1bin",
                "normal"
            ],
            [
                5,
                10,
                19,
                "best1bin",
                "normal"
            ],
            [
                10,
                10,
                19,
                "best1bin",
                "normal"
            ],
            [
                20,
                10,
                19,
                "best1bin",
                "normal"
            ],
            [
                5,
                10,
                14,
                "best1bin",
                "normal"
            ],
            [
                10,
                10,
                14,
                "best1bin",
                "normal"
            ],
            [
                20,
                10,
                14,
                "best1bin",
                "normal"
            ],
            [
                5,
                10,
                5,
                "best1bin",
                "normal"
            ],
            [
                10,
                10,
                5,
                "best1bin",
                "normal"
            ],
            [
                20,
                10,
                5,
                "best1bin",
                "normal"
            ],
            [
                5,
                10,
                13,
                "best1bin",
                "normal"
            ],
            [
                10,
                10,
                13,
                "best1bin",
                "normal"
            ],
            [
                20,
                10,
                13,
                "best1bin",
                "normal"
            ],
            [
                5,
                10,
                7,
                "best1bin",
                "normal"
            ],
            [
                10,
                10,
                7,
                "best1bin",
                "normal"
            ],
            [
                20,
                10,
                7,
                "best1bin",
                "normal"
            ],
            [
                5,
                10,
                9,
                "best1bin",
                "normal"
            ],
            [
                10,
                10,
                9,
                "best1bin",
                "normal"
            ],
            [
                20,
                10,
                9,
                "best1bin",
                "normal"
            ],
            [
                5,
                10,
                18,
                "best1bin",
                "normal"
            ],
            [
                10,
                10,
                18,
                "best1bin",
                "normal"
            ],
            [
                20,
                10,
                18,
                "best1bin",
                "normal"
            ],
            [
                5,
                10,
                24,
                "best1bin",
                "normal"
            ],
            [
                10,
                10,
                24,
                "best1bin",
                "normal"
            ],
            [
                20,
                10,
                24,
                "best1bin",
                "normal"
            ],
            [
                5,
                10,
                21,
                "best1bin",
                "normal"
            ],
            [
                10,
                10,
                21,
                "best1bin",
                "normal"
            ],
            [
                20,
                10,
                21,
                "best1bin",
                "normal"
            ],
            [
                5,
                10,
                22,
                "best1bin",
                "normal"
            ],
            [
                10,
                10,
                22,
                "best1bin",
                "normal"
            ],
            [
                20,
                10,
                22,
                "best1bin",
                "normal"
            ]
        ],
        "maximize": false,
        "steps": 50,
        "state_metrics_names": [
            "DifferenceOfBest",
            "DeltaFitPop",
            "DeltaX",
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
                40,
                false
            ],
            [
                40,
                false,
                true
            ],
            [
                {
                    "F_mean": {
                        "max": [
                            2
                        ],
                        "min": [
                            0
                        ]
                    },
                    "F_stdev": {
                        "max": [
                            1
                        ],
                        "min": [
                            0
                        ]
                    },
                    "CR_mean": {
                        "max": [
                            1
                        ],
                        "min": [
                            0
                        ]
                    },
                    "CR_stdev": {
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
            "F_mean": {
                "max": 2,
                "min": 0
            },
            "F_stdev": {
                "max": 1,
                "min": 0
            },
            "CR_mean": {
                "max": 1,
                "min": 0
            },
            "CR_stdev": {
                "max": 1,
                "min": 0
            }
        }
    }
}
```