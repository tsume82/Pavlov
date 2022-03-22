## Comparison Table
| Function    | p(PPO mf wIntraDeltaF + IntraDeltaX < jDE) with AUC metric | p(PPO mf wIntraDeltaF + IntraDeltaX < jDE) with best of the run metric |
| :---------- | ------------------------------ | ------------------------------- |
| BentCigar | 0.354 | **0.5908** |
| Discus | 0.4384 | 0.2972 |
| Ellipsoid | 0.3396 | **0.5296** |
| Katsuura | **0.5184** | 0.4516 |
| Rastrigin | 0.3556 | 0.4748 |
| Rosenbrock | 0.3824 | **0.5428** |
| Schaffers | 0.3376 | 0.4252 |
| Schwefel | 0.2472 | 0.3232 |
| Sphere | 0.2188 | 0.4524 |
| Weierstrass | 0.4112 | 0.3592 |
| AttractiveSector_5D | 0.422 | **0.5452** |
| AttractiveSector_10D | 0.3276 | **0.5352** |
| AttractiveSector_20D | 0.1676 | **0.542** |
| BuecheRastrigin_5D | 0.3284 | 0.4808 |
| BuecheRastrigin_10D | 0.3952 | **0.5352** |
| BuecheRastrigin_20D | 0.2876 | **0.5056** |
| CompositeGR_5D | 0.2964 | 0.4484 |
| CompositeGR_10D | 0.1964 | 0.3704 |
| CompositeGR_20D | 0.098 | 0.2992 |
| DifferentPowers_5D | 0.208 | **0.5448** |
| DifferentPowers_10D | 0.3244 | **0.5624** |
| DifferentPowers_20D | 0.35 | **0.66** |
| LinearSlope_5D | 0.3788 | 0.21 |
| LinearSlope_10D | 0.4796 | **0.64** |
| LinearSlope_20D | 0.318 | **0.5056** |
| SharpRidge_5D | 0.2808 | **0.5736** |
| SharpRidge_10D | 0.3284 | **0.618** |
| SharpRidge_20D | 0.3636 | **0.6472** |
| StepEllipsoidal_5D | 0.4204 | **0.698** |
| StepEllipsoidal_10D | 0.3848 | **0.7908** |
| StepEllipsoidal_20D | 0.3412 | **0.5884** |
| RosenbrockRotated_5D | 0.3044 | **0.558** |
| RosenbrockRotated_10D | 0.242 | **0.5384** |
| RosenbrockRotated_20D | 0.1336 | 0.3888 |
| SchaffersIllConditioned_5D | 0.3496 | **0.514** |
| SchaffersIllConditioned_10D | 0.4528 | **0.6088** |
| SchaffersIllConditioned_20D | 0.3852 | **0.5532** |
| LunacekBiR_5D | 0.3336 | 0.3184 |
| LunacekBiR_10D | 0.3692 | **0.51** |
| LunacekBiR_20D | 0.2728 | 0.4788 |
| GG101me_5D | 0.4444 | **0.594** |
| GG101me_10D | 0.3364 | **0.5532** |
| GG101me_20D | 0.2844 | 0.4624 |
| GG21hi_5D | 0.4372 | **0.5324** |
| GG21hi_10D | 0.422 | **0.6228** |
| GG21hi_20D | 0.3844 | 0.4732 |
| **Total p(PPO mf wIntraDeltaF + IntraDeltaX < jDE)** | 2.2% (1/46) | **63.0**% (29/46) |

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
                "uniform"
            ],
            [
                10,
                10,
                11,
                "best1bin",
                "uniform"
            ],
            [
                10,
                10,
                2,
                "best1bin",
                "uniform"
            ],
            [
                10,
                10,
                23,
                "best1bin",
                "uniform"
            ],
            [
                10,
                10,
                15,
                "best1bin",
                "uniform"
            ],
            [
                10,
                10,
                8,
                "best1bin",
                "uniform"
            ],
            [
                10,
                10,
                17,
                "best1bin",
                "uniform"
            ],
            [
                10,
                10,
                20,
                "best1bin",
                "uniform"
            ],
            [
                10,
                10,
                1,
                "best1bin",
                "uniform"
            ],
            [
                10,
                10,
                16,
                "best1bin",
                "uniform"
            ],
            [
                5,
                10,
                6,
                "best1bin",
                "uniform"
            ],
            [
                10,
                10,
                6,
                "best1bin",
                "uniform"
            ],
            [
                20,
                10,
                6,
                "best1bin",
                "uniform"
            ],
            [
                5,
                10,
                4,
                "best1bin",
                "uniform"
            ],
            [
                10,
                10,
                4,
                "best1bin",
                "uniform"
            ],
            [
                20,
                10,
                4,
                "best1bin",
                "uniform"
            ],
            [
                5,
                10,
                19,
                "best1bin",
                "uniform"
            ],
            [
                10,
                10,
                19,
                "best1bin",
                "uniform"
            ],
            [
                20,
                10,
                19,
                "best1bin",
                "uniform"
            ],
            [
                5,
                10,
                14,
                "best1bin",
                "uniform"
            ],
            [
                10,
                10,
                14,
                "best1bin",
                "uniform"
            ],
            [
                20,
                10,
                14,
                "best1bin",
                "uniform"
            ],
            [
                5,
                10,
                5,
                "best1bin",
                "uniform"
            ],
            [
                10,
                10,
                5,
                "best1bin",
                "uniform"
            ],
            [
                20,
                10,
                5,
                "best1bin",
                "uniform"
            ],
            [
                5,
                10,
                13,
                "best1bin",
                "uniform"
            ],
            [
                10,
                10,
                13,
                "best1bin",
                "uniform"
            ],
            [
                20,
                10,
                13,
                "best1bin",
                "uniform"
            ],
            [
                5,
                10,
                7,
                "best1bin",
                "uniform"
            ],
            [
                10,
                10,
                7,
                "best1bin",
                "uniform"
            ],
            [
                20,
                10,
                7,
                "best1bin",
                "uniform"
            ],
            [
                5,
                10,
                9,
                "best1bin",
                "uniform"
            ],
            [
                10,
                10,
                9,
                "best1bin",
                "uniform"
            ],
            [
                20,
                10,
                9,
                "best1bin",
                "uniform"
            ],
            [
                5,
                10,
                18,
                "best1bin",
                "uniform"
            ],
            [
                10,
                10,
                18,
                "best1bin",
                "uniform"
            ],
            [
                20,
                10,
                18,
                "best1bin",
                "uniform"
            ],
            [
                5,
                10,
                24,
                "best1bin",
                "uniform"
            ],
            [
                10,
                10,
                24,
                "best1bin",
                "uniform"
            ],
            [
                20,
                10,
                24,
                "best1bin",
                "uniform"
            ],
            [
                5,
                10,
                21,
                "best1bin",
                "uniform"
            ],
            [
                10,
                10,
                21,
                "best1bin",
                "uniform"
            ],
            [
                20,
                10,
                21,
                "best1bin",
                "uniform"
            ],
            [
                5,
                10,
                22,
                "best1bin",
                "uniform"
            ],
            [
                10,
                10,
                22,
                "best1bin",
                "uniform"
            ],
            [
                20,
                10,
                22,
                "best1bin",
                "uniform"
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