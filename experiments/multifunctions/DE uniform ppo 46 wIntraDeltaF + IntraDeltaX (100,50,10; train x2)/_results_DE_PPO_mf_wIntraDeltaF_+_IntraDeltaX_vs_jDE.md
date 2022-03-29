## Comparison Table
| Function    | p(PPO mf wIntraDeltaF + IntraDeltaX < jDE) with AUC metric | p(PPO mf wIntraDeltaF + IntraDeltaX < jDE) with best of the run metric |
| :---------- | ------------------------------ | ------------------------------- |
| BentCigar | 0.2308 | 0.3644 |
| Discus | **0.51** | 0.4056 |
| Ellipsoid | 0.2048 | 0.226 |
| Katsuura | **0.5864** | **0.5236** |
| Rastrigin | 0.268 | **0.5584** |
| Rosenbrock | 0.286 | 0.2744 |
| Schaffers | 0.2456 | 0.4416 |
| Schwefel | 0.2204 | 0.206 |
| Sphere | 0.0888 | 0.3668 |
| Weierstrass | 0.436 | **0.5028** |
| AttractiveSector_5D | **0.5688** | 0.3552 |
| AttractiveSector_10D | 0.35 | 0.3392 |
| AttractiveSector_20D | 0.078 | 0.252 |
| BuecheRastrigin_5D | 0.286 | 0.4852 |
| BuecheRastrigin_10D | 0.2672 | 0.444 |
| BuecheRastrigin_20D | 0.1892 | **0.532** |
| CompositeGR_5D | 0.326 | 0.3724 |
| CompositeGR_10D | 0.1616 | 0.3788 |
| CompositeGR_20D | 0.0744 | 0.2696 |
| DifferentPowers_5D | 0.2208 | **0.5144** |
| DifferentPowers_10D | 0.2904 | **0.5088** |
| DifferentPowers_20D | 0.2512 | **0.526** |
| LinearSlope_5D | 0.3464 | 0.1976 |
| LinearSlope_10D | 0.4192 | **0.5772** |
| LinearSlope_20D | 0.28 | 0.4756 |
| SharpRidge_5D | 0.1284 | 0.4112 |
| SharpRidge_10D | 0.2824 | **0.5176** |
| SharpRidge_20D | 0.2944 | **0.5948** |
| StepEllipsoidal_5D | 0.3892 | **0.6084** |
| StepEllipsoidal_10D | 0.2956 | **0.5956** |
| StepEllipsoidal_20D | 0.1928 | 0.4276 |
| RosenbrockRotated_5D | 0.2532 | 0.4412 |
| RosenbrockRotated_10D | 0.1188 | 0.3796 |
| RosenbrockRotated_20D | 0.066 | 0.2164 |
| SchaffersIllConditioned_5D | 0.2372 | 0.4144 |
| SchaffersIllConditioned_10D | 0.3056 | 0.478 |
| SchaffersIllConditioned_20D | 0.2464 | 0.3892 |
| LunacekBiR_5D | 0.2636 | 0.3148 |
| LunacekBiR_10D | 0.216 | 0.47 |
| LunacekBiR_20D | 0.2168 | 0.454 |
| GG101me_5D | 0.4016 | **0.5212** |
| GG101me_10D | 0.29 | 0.4628 |
| GG101me_20D | 0.2556 | 0.3856 |
| GG21hi_5D | 0.4384 | 0.4924 |
| GG21hi_10D | 0.3852 | **0.5224** |
| GG21hi_20D | 0.43 | **0.5124** |
| **Total p(PPO mf wIntraDeltaF + IntraDeltaX < jDE)** | 6.5% (3/46) | 32.6% (15/46) |

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
            100,
            50,
            10
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