## Comparison Table
| Function    | p(PPO mf wIntraDeltaF + IntraDeltaX + InterDeltaX < iDE) with AUC metric | p(PPO mf wIntraDeltaF + IntraDeltaX + InterDeltaX < iDE) with best of the run metric |
| :---------- | ------------------------------ | ------------------------------- |
| BentCigar | 0.4356 | **0.588** |
| Discus | 0.3984 | **0.5828** |
| Ellipsoid | 0.4192 | **0.7512** |
| Katsuura | **0.5184** | 0.4984 |
| Rastrigin | 0.3468 | **0.564** |
| Rosenbrock | 0.352 | **0.5956** |
| Schaffers | 0.422 | **0.6044** |
| Schwefel | 0.3188 | **0.6652** |
| Sphere | 0.358 | **0.5788** |
| Weierstrass | **0.5304** | **0.5292** |
| AttractiveSector_5D | 0.404 | 0.4624 |
| AttractiveSector_10D | **0.582** | **0.7948** |
| AttractiveSector_20D | **0.916** | **0.9156** |
| BuecheRastrigin_5D | 0.3432 | 0.242 |
| BuecheRastrigin_10D | **0.8248** | **0.9324** |
| BuecheRastrigin_20D | 0.39 | 0.4848 |
| CompositeGR_5D | 0.3232 | 0.4228 |
| CompositeGR_10D | 0.2132 | 0.4084 |
| CompositeGR_20D | 0.4176 | **0.6568** |
| DifferentPowers_5D | **0.602** | **0.8636** |
| DifferentPowers_10D | **0.5184** | 0.4592 |
| DifferentPowers_20D | 0.188 | 0.448 |
| LinearSlope_5D | **0.5612** | **0.7304** |
| LinearSlope_10D | **0.652** | **0.9636** |
| LinearSlope_20D | 0.4224 | **0.7804** |
| SharpRidge_5D | **0.598** | **0.7624** |
| SharpRidge_10D | **0.9536** | **1.0** |
| SharpRidge_20D | 0.3752 | **0.7652** |
| StepEllipsoidal_5D | **0.5112** | **0.6464** |
| StepEllipsoidal_10D | **0.8248** | **0.9844** |
| StepEllipsoidal_20D | **0.9204** | **0.9164** |
| RosenbrockRotated_5D | 0.3272 | **0.502** |
| RosenbrockRotated_10D | **0.682** | **0.8244** |
| RosenbrockRotated_20D | 0.1252 | 0.4424 |
| SchaffersIllConditioned_5D | 0.2932 | 0.4448 |
| SchaffersIllConditioned_10D | 0.1852 | 0.3816 |
| SchaffersIllConditioned_20D | 0.2016 | 0.3124 |
| LunacekBiR_5D | 0.4276 | 0.4684 |
| LunacekBiR_10D | 0.2708 | 0.4956 |
| LunacekBiR_20D | 0.134 | 0.4076 |
| GG101me_5D | **0.5288** | **0.5488** |
| GG101me_10D | **0.7212** | **0.8132** |
| GG101me_20D | 0.2344 | 0.3256 |
| GG21hi_5D | 0.4444 | 0.4464 |
| GG21hi_10D | 0.4248 | **0.5612** |
| GG21hi_20D | 0.2976 | **0.5764** |
| **Total p(PPO mf wIntraDeltaF + IntraDeltaX + InterDeltaX < iDE)** | 37.0% (17/46) | **63.0**% (29/46) |

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
                40,
                false,
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