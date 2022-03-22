## Comparison Table
| Function    | p(PPO mf wIntraDeltaF + IntraDeltaX + InterDeltaX < jDE) with AUC metric | p(PPO mf wIntraDeltaF + IntraDeltaX + InterDeltaX < jDE) with best of the run metric |
| :---------- | ------------------------------ | ------------------------------- |
| BentCigar | 0.2772 | 0.4428 |
| Discus | 0.4648 | 0.4356 |
| Ellipsoid | 0.2124 | 0.314 |
| Katsuura | 0.484 | 0.4476 |
| Rastrigin | 0.2416 | 0.3976 |
| Rosenbrock | 0.2424 | 0.388 |
| Schaffers | 0.3328 | 0.3672 |
| Schwefel | 0.312 | 0.2228 |
| Sphere | 0.1188 | 0.2908 |
| Weierstrass | **0.6628** | **0.6832** |
| AttractiveSector_5D | 0.4856 | 0.4108 |
| AttractiveSector_10D | 0.3292 | 0.3904 |
| AttractiveSector_20D | 0.1408 | 0.3028 |
| BuecheRastrigin_5D | 0.3132 | 0.4516 |
| BuecheRastrigin_10D | 0.2772 | 0.4576 |
| BuecheRastrigin_20D | 0.1872 | 0.3288 |
| CompositeGR_5D | 0.3296 | 0.3768 |
| CompositeGR_10D | 0.2184 | 0.3084 |
| CompositeGR_20D | 0.1096 | 0.248 |
| DifferentPowers_5D | 0.2472 | 0.4916 |
| DifferentPowers_10D | 0.2756 | 0.4044 |
| DifferentPowers_20D | 0.184 | 0.3724 |
| LinearSlope_5D | 0.3596 | 0.1892 |
| LinearSlope_10D | 0.3612 | **0.5216** |
| LinearSlope_20D | 0.1568 | 0.2764 |
| SharpRidge_5D | 0.232 | 0.4524 |
| SharpRidge_10D | 0.2552 | 0.4548 |
| SharpRidge_20D | 0.2128 | 0.448 |
| StepEllipsoidal_5D | 0.3332 | **0.6004** |
| StepEllipsoidal_10D | 0.3352 | **0.6124** |
| StepEllipsoidal_20D | 0.2016 | 0.3884 |
| RosenbrockRotated_5D | 0.3604 | 0.49 |
| RosenbrockRotated_10D | 0.1976 | 0.4996 |
| RosenbrockRotated_20D | 0.1216 | 0.1776 |
| SchaffersIllConditioned_5D | 0.3288 | 0.4548 |
| SchaffersIllConditioned_10D | 0.242 | 0.3072 |
| SchaffersIllConditioned_20D | 0.2796 | 0.3248 |
| LunacekBiR_5D | 0.3144 | 0.3504 |
| LunacekBiR_10D | 0.216 | 0.3392 |
| LunacekBiR_20D | 0.1708 | 0.3024 |
| GG101me_5D | 0.4804 | **0.5812** |
| GG101me_10D | 0.372 | **0.5008** |
| GG101me_20D | 0.2352 | 0.3596 |
| GG21hi_5D | 0.4016 | 0.4592 |
| GG21hi_10D | 0.2928 | 0.4156 |
| GG21hi_20D | 0.3288 | 0.3736 |
| **Total p(PPO mf wIntraDeltaF + IntraDeltaX + InterDeltaX < jDE)** | 2.2% (1/46) | 13.0% (6/46) |

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