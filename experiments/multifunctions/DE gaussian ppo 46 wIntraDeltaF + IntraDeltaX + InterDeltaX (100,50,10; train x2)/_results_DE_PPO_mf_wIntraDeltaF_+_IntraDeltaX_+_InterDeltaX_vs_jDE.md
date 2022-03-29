## Comparison Table
| Function    | p(PPO mf wIntraDeltaF + IntraDeltaX + InterDeltaX < jDE) with AUC metric | p(PPO mf wIntraDeltaF + IntraDeltaX + InterDeltaX < jDE) with best of the run metric |
| :---------- | ------------------------------ | ------------------------------- |
| BentCigar | 0.4328 | 0.3816 |
| Discus | **0.5176** | 0.4384 |
| Ellipsoid | 0.4036 | 0.3124 |
| Katsuura | **0.5152** | 0.4868 |
| Rastrigin | **0.5464** | **0.6508** |
| Rosenbrock | 0.3808 | 0.33 |
| Schaffers | 0.3744 | 0.3532 |
| Schwefel | 0.3972 | 0.2984 |
| Sphere | 0.284 | 0.1996 |
| Weierstrass | **0.5612** | **0.76** |
| AttractiveSector_5D | **0.5068** | 0.408 |
| AttractiveSector_10D | 0.3832 | 0.4952 |
| AttractiveSector_20D | 0.2544 | 0.3096 |
| BuecheRastrigin_5D | 0.3484 | 0.3256 |
| BuecheRastrigin_10D | 0.3768 | 0.3628 |
| BuecheRastrigin_20D | 0.314 | 0.2924 |
| CompositeGR_5D | 0.3572 | **0.528** |
| CompositeGR_10D | 0.2512 | 0.37 |
| CompositeGR_20D | 0.2708 | 0.41 |
| DifferentPowers_5D | 0.254 | **0.5204** |
| DifferentPowers_10D | 0.3756 | 0.2824 |
| DifferentPowers_20D | 0.4436 | 0.4488 |
| LinearSlope_5D | 0.4332 | 0.1768 |
| LinearSlope_10D | **0.506** | 0.4564 |
| LinearSlope_20D | 0.3648 | 0.2808 |
| SharpRidge_5D | 0.3816 | **0.5532** |
| SharpRidge_10D | 0.4336 | **0.5048** |
| SharpRidge_20D | 0.3932 | 0.4072 |
| StepEllipsoidal_5D | 0.4116 | 0.4764 |
| StepEllipsoidal_10D | 0.486 | **0.566** |
| StepEllipsoidal_20D | 0.3456 | 0.3644 |
| RosenbrockRotated_5D | 0.36 | 0.466 |
| RosenbrockRotated_10D | 0.3664 | 0.4032 |
| RosenbrockRotated_20D | 0.2776 | 0.244 |
| SchaffersIllConditioned_5D | 0.4428 | 0.4148 |
| SchaffersIllConditioned_10D | 0.4532 | 0.4748 |
| SchaffersIllConditioned_20D | **0.5124** | **0.5828** |
| LunacekBiR_5D | 0.3452 | 0.4384 |
| LunacekBiR_10D | 0.4472 | **0.5436** |
| LunacekBiR_20D | 0.3688 | 0.4912 |
| GG101me_5D | **0.5092** | **0.5296** |
| GG101me_10D | 0.4916 | **0.5536** |
| GG101me_20D | 0.3492 | 0.328 |
| GG21hi_5D | 0.4964 | 0.4972 |
| GG21hi_10D | 0.4 | 0.3764 |
| GG21hi_20D | 0.4452 | 0.4152 |
| **Total p(PPO mf wIntraDeltaF + IntraDeltaX + InterDeltaX < jDE)** | 17.4% (8/46) | 23.9% (11/46) |

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