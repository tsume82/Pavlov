## Comparison Table
| Function    | p(PPO mf wIntraDeltaF + IntraDeltaX + InterDeltaX < iDE) with AUC metric | p(PPO mf wIntraDeltaF + IntraDeltaX + InterDeltaX < iDE) with best of the run metric |
| :---------- | ------------------------------ | ------------------------------- |
| BentCigar | 0.4656 | 0.4104 |
| Discus | 0.4748 | **0.7364** |
| Ellipsoid | 0.4424 | **0.5196** |
| Katsuura | 0.4188 | 0.4168 |
| Rastrigin | **0.5988** | **0.7356** |
| Rosenbrock | 0.3128 | 0.4416 |
| Schaffers | 0.4336 | 0.4608 |
| Schwefel | 0.4 | **0.5728** |
| Sphere | 0.4192 | 0.3876 |
| Weierstrass | **0.6096** | **0.7476** |
| AttractiveSector_5D | 0.418 | 0.2436 |
| AttractiveSector_10D | **0.5712** | **0.7596** |
| AttractiveSector_20D | **0.9352** | **0.762** |
| BuecheRastrigin_5D | 0.4112 | 0.0788 |
| BuecheRastrigin_10D | **0.7808** | **0.6728** |
| BuecheRastrigin_20D | 0.3624 | 0.1764 |
| CompositeGR_5D | 0.3844 | **0.5048** |
| CompositeGR_10D | 0.2224 | 0.3816 |
| CompositeGR_20D | **0.5712** | **0.712** |
| DifferentPowers_5D | **0.6244** | **0.6444** |
| DifferentPowers_10D | **0.5168** | 0.1096 |
| DifferentPowers_20D | 0.2804 | 0.2348 |
| LinearSlope_5D | **0.6228** | **0.6964** |
| LinearSlope_10D | **0.6932** | **0.91** |
| LinearSlope_20D | 0.4316 | 0.4144 |
| SharpRidge_5D | **0.7304** | **0.7168** |
| SharpRidge_10D | **0.9288** | **0.98** |
| SharpRidge_20D | 0.4152 | 0.468 |
| StepEllipsoidal_5D | **0.5184** | 0.3932 |
| StepEllipsoidal_10D | **0.8004** | **0.8808** |
| StepEllipsoidal_20D | **0.8952** | **0.7396** |
| RosenbrockRotated_5D | 0.368 | 0.4036 |
| RosenbrockRotated_10D | **0.802** | **0.696** |
| RosenbrockRotated_20D | 0.2412 | 0.2328 |
| SchaffersIllConditioned_5D | 0.3592 | 0.2604 |
| SchaffersIllConditioned_10D | 0.2768 | 0.3164 |
| SchaffersIllConditioned_20D | 0.3504 | 0.4332 |
| LunacekBiR_5D | 0.4564 | **0.5476** |
| LunacekBiR_10D | 0.4564 | **0.616** |
| LunacekBiR_20D | 0.2904 | 0.4616 |
| GG101me_5D | **0.5844** | 0.4388 |
| GG101me_10D | **0.8128** | **0.7816** |
| GG101me_20D | 0.2756 | 0.2024 |
| GG21hi_5D | **0.5464** | 0.4552 |
| GG21hi_10D | 0.396 | 0.336 |
| GG21hi_20D | 0.3484 | 0.4012 |
| **Total p(PPO mf wIntraDeltaF + IntraDeltaX + InterDeltaX < iDE)** | 41.3% (19/46) | 45.7% (21/46) |

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