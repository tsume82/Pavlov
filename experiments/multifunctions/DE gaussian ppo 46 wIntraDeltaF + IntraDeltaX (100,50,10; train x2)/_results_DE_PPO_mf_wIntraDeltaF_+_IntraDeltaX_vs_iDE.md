## Comparison Table
| Function    | p(PPO mf wIntraDeltaF + IntraDeltaX < iDE) with AUC metric | p(PPO mf wIntraDeltaF + IntraDeltaX < iDE) with best of the run metric |
| :---------- | ------------------------------ | ------------------------------- |
| BentCigar | 0.298 | **0.546** |
| Discus | 0.4584 | **0.6456** |
| Ellipsoid | 0.3176 | **0.6856** |
| Katsuura | 0.4632 | **0.5044** |
| Rastrigin | 0.4444 | **0.7028** |
| Rosenbrock | 0.2828 | **0.5864** |
| Schaffers | 0.4172 | **0.5716** |
| Schwefel | 0.3772 | **0.7324** |
| Sphere | 0.3564 | **0.5444** |
| Weierstrass | **0.618** | **0.6544** |
| AttractiveSector_5D | 0.3588 | 0.4748 |
| AttractiveSector_10D | **0.5184** | **0.7204** |
| AttractiveSector_20D | **0.9248** | **0.9032** |
| BuecheRastrigin_5D | 0.4692 | 0.2284 |
| BuecheRastrigin_10D | **0.794** | **0.9024** |
| BuecheRastrigin_20D | 0.2492 | 0.3788 |
| CompositeGR_5D | 0.3036 | 0.4756 |
| CompositeGR_10D | 0.2076 | 0.4124 |
| CompositeGR_20D | **0.5124** | **0.748** |
| DifferentPowers_5D | **0.6292** | **0.8696** |
| DifferentPowers_10D | **0.5676** | 0.4564 |
| DifferentPowers_20D | 0.144 | 0.3276 |
| LinearSlope_5D | **0.5236** | **0.7548** |
| LinearSlope_10D | **0.6908** | **0.964** |
| LinearSlope_20D | 0.3808 | **0.6316** |
| SharpRidge_5D | **0.5916** | **0.6788** |
| SharpRidge_10D | **0.9476** | **0.9992** |
| SharpRidge_20D | 0.3364 | **0.578** |
| StepEllipsoidal_5D | **0.5176** | **0.6244** |
| StepEllipsoidal_10D | **0.782** | **0.9652** |
| StepEllipsoidal_20D | **0.9124** | **0.9356** |
| RosenbrockRotated_5D | 0.3576 | 0.4996 |
| RosenbrockRotated_10D | **0.6996** | **0.8824** |
| RosenbrockRotated_20D | 0.1448 | 0.3084 |
| SchaffersIllConditioned_5D | 0.2616 | 0.3684 |
| SchaffersIllConditioned_10D | 0.2032 | 0.3928 |
| SchaffersIllConditioned_20D | 0.2648 | 0.39 |
| LunacekBiR_5D | 0.4628 | **0.586** |
| LunacekBiR_10D | 0.378 | **0.5812** |
| LunacekBiR_20D | 0.1908 | 0.3736 |
| GG101me_5D | **0.5988** | **0.562** |
| GG101me_10D | **0.7664** | **0.7732** |
| GG101me_20D | 0.2012 | 0.2132 |
| GG21hi_5D | 0.4872 | 0.4924 |
| GG21hi_10D | 0.3896 | 0.4976 |
| GG21hi_20D | 0.3396 | 0.4764 |
| **Total p(PPO mf wIntraDeltaF + IntraDeltaX < iDE)** | 37.0% (17/46) | **63.0**% (29/46) |

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