## Comparison Table
| Function    | p(PPO < CSA) with AUC metric | p(PPO < CSA) with best of the run metric |
| :---------- | ------------------------------ | ------------------------------- |
| BentCigar | 0.2808 | **0.7736** |
| Discus | 0.2784 | 0.126 |
| Ellipsoid | 0.1256 | **0.6084** |
| Katsuura | 0.414 | 0.4532 |
| Rastrigin | **1.0** | **0.9984** |
| Rosenbrock | 0.2168 | **0.6032** |
| Schaffers | **0.8092** | **0.7616** |
| Schwefel | **0.9692** | 0.398 |
| Sphere | 0.094 | 0.4156 |
| Weierstrass | 0.4644 | **0.5176** |
| AttractiveSector_5D | **0.522** | 0.106 |
| AttractiveSector_10D | 0.4688 | 0.2348 |
| AttractiveSector_20D | **0.8056** | **0.5728** |
| BuecheRastrigin_5D | **0.5592** | 0.3972 |
| BuecheRastrigin_10D | 0.4476 | 0.216 |
| BuecheRastrigin_20D | **0.606** | **0.86** |
| CompositeGR_5D | 0.19 | 0.168 |
| CompositeGR_10D | 0.1856 | 0.2308 |
| CompositeGR_20D | 0.0488 | 0.0968 |
| DifferentPowers_5D | 0.1588 | 0.0 |
| DifferentPowers_10D | 0.4988 | 0.3316 |
| DifferentPowers_20D | 0.0752 | 0.1116 |
| LinearSlope_5D | **0.858** | 0.0 |
| LinearSlope_10D | **0.922** | 0.0 |
| LinearSlope_20D | **1.0** | 0.0 |
| SharpRidge_5D | 0.1044 | 0.0684 |
| SharpRidge_10D | 0.3308 | 0.336 |
| SharpRidge_20D | **0.6376** | **0.8632** |
| StepEllipsoidal_5D | 0.4752 | 0.2256 |
| StepEllipsoidal_10D | **0.7268** | 0.298 |
| StepEllipsoidal_20D | 0.3372 | 0.308 |
| RosenbrockRotated_5D | 0.0 | 0.046 |
| RosenbrockRotated_10D | 0.0 | 0.0 |
| RosenbrockRotated_20D | 0.0012 | 0.0192 |
| SchaffersIllConditioned_5D | **0.9172** | **0.7752** |
| SchaffersIllConditioned_10D | 0.2132 | 0.3416 |
| SchaffersIllConditioned_20D | **0.7448** | **0.7948** |
| LunacekBiR_5D | 0.4704 | 0.4708 |
| LunacekBiR_10D | 0.4652 | **0.618** |
| LunacekBiR_20D | 0.278 | 0.4796 |
| GG101me_5D | **0.8572** | **0.7928** |
| GG101me_10D | **0.5936** | 0.404 |
| GG101me_20D | 0.21 | 0.4268 |
| GG21hi_5D | **0.8536** | **0.6944** |
| GG21hi_10D | **0.7388** | 0.4536 |
| GG21hi_20D | 0.0848 | 0.342 |
| **Total p(PPO < CSA)** | 39.1% (18/46) | 30.4% (14/46) |

## Configuration

```json
{
    "agent.algorithm": "RayProximalPolicyOptimization",
    "agent.algorithm.render_env": false,
    "agent.algorithm.num_workers": 0,
    "agent.algorithm.batch_mode": "complete_episodes",
    "agent.algorithm.lr": 1e-05,
    "agent.algorithm.train_batch_size": 200,
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
        "solver_driver": "CMAdriver",
        "solver_driver_args": [
            10,
            10,
            12,
            0.5
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
                true
            ],
            [
                {
                    "step_size": {
                        "max": 3,
                        "min": 0
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
            "step_size": {
                "max": 3,
                "min": 1e-05
            }
        }
    }
}
```