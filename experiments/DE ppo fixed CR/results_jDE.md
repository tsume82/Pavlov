## Comparison Table

Probability of PPO trained policy outperforming CSA using 2 different metrics: Area under the curve and the absolute best of the run.
| Function    | p(PPO_fix_CR < jDE) with AUC metric | p(PPO_fix_CR < jDE) with best of the run metric |
| :---------- | ------------------------------ | ------------------------------- |
| BentCigar | 0.4136 | 0.4128 |
| Discus | 0.4712 | 0.1844 |
| Ellipsoid | 0.2084 | 0.0228 |
| Katsuura | **0.6164** | **0.5404** |
| Rastrigin | 0.328 | 0.384 |
| Rosenbrock | 0.3388 | 0.1208 |
| Schaffers | 0.1764 | 0.0572 |
| Schwefel | 0.396 | 0.226 |
| Sphere | 0.088 | 0.0352 |
| Weierstrass | 0.314 | 0.1992 |

## Plots

**for PPO, CR is fixed to 0.7**

##### BentCigar

![](BentCigar/jDE_BentCigar_comparison.png)

##### Discus

![](Discus/jDE_Discus_comparison.png)

##### Ellipsoid

![](Ellipsoid/jDE_Ellipsoid_comparison.png)

##### Katsuura

![](Katsuura/jDE_Katsuura_comparison.png)

##### Rastrigin

![](Rastrigin/jDE_Rastrigin_comparison.png)

##### Rosenbrock

![](Rosenbrock/jDE_Rosenbrock_comparison.png)

##### Schaffers

![](Schaffers/jDE_Schaffers_comparison.png)

##### Schwefel

![](Schwefel/jDE_Schwefel_comparison.png)

##### Sphere

![](Sphere/jDE_Sphere_comparison.png)

##### Weierstrass

![](Weierstrass/jDE_Weierstrass_comparison.png)

