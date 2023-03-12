# Surface Energy, Stiffness ans Wulff Shape

## Surface Energy

```{eval-rst}
.. todo:: surface energy derivation
```

For the energy function in eq. {eq}`eqn:apfc_energy_functional`:

$$
\begin{equation}
\gamma(\theta) = 2 A \int \mathrm{d}x \sum\limits_m \left[
    4 (G_m^{x} \cos \theta + G_m^{(y)} \sin \theta)^2
    \left( \frac{\partial \eta_m}{\partial x} \right)^2 +
    2 \left( \frac{\partial^2 \eta_m}{\partial x^2} \right)^2
    - \frac{\partial \eta_m}{\partial x} \frac{\partial^3 \eta_m}{\partial x^3}
\right]
\end{equation}
$$ (eqn:surf_en_calc_1d)

## Stiffness

```{eval-rst}
.. todo:: stiffness section
```

Based on {cite:t}`2018Ofori_anisotrop_pfc` the stiffness $\Gamma$
can be calculated from the surface energy in the following way:

$$
\begin{equation}
\Gamma(\theta) = \gamma(\theta) + \gamma^{\prime\prime}(\theta)
\end{equation}
$$ (eqn:stiffness)

## Wulff Shape

```{eval-rst}
.. todo:: wulff shape section
```

Based on {cite:t}`2018Ofori_anisotrop_pfc` the Wulff-shape
can be calculated from the surface energy in the following way:

$$
\begin{aligned}
x(s) &= \gamma(\theta) \cos \theta - \gamma^\prime(\theta) \sin \theta \\
y(s) &= \gamma(\theta) \sin \theta - \gamma^\prime(\theta) \cos \theta
\end{aligned}
$$ (eqn:wulff_shape)

where $s$ is the arclength.

## Fits

For a triangular lattice the surface energy can be fit to
{cite:p}`2018Ofori_anisotrop_pfc`

$$
\begin{equation}
\gamma(\theta) = \gamma_0 \left(
    1 + \sum\limits_{u=1}^N \varepsilon \cos(6 u \theta)
\right)
\end{equation}
$$ (eqn:surf_en_theo_triangular)

where $N$ is the order of the fit.
This gives 2 parameter to characterize the surface energy.
