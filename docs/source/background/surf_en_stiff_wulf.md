# Surface Energy, Stiffness ans Wulff Shape

This section is based on {cite:t}`2018Ofori_anisotrop_pfc`.

(ch:surf_en)=
## Surface Energy

According to {cite:t}`2018Ofori_anisotrop_pfc`, the surface energy
$\gamma$ can be defined as

$$
\begin{equation}
\gamma = \frac{\Omega - \Omega_\nu}{A}
\end{equation}
$$(eqn:general_surf_en_definition)

Where $A$ is the area of the interface, $\Omega$ is the total grand potential,
and $\Omega_\nu$ is the grand potential of one of the bulk phases. Where
$\nu$ can be either the liquid $l$ or solid $s$ phase.

They also define a way to calculate the grand potential as follows
{cite:p}`2018Ofori_anisotrop_pfc`:

$$
\begin{equation}
\Omega = F - \mu \int \mathrm{d}\boldsymbol{r} \; n_0(\boldsymbol{r})
\end{equation}
$$

Where $F$ is the functional in {eq}`eqn:apfc_energy_functional`, $\mu$ is the
chemical potential and $n_0(\boldsymbol{r})$ is the spacially dependent
density.

Their assumption was that $n_0$ is always spatially dependent. This also shows
in their definition for the chemical potential
$\mu = (f_s - f_l) / (n_{0, s} - n_{0, l})$. Here they used the
"tangent rule"{cite:p}`2018Ofori_anisotrop_pfc`, where $f_\nu$ is the integrand
of {eq}`eqn:apfc_energy_functional`. The indecies $s, l$ denote that it is
evaluated somewhere in the solid or liquid phase respectively, assuming
equilibrium.

If we wanted to include the case of the average densities being set constant
everywhere, then another definition for $\mu$ needs to be chosen. One of these
choices is:

$$
\begin{equation}
\mu(\boldsymbol{r}) = \frac{\delta F}{\delta n_0} =
\frac{\partial f}{\partial n_0}
\end{equation}
$$ (eqn:chem_pot_def_variation)

Since it is now spatially dependent, $\mu$ will be included in the integrand <br>
For $\Omega_\nu$ the liquid phase $\nu = l$ was chosen, since it can be
computed by setting the amplitudes to be 0 everywhere. <br>
This results in the following definition for the surface energy:

$$
\begin{equation}
\gamma = \frac{1}{A} \int \mathrm{d} \boldsymbol{r} \left[
    f - \mu n_0
    - f_l + \mu_l n_{0, l}
\right]
\end{equation}
$$ (eqn:surf_en_calc)

```{eval-rst}
.. todo:: show plots equal to other paper
```

```{eval-rst}
.. todo:: this entire thing was done to include const n0 --> show how these behave.
```

- growing $\gamma_0$
- $\varepsilon$ settles to one value?

- validity of pre equilib compute:
  - $n_{0, l}$ assumed equilib. -> already stable in sim, grain just grows
    - usable after the first couple timesteps
    - and $f$ grows with volume. $f_l$ too since n0 is const.?

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
