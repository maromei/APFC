# Fourier

## Base equation

General Representation

$$
\begin{equation}
\frac{\partial \eta_m}{\partial t} \approx
- | \boldsymbol{G}_m |^2 \left[
    A \mathcal{G}_m^2 \eta_m + B(t) \eta_m + 3 D (\Phi - |\eta_m|^2) \eta_m + \frac{\partial f^s}{\partial \eta_m^*}
\right]
\end{equation}
$$ (eqn:apfc_flow)

With

$$
\begin{aligned}
A &= B^x \\
B(t) &= \Delta B^0 - 2 t n_0 + 3 v n_0^2 \\
C(t) &= - (t + 3 n_0) \\
D &= v \\
\Phi &= 2 \sum\limits_m^M |\eta_m|^2 \\
\mathcal{G}_m &= \nabla^2 + 2 \mathbb{i} \boldsymbol{G}_m \nabla
\end{aligned}
$$ (eqn:apfc_flow_constants)

For a triangular crystal with one-mode approx.:

$$
\begin{gathered}
f = 2 C(t) (\eta_1 \eta_2 \eta_3 + \eta_1^* \eta_2^* \eta_3^*) \\
\boldsymbol{G}_1 = \begin{bmatrix} - \sqrt{3} / 2 \\ - 1 / 2 \end{bmatrix}, \quad
\boldsymbol{G}_2 = \begin{bmatrix} 0 \\ 1 \end{bmatrix}, \quad
\boldsymbol{G}_3 = \begin{bmatrix} \sqrt{3} / 2 \\ - 1 / 2 \end{bmatrix}
\end{gathered}
$$ (eqn:onemodetriangluar)

## General Fourier Approach

For fourier method transform the base equation into the form

$$
\begin{equation}
\frac{\partial \eta_m}{\partial t} = \mathcal{L}_m \eta_m + N(\eta_m)
\end{equation}
$$ (eq:ode_form)

where $\mathcal{L}_m$ is a linear operator and $N(\eta_m)$ is a non-linear function of $\eta_m$.
Then fourier transform it.

$$
\begin{equation}
\frac{\partial \widehat{\eta}_k}{\partial t} = \mathcal{L}_k \widehat{\eta}_k + \widehat{N}_k
\end{equation}
$$ (eqn:ode_fourier_form)

This ODE is solved by

$$
\begin{equation}
\widehat{\eta}_k (t) = e^{\mathcal{L}_k t} \widehat{\eta}_k(0) + e^{\mathcal{L}_k t}
\int\limits_0^t \mathcal{d}t^\prime e^{- \mathcal{L}_k t^\prime} \widehat{N}_k(t^\prime)
\end{equation}
$$

under the assumption that $\mathcal{L}_k$ does not depend on time. With another approximation of
$\widehat{N}_k(t^\prime) \approx \widehat{N}_k(t)$ the equation for $\widehat{\eta}_k (t + \Delta t)$ reads

$$
\begin{equation}
\widehat{\eta}_k (t + \Delta t) =
e^{\mathcal{L}_k \Delta t} \widehat{\eta}_k(t) +
\frac{e^{\mathcal{L}_k \Delta t} - 1}{\mathcal{L}_k} \widehat{N}_k(t)
\end{equation}
$$ (eqn:fourier_approx_sol)

## Fourier Method applied to Base equation

Expanding equation {eq}`eqn:apfc_flow` with $\Phi$ from {eq}`eqn:apfc_flow_constants`.

$$
\begin{equation}
\frac{\partial \eta_m}{\partial t} =
- |\boldsymbol{G}_m|^2 \left[
    A \mathcal{G}_m^2 \eta_m + B(t) \eta_m
    + 3 D (2 \sum\limits_{i \neq m}|\eta_i|^2 + |\eta_m|^2) \eta_m
    + \frac{\partial f^s}{\partial \eta_m^*}
\right]
\end{equation}
$$

Then the summands are separated into linear and nonlinear parts. <br>
Decoupling the Amplitudes from each other allows part of $\Phi$ to be moved
to the linear part. <br>
The summand with $B(t)$ depends on time. However, {eq}`eqn:fourier_approx_sol` assumes $\mathcal{L}_k$
is not dependent on time. Thats why I split $B(t)$ into one constant $\gamma$ and one time dependant part $\lambda (t)$.

$$
\begin{align}
B(t) &= \Delta B^0 - 2 t n_0 + 3 v n_0^2 \\
&= \gamma + \lambda(t) \\
\gamma &= \Delta B^0 + 3 v n_0^2 \\
\lambda &= - 2 t n_0
\end{align}
$$ (eqn:B_t_split)

Just reordering eq. {eq}`eqn:apfc_flow` gives:

$$
\begin{equation}
\frac{\partial \eta_m}{\partial t} =
- |\boldsymbol{G}_m|^2 \left[
    \left( A \mathcal{G}_m^2 + \gamma + 6 D \sum\limits_{i \neq m}|\eta_i|^2 \right) \eta_m
    + \left( 3 D |\eta_m|^2 + \lambda(t) \right) \eta_m
    + \frac{\partial f^s}{\partial \eta_m^*}
\right]
\end{equation}
$$

This defines $\mathcal{L}_m$ and $N(\eta_m)$

$$
\begin{align}
\mathcal{L}_m &= - |\boldsymbol{G}_m|^2 \left[
    A \mathcal{G}_m^2 + \gamma + 6 D \sum\limits_{i \neq m}|\eta_i|^2
\right]
\\
N(\eta_m) &= - |\boldsymbol{G}_m|^2 \left[
    \left( 3 D |\eta_m|^2 + \lambda(t) \right) \eta_m + \frac{\partial f^s}{\partial \eta_m^*}
\right]
\end{align}
$$ (eqn:lin_non_lin_part)

The next questions is how these operators transform to the fourier space.

$$
\begin{align}
\mathcal{G}_m^2 &= \left( \nabla^2 + 2 \mathbb{i} \boldsymbol{G}_m \nabla \right)^2 \\
&= \nabla^4 + 2 i \boldsymbol{G}_m \nabla^3 - 4 |\boldsymbol{G}_m|^2 \nabla^2 \\
\widehat{\mathcal{G}_m^2}(\boldsymbol{k}) &=
k^4_x + k^4_y
+ 2 |\boldsymbol{G}_m| \left( k^3_x + k^3_y \right)
+ 4 |\boldsymbol{G}_m|^2 \left( k^2_x + k^2_y \right)
\end{align}
$$
