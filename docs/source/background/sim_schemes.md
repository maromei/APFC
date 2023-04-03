# Simulation Schemes

(ch:scheme_ampl)=
## Just the amplitudes

We use the gradient flow for the amplitudes (eq. {eq}`eqn:apfc_flow`) and
[this](ch:fourier_etd) fourier scheme. The linear and non linear part
are chosen as follows:

$$
\begin{gathered}
\mathcal{L} = - |\boldsymbol{G}_m|^2
\left[ A \mathcal{G}^2 + B \right] \\
N = - |\boldsymbol{G}_m|^2 \left[
    3 D (\Phi - |\eta_m|^2) \eta_m + \frac{\partial f^s}{\partial \eta_m^*}
\right]
\end{gathered}
$$ (eqn:linnonlinjustamplitudes)

Under the fourier transform in the linear part, $B$ stays constant, and
$\mathcal{G}^2$ becomes:

$$
\begin{align}
\mathcal{G}_m &= \sqrt\beta \nabla^2 + 2 \mathbb{i} \boldsymbol{G}_m \nabla \\
\widehat{\mathcal{G}_m^2} &=
\left[ \sqrt\beta\left(k_x^2 + k_y^2\right) + 2 G_m^{(x)} k_x + 2 G_m^{(y)} k_y \right]^2
\end{align}
$$ (eqn:g_sq_op_fourier)

Where $k_x$ and $k_y$ are the frequency variables.

(ch:scheme_ampl_n0)=
## Amplitudes and average density

For both the amplitudes and the densities
[this](ch:fourier_imex) scheme is used.

For the amplitudes the linear and non-linear part can not be chosen in the
same way as in eq. {eq}`eqn:linnonlinjustamplitudes`, since $B$ contains
$n_0$. This is why $B$ is split into the linear and non-linear parts:

$$
\begin{gathered}
\mathcal{L} = - |\boldsymbol{G}_m|^2
\left[ A \mathcal{G}^2 + \Delta B^0 \right] \\
N = - |\boldsymbol{G}_m|^2 \left[
    (3 v n_0^2 - 2 t n_0) \eta_m +
    3 D (\Phi - |\eta_m|^2) \eta_m +
    \frac{\partial f^s}{\partial \eta_m^*}
\right]
\end{gathered}
$$ (eqn:linnonlinn0amplitudes)

For the average density (eq. {eq}`eqn:apfc_n0_flow`)
only $(\Delta B^0 + B^x) n_0$ can be used as the linear part.

$$
\begin{gathered}
\mathcal{L} = \Delta B^0 + B^x \\
N = 3 v \Phi n_0
+ 3 v P - \Phi t
- t n_0^2 + v n_0^3
\end{gathered}
$$ (eqn:linnonlinn0)

The $\nabla^2$ operator was excluded from the equation above. Since $N$ needs
to first be calculated and then be fourier-transformed, the nabla operator will
just be applied after the transformation. Resulting in the scheme:

$$
\begin{gathered}
\widehat{\phi}_{t+1} = \frac{
    \widehat{\phi}_t + \tau \widehat{\nabla^2} \widehat{N}_t
} {
    1 - \tau \widehat{\nabla^2} \widehat{\mathcal{L}}
} \\
\text{with} \quad
\widehat{\nabla^2} = -(k_x^2 + k_y^2)
\end{gathered}
$$ (eqn:n0sim_imex_scheme)
