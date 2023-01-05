# Finite Differences

$$
\begin{equation}
\frac{\partial \eta_m}{\partial t} \approx
- | \boldsymbol{G}_m |^2 \left[
    A \mathcal{G}_m^2 \eta_m + B \eta_m + 3 D (\Phi - |\eta_m|^2) \eta_m + \frac{\partial f^s}{\partial \eta_m^*}
\right]
\end{equation}
$$ (eqn:apfc_flow_fd)

With

$$
\begin{aligned}
A &= B^x \\
B &= \Delta B^0 - 2 t n_0 + 3 v n_0^2 \\
C &= - (t + 3 n_0) \\
D &= v \\
\Phi &= 2 \sum\limits_m^M |\eta_m|^2 \\
\mathcal{G}_m &= \nabla^2 + 2 \mathbb{i} \boldsymbol{G}_m \nabla
\end{aligned}
$$ (eqn:apfc_flow_constants_fd)

For a triangular crystal with one-mode approx.:

$$
\begin{gathered}
f = 2 C (\eta_1 \eta_2 \eta_3 + \eta_1^* \eta_2^* \eta_3^*) \\
\boldsymbol{G}_1 = \begin{bmatrix} - \sqrt{3} / 2 \\ - 1 / 2 \end{bmatrix}, \quad
\boldsymbol{G}_2 = \begin{bmatrix} 0 \\ 1 \end{bmatrix}, \quad
\boldsymbol{G}_3 = \begin{bmatrix} \sqrt{3} / 2 \\ - 1 / 2 \end{bmatrix}
\end{gathered}
$$ (eqn:onemodetriangluar_fd)

Simplification:

$$
\begin{gathered}
N(\eta_m) := 3 D (\Phi - |\eta_m|^2) \eta_m + \frac{\partial f^s}{\partial \eta_m^*}
\end{gathered}
$$

## Operator Approximation

Only $\mathcal{G}_m = \nabla^2 + 2 \mathbb{i} \boldsymbol{G}_m \nabla$ needs to be approximated.

Laplace operator via five point stencil:

$$
\begin{gathered}
\nabla^2 f(x, y) \approx \frac{
    f(x-h, y) + f(x + h, y) + f(x, y-h) + f(x, y+h) - 4 f(x, y)
}{
    h^2
}
\end{gathered}
$$

$\boldsymbol{G}_m \nabla$ via central difference:

$$
\begin{gathered}
f^\prime (x) \approx \frac{f(x+h) - f(x-h)}{2 h} \\
\boldsymbol{G}_m \nabla f(x, y) \approx
G_m^x \frac{f(x+h,y) - f(x-h,y)}{2 h} +
G_m^y \frac{f(x,y+h) - f(x,y-h)}{2 h}
\end{gathered}
$$

Idea for BC: use forward / backward difference on boundary.

## Discretization

$$
\begin{cases}
    \partial_t \eta_m = - | \boldsymbol{G}_m |^2 \left[
        A \mathcal{G}_m \lambda_m + B \eta_m + N(\eta_m)
    \right] \\
    \lambda_m = \mathcal{G}_m \eta_m
\end{cases}
$$

### Splitting which does nothing

$$
\begin{cases}
    \eta_m^{n+1} = \eta_m^n - | \boldsymbol{G}_m |^2 \mathrm{d}t \left[
        A \mathcal{G}_m \lambda_m^{n+1} + B \eta_m^{n+1} + N^n(\eta_m)
    \right] \\
    \lambda^{n+1}_m = \mathcal{G}_m \eta^{n}_m
\end{cases}
$$

### Probably the right way

$$
\begin{cases}
    \eta_m^{n+1} = \eta_m^n - | \boldsymbol{G}_m |^2 \mathrm{d}t \left[
        A \mathcal{G}_m \lambda_m^{n+1} + B \eta_m^{n+1} + N^n(\eta_m)
    \right] \\
    \lambda^{n+1}_m = \mathcal{G}_m \eta^{n+1}_m
\end{cases}
$$

Which leads to

$$
\begin{cases}
    \eta_m^{n+1} = F \eta_m^n - | \boldsymbol{G}_m |^2 \mathrm{d}t F \left[
        A \mathcal{G}_m \lambda_m^{n+1} + N^n(\eta_m)
    \right] \\
    \lambda^{n+1}_m = \mathcal{G}_m \eta^{n+1}_m
\end{cases}
$$

with $F = (1 + |\boldsymbol{G}_m|^2 B \mathrm{d}t)^{-1}$

In Matrix form:

$$
\begin{bmatrix}
1 & F | \boldsymbol{G}_m |^2 \mathrm{d}t A \mathcal{G}_m \\
- \mathcal{G}_m & 1
\end{bmatrix}
\begin{bmatrix}
\eta_m^{n+1} \\
\lambda_m^{n+1}
\end{bmatrix}
=
\begin{bmatrix}
F (\eta_m^n - | \boldsymbol{G}_m |^2 \mathrm{d}t N^n(\eta_m) ) \\
0
\end{bmatrix}
$$
