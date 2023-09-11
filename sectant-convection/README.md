# Sphere sectant convection cell modelling with DOLFINx

This implementation provides a 3D solver for a simple mantle
convection model.

![Sectant convection cell](img/sectant_convection.gif)

# Model description

Let $\Omega$ be the domain of interest with boundary
$\partial \Omega$ with outward pointing unit vector $\vec{n}$
and tangential unit vectors $\vec{\tau}_i$, $i=1,2$.
In $\Omega$ we seek finite element approximation of velocity, pressure
and temperature, $\vec{u}$, $p$ and $T$, respectively, such that

$$
\begin{align}
-\mu \nabla^2 \vec{u}_h + \nabla p_h &= \mathrm{Ra} \\: T_h \\: \hat{\vec{r}}, \\
-\nabla \cdot \vec{u}_h &= 0, \\
-\nabla^2 T_h + \vec{u}_h \cdot \nabla T_h &= 0,
\end{align}
$$

subject to the boundary conditions

$$
\begin{align}
\vec{u}_h \cdot \vec{n} &= 0 \text{ on } \partial\Omega, \\
(\mu \nabla \vec{u}_h \cdot \vec{n}) \cdot \vec{\tau}_i &= 0 \text{ on } \partial\Omega, \\: i = 1, 2 \\
T_h &= 0 \text{ on } \partial\Omega\_\text{surface}, \\
T_h &= 1 \text{ on } \partial\Omega\_\text{CMB}, \\
\nabla T_h \cdot \vec{n} &= 0 \text{ on } \partial\Omega\_\text{symmetry}.
\end{align}
$$

Here $\mu$ is the constant viscosity, $\mathrm{Ra}$ is the Rayleigh number.
Futhermore the domain boundary has been divided into nonempty
components $\partial\Omega = \partial\Omega\_\text{surface}
\cup \partial\Omega\_\text{CMB} \cup \partial\Omega\_\text{symmetry}$
corresponding to the Earth's surface, its core-mantle-boundary and
symmetry components of the sectant, respectively.
