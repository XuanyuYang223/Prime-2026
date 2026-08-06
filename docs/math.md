# Math used in this repository

## Video representation

The complete clip is one tensor,

$$x \in \mathbb{R}^{C \times T \times H \times W},$$

rather than a collection of independently generated frames. The included checkpoint uses one grayscale channel, 12 frames, and a resolution of 64 by 128 pixels.

The condition $c$ concatenates three tensors for each character: a canonical glyph, a Gaussian position heatmap, and an aligned glyph layout. With at most $K=8$ characters, this gives $3K=24$ condition channels.

## Conditional path

For a target video $x_1$ and Gaussian noise $x_0$, training samples a time $t \sim U[0,1]$ and uses the straight path

$$x_t=(1-t)x_0+t x_1.$$

I use a straight path because its velocity is available in closed form. This keeps training simple; numerical integration is only needed during sampling.

## Target velocity and loss

Differentiating the path gives

$$u_t=x_1-x_0.$$

The 3D U-Net predicts $v_\theta(x_t,t,c)$, and the conditional flow-matching objective is

$$\mathcal{L}_{\mathrm{CFM}}=
\mathbb{E}\left[\left\|v_\theta(x_t,t,c)-(x_1-x_0)\right\|_2^2\right].$$

The final run adds a foreground weight because text occupies only a small part of each frame. This changes the relative pixel weights, not the target velocity.

## Euler sampling

Sampling begins with a Gaussian-noise video and follows the learned ODE using

$$x_{k+1}=x_k+h\,v_\theta(x_k,t_k,c), \qquad h=1/N.$$

The reported character model uses $N=20$ Euler steps. Its aligned layout is supplied as a condition, but no extra guidance term is applied during sampling.

