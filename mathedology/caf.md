Methodology: CAF — Competitive Activation Function

1. Introduction

Activation functions play a critical role in deep neural networks by introducing non-linearity into the learning process. Popular activation functions such as ReLU, Leaky ReLU, ELU, and Swish have demonstrated strong performance across various computer vision tasks. However, many existing activation functions either suppress negative inputs completely or provide limited control over feature amplification.

To enhance feature representation and improve gradient flow, a custom activation function named Adaptive Quadratic Scaling Activation Function (AQSAF) was developed.

The proposed activation function introduces an adaptive scaling mechanism that increases the response of neurons for larger activations while maintaining smooth differentiability.

The activation function is defined as:

[
f(x)=x\left(1+\alpha\frac{x^2}{1+x^2}\right)
]

where:

- x is the input activation.
- \alpha is a trainable or predefined scaling parameter controlling the degree of amplification.

---

2. Motivation

The Rectified Linear Unit (ReLU) is defined as:

[
f(x)=\max(0,x)
]

Although ReLU is computationally efficient, it suffers from several limitations:

- Zero output for all negative inputs.
- Dead neuron problem.
- Limited adaptability.
- Inability to amplify informative features.

Similarly, linear activations do not provide sufficient non-linearity for learning complex patterns.

To overcome these limitations, an adaptive scaling mechanism was introduced that strengthens important activations while preserving smooth gradients.

---

3. Mathematical Formulation

The proposed activation function is given by:

[
f(x)=x\left(1+\alpha\frac{x^2}{1+x^2}\right)
]

The function consists of two components:

Linear Component

[
x
]

which preserves the original information.

Adaptive Scaling Component

[
1+\alpha\frac{x^2}{1+x^2}
]

which dynamically modifies the activation magnitude based on the input value.

The scaling factor increases as the magnitude of the input grows.

---

4. Behavior Analysis

Case 1: Input Near Zero

When

[
x \rightarrow 0
]

the fraction

[
\frac{x^2}{1+x^2}
]

approaches zero.

Therefore,

[
f(x)\approx x
]

The activation behaves almost linearly near the origin, ensuring stable gradients during training.

---

Case 2: Large Positive Inputs

When

[
x \rightarrow +\infty
]

the fraction approaches one:

[
\frac{x^2}{1+x^2}\rightarrow1
]

Thus,

[
f(x)\approx x(1+\alpha)
]

Large positive activations are amplified by a factor controlled by \alpha.

---

Case 3: Large Negative Inputs

When

[
x \rightarrow -\infty
]

the fraction also approaches one.

Therefore,

[
f(x)\approx x(1+\alpha)
]

Negative activations are preserved rather than being completely suppressed, helping avoid the dead neuron problem.

---

5. Adaptive Feature Amplification

The term

[
\frac{x^2}{1+x^2}
]

acts as a bounded adaptive weighting function.

Its value always remains between 0 and 1:

[
0 \leq \frac{x^2}{1+x^2} < 1
]

This property ensures that the scaling factor remains stable and prevents uncontrolled growth of activations.

As input magnitude increases:

- Small activations receive little amplification.
- Strong activations receive larger amplification.
- The amplification gradually saturates.

This allows the network to emphasize informative features while maintaining numerical stability.

---

6. Derivative Analysis

Differentiability is important for gradient-based optimization.

The derivative of the proposed activation function is continuous over the entire input domain.

Because no discontinuities exist:

- Backpropagation remains stable.
- Gradient flow is improved.
- Optimization becomes smoother.

Unlike ReLU, the derivative never experiences an abrupt discontinuity at zero.

---

7. Algorithm

Step 1

Receive input activation value from the previous layer.

Step 2

Compute the adaptive scaling factor:

[
ScalingFactor=
1+\alpha\frac{x^2}{1+x^2}
]

Step 3

Multiply the input by the scaling factor:

[
Output=x\times ScalingFactor
]

Step 4

Pass the transformed output to the next layer.

Step 5

During backpropagation, compute gradients using the differentiable activation function.

---

8. Advantages of the Proposed Activation Function

Smooth Differentiability

The function is continuous and differentiable across the entire input range.

Improved Gradient Flow

Reduces the likelihood of vanishing gradients and dead neurons.

Adaptive Feature Enhancement

Amplifies important activations automatically.

Preservation of Negative Information

Unlike ReLU, negative activations are not discarded.

Stable Training

The bounded scaling mechanism prevents excessive activation growth.

Better Representation Learning

Allows the network to learn richer and more discriminative features.

---

9. Parameters Used

Parameter| Description
x| Input activation value
α| Adaptive scaling coefficient
Backbone Network| Modified VGG16
Dataset| Fruit Recognition Dataset

---

10. Conclusion

An Adaptive Quadratic Scaling Activation Function (AQSAF) was proposed to improve feature representation and learning capability in deep neural networks. The activation function combines a linear component with an adaptive bounded scaling mechanism that selectively amplifies significant activations while preserving smooth gradient flow. By maintaining differentiability across the entire input domain and avoiding the dead neuron problem, the proposed activation function contributes to more robust learning and improved classification performance in fruit recognition tasks.