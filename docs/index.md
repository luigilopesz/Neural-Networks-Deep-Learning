<div class="ann-hero" markdown>

# Artificial Neural Networks & Deep Learning

Coursework for Artificial Neural Networks and Deep Learning at Insper, 2026.2, by Luigi Lopes ([@luigilopesz](https://github.com/luigilopesz){:target="_blank"}). Each exercise is a report: it states the task, shows the code behind every number, and says what the results do and do not show.

</div>

<div class="ex-grid">
<article class="ex-card">
<svg class="ex-glyph" viewBox="0 0 48 48" aria-hidden="true" focusable="false"><g class="hi"><circle cx="9" cy="13" r="2.7"/><circle cx="17" cy="8" r="2.7"/><circle cx="15" cy="19" r="2.7"/><circle cx="24" cy="15" r="2.7"/><circle cx="10" cy="26" r="2.7"/><circle cx="23" cy="26" r="2.7"/></g><circle cx="30" cy="22" r="2.7"/><circle cx="38" cy="17" r="2.7"/><circle cx="36" cy="30" r="2.7"/><circle cx="28" cy="36" r="2.7"/><circle cx="41" cy="37" r="2.7"/><circle cx="19" cy="38" r="2.7"/></svg>
<h2 class="ex-title"><a href="./exercises/data/">Data</a></h2>
<p class="ex-desc">How the spread of a point cloud decides whether a straight line can separate its classes, in 2D, in 5D and on a real table.</p>
<p class="ex-due">Due <time datetime="2026-09-10T23:59">10 Sep 2026, 23:59</time></p>
</article>
<article class="ex-card">
<svg class="ex-glyph" viewBox="0 0 48 48" aria-hidden="true" focusable="false"><line x1="6" y1="40" x2="42" y2="8"/><g class="hi"><circle cx="10" cy="12" r="2.7"/><circle cx="20" cy="10" r="2.7"/><circle cx="14" cy="24" r="2.7"/><circle cx="24" cy="18" r="2.7"/><circle cx="8" cy="28" r="2.7"/></g><circle cx="36" cy="22" r="2.7"/><circle cx="30" cy="36" r="2.7"/><circle cx="40" cy="38" r="2.7"/><circle cx="24" cy="34" r="2.7"/><circle cx="42" cy="28" r="2.7"/></svg>
<h2 class="ex-title"><a href="./exercises/perceptron/">Perceptron</a></h2>
<p class="ex-desc">One perceptron, two datasets. It converges when the classes are separable and never settles when they overlap, which motivates the pocket algorithm.</p>
<p class="ex-due">Due <time datetime="2026-10-18T23:59">18 Oct 2026, 23:59</time></p>
</article>
<article class="ex-card">
<svg class="ex-glyph" viewBox="0 0 48 48" aria-hidden="true" focusable="false"><line x1="8" y1="17" x2="24" y2="10"/><line x1="8" y1="17" x2="24" y2="24"/><line x1="8" y1="17" x2="24" y2="38"/><line x1="8" y1="31" x2="24" y2="10"/><line x1="8" y1="31" x2="24" y2="24"/><line x1="8" y1="31" x2="24" y2="38"/><line x1="24" y1="10" x2="40" y2="24"/><line x1="24" y1="24" x2="40" y2="24"/><line x1="24" y1="38" x2="40" y2="24"/><circle class="node" cx="8" cy="17" r="3.4"/><circle class="node" cx="8" cy="31" r="3.4"/><circle class="hi" cx="24" cy="10" r="3.4"/><circle class="hi" cx="24" cy="24" r="3.4"/><circle class="hi" cx="24" cy="38" r="3.4"/><circle class="node" cx="40" cy="24" r="3.4"/></svg>
<h2 class="ex-title"><a href="./exercises/mlp/">MLP</a></h2>
<p class="ex-desc">What a hidden layer buys you: a backward pass done by hand, a dataset no straight line can split, three classes, and a test of whether a second hidden layer pays for itself.</p>
<p class="ex-due">Due <time datetime="2026-10-25T23:59">25 Oct 2026, 23:59</time></p>
</article>
<article class="ex-card">
<svg class="ex-glyph" viewBox="0 0 48 48" aria-hidden="true" focusable="false"><path d="M4 11 L19 19 L19 29 L4 37 Z"/><path d="M44 11 L29 19 L29 29 L44 37 Z"/><line x1="19" y1="24" x2="29" y2="24"/><circle class="hi" cx="24" cy="24" r="3.6"/></svg>
<h2 class="ex-title"><a href="./exercises/vae/">VAE</a></h2>
<p class="ex-desc">A variational autoencoder on MNIST. Reconstruction and KL divergence are tracked separately, then a 2-D latent space shows what the prior buys and what it costs.</p>
<p class="ex-due">Due <time datetime="2026-11-06T23:59">6 Nov 2026, 23:59</time></p>
</article>
</div>

The semester project asks whether a river station will report a flood tomorrow. Its dataset is synthetic, so nothing in it is a claim about real flood risk. Start with the [project brief](projects/index.md) or go straight to the [exploratory analysis](projects/eda/index.md).

## What a neural network is

A neural network is a parametric function built from one repeated piece: an affine map (multiply by a weight matrix, add a bias) followed by a fixed nonlinearity. Each layer re-represents what the layer before it produced, and with enough layers and width the stack can approximate a very broad class of continuous input-output mappings, the *universal approximation* property.

"Learning" means searching for the weights that make the output match known examples. The search is gradient descent, and the gradients come from **backpropagation**: the chain rule applied layer by layer, from the output back to the input. Whether a network *can* represent a function is rarely the binding constraint. Whether gradient descent can actually *reach* that function from a random start, on finite data, in finite time, usually is. Initialization schemes, normalization layers, optimizers and architectures with built-in structure exist largely to close that gap.

!!! note "Learned, not designed"

    A classical pipeline hand-crafts features before a simple classifier sees the data. A neural network folds feature construction into the same optimization that fits the decision, so its intermediate representations are **learned** and adapt to whatever the data contains.

## Why depth mattered

"Deep" means many layers, and what depth buys is **hierarchy**. In a vision network, early layers tend to respond to edges and textures, middle layers to parts, later layers to whole objects, and nobody specifies that ladder by hand. Language, audio and tabular problems show the same pattern.

The idea is old. Three trends made it practical:

- **Data** large enough to constrain millions of parameters without memorizing them.
- **Compute** cheap and parallel enough to fit them: GPUs, and the matrix-multiply-heavy shape of neural nets.
- **Algorithmic fixes**: better initialization, normalization layers, sturdier optimizers, and architectures such as convolutions and attention that encode a useful prior about the data's structure.

The result is the default tool when a mapping is too intricate to write as explicit rules but there are plenty of examples: recognizing images and speech, generating text and media, ranking and recommending, and increasingly deciding in systems that used to run on hand-tuned heuristics.

## A short, selective timeline

| Year | Milestone |
|---|---|
| 1958 | Frank Rosenblatt builds the **Perceptron**, a single-layer linear classifier. It is the field's founding architecture and, soon after, the source of its first well-known limitation: a lone perceptron cannot represent XOR. |
| 1986 | Rumelhart, Hinton and Williams popularize **backpropagation** for multi-layer networks, turning "stack more layers" from a theoretical idea into something that can be trained end to end. |
| 1998 | Yann LeCun's **LeNet** applies convolutional networks to handwritten digit recognition at production scale, setting weight sharing and pooling as the template for image architectures for a decade. |
| 2012 | **AlexNet**, a deep convolutional network trained on GPUs, wins the ImageNet competition ahead of the hand-engineered vision pipelines that preceded it, commonly cited as the point where convolutional networks became the standard in computer vision. |
| 2013 | Kingma and Welling introduce the **variational autoencoder**, a generative model trained by backpropagating through a sampling step with the reparameterization trick. |
| 2015 | **Batch normalization** and **residual connections** (ResNets) arrive in close succession and make networks hundreds of layers deep reliably trainable. |
| 2017 | *Attention Is All You Need* introduces the **Transformer**, replacing recurrence with self-attention. It becomes the backbone of nearly every large language and multimodal model that follows. |
| 2020s | **Scaling** (more parameters, more data, more compute, mostly on transformer backbones) becomes a research strategy of its own and yields general-purpose language and multimodal models used as foundations for downstream systems. |

## How the reports are organized

Each exercise and project phase follows the same shape: the task, the data or steps involved, the executed code, and a discussion of what the results show.

!!! tip "Reading order"

    - **Exercises** are reports with the code, figures and numbers behind every claim. Data comes first, because what a network can learn is bounded by how its data was generated or prepared: the geometry of the classes, whether a linear model can separate them at all, and how a preprocessing pipeline can leak information from a held-out split. Perceptron and MLP then build the models from scratch, and VAE turns to generative models.
    - **Projects** hold the semester-long project, applied to one dataset phase by phase. The dataset is synthetic, not a record of real floods; each phase says so and draws no real-world conclusions from it.

Source for this repository is on [GitHub](https://github.com/luigilopesz/Neural-Networks-Deep-Learning){:target="_blank"}.
