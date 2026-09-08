This site is the coursework repository for **Artificial Neural Networks and Deep Learning**, maintained by Luigi Lopes ([@luigilopesz](https://github.com/luigilopesz){:target="_blank"}). It works as a lab notebook: a short primer on the field below, followed by the roteiros and exercises worked through as the course progresses.

## What a Neural Network Actually Is

Strip away the biological metaphor and a neural network is just a parametric function built from simple, repeated pieces: an affine map (multiply by a weight matrix, add a bias) followed by a fixed nonlinearity, stacked layer after layer. One layer alone is close to linear regression; stacking many of them is what makes the difference. Each layer transforms the representation it receives from the one before it, and with enough layers and enough width the resulting function can approximate a very broad class of continuous input-output mappings — the *universal approximation* property.

"Learning" means searching for the weight values that make the network's output match a set of known examples. That search is gradient descent, and the gradients come from **backpropagation** — an automated, layer-by-layer application of the calculus chain rule, computed from the output back to the input. Whether a large-enough network *can* represent a given function is usually not the binding constraint; whether that function is actually *reachable* by gradient descent, from a random starting point, on a finite dataset, in finite time, is. Most of the field's day-to-day methodology — initialization schemes, normalization layers, optimizer design, architectures with built-in structural priors — exists to close that gap between representational capacity and what gradient descent actually finds.

!!! note "The other defining trait"

    In a classical pipeline, a human hand-crafts features before a simple classifier ever sees the data. A neural network folds feature construction into the same optimization that fits the final decision, so the intermediate representations are **learned**, not designed — they adapt to whatever the data actually contains.

## Why Depth Changed Everything

"Deep" simply means many layers, but depth buys something specific: **hierarchy**. Each layer builds on the abstractions the previous one produced, so a vision network's early layers tend to respond to edges and simple textures, middle layers to parts and motifs, and later layers to whole objects — without anyone specifying that hierarchy by hand. The same pattern shows up in language, audio, and tabular problems alike: depth lets a model build its own ladder of increasingly abstract features on the way from raw input to prediction.

That idea existed long before it was practical. What actually made deep networks trainable and worth using at scale was three trends converging at once:

- **Data** large enough to constrain millions of parameters without simply memorizing it.
- **Compute** cheap and parallel enough — GPUs, and the matrix-multiply-heavy nature of neural nets — to fit those parameters in reasonable time.
- **Algorithmic fixes**: better weight initialization, normalization layers, more robust optimizers, and architectures such as convolutions and attention that bake in a useful prior about the data's structure instead of leaving the network to rediscover it from scratch.

The result is a tool that is now the default choice whenever a mapping from input to output is too intricate to specify with explicit rules, but there is enough example data to let a model infer it: recognizing images and speech, generating text and media, ranking and recommending, and increasingly, driving decisions in systems that used to run on hand-tuned heuristics.

## A Short, Selective Timeline

| Year | Milestone |
|---|---|
| 1958 | Frank Rosenblatt builds the **Perceptron**, a single-layer linear classifier — the field's founding architecture, and soon after, its first well-known limitation: a lone perceptron cannot represent a function as simple as XOR. |
| 1986 | Rumelhart, Hinton, and Williams popularize **backpropagation** for multi-layer networks, turning "stack more layers" from a theoretical idea into something that could actually be trained end-to-end. |
| 1998 | Yann LeCun's **LeNet** applies convolutional networks to handwritten digit recognition at production scale, establishing weight sharing and pooling as the template for image architectures for the next decade. |
| 2012 | **AlexNet**, a deep convolutional network trained on GPUs, wins the ImageNet competition, outperforming the hand-engineered computer-vision pipelines that preceded it — commonly cited as the point where convolutional networks became the standard approach in computer vision. |
| 2015 | **Batch normalization** and **residual connections** (ResNets) arrive in close succession, making networks hundreds of layers deep reliably trainable for the first time. |
| 2017 | *Attention Is All You Need* introduces the **Transformer**, replacing recurrence with self-attention. It becomes the backbone architecture behind nearly every large-scale language and multimodal model that follows. |
| 2020s | **Scaling** — more parameters, more data, more compute, largely on transformer backbones — becomes a research strategy in its own right, producing general-purpose large language and multimodal models now used as foundations for downstream systems. |

## About This Repository

Each roteiro and exercise in the navigation follows the same shape: a stated goal, the steps or dataset involved, the executed code, and a discussion of what the results show.

!!! tip "Where to go next"

    - **Roteiros** — the guided lab worksheets for the course, one per assignment.
    - **Exercises** — the coursework notebooks, where the datasets, models, and results actually live. Exercise Set 1 (Data) comes first, because a network's output depends on how the data it sees was generated or prepared: its geometry, whether a linear model can separate the classes at all, and how a preprocessing pipeline can leak information from a held-out split, all constrain what any model trained on it can subsequently learn.

Source for this repository is on [GitHub](https://github.com/luigilopesz/Artificial-Neural-Networks-and-Deep-Learning){:target="_blank"}.
