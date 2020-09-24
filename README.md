<h1 align="center">FootVid F.C.</h1>
<h2 align="center">Football Video Frames Classification</h2>

<p align="center">
    <a href="https://www.python.org"><img src="https://img.shields.io/badge/python-3.7%20%7C%203.8-blue" alt="Code style: black"></a>
    <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black"></a>
</p>

The establishment of **FootVid F.C.** was guided by the simple idea:

<!-- prettier-ignore -->
> Revolutionize the classification of match video frames using AI

In this repository you will find implementation of ML-based methods targeting
the aforementioned task. Sample data available for the project are visualized
below using the source package function:
[`footvid.utils.images.plot_directories_representation`](https://github.com/mrtovsky/footvid/blob/master/footvid/utils/images.py#L32).

<p align="center">
    <img src="docs/images/example-video-frames.png" alt="example-video-frames" class="center" width="750">
</p>

## Project Organisation

    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── data
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs
    │   └── images         <- Folder for storing images used across the package documentation.
    │
    ├── logs               <- Tensorboard model training logs.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries.
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering) and
    │                         a short `-` delimited description, e.g. `00-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    |
    ├── poetry.lock        <- File to resolve and install all dependencies listed in the
    │                         pyproject.toml file.
    ├── pyproject.toml     <- File orchestrating the project and its dependencies.
    │
    ├── footvid            <- Source code for use in this project.

### Notebooks

The project is designed to separate the particular modeling steps into
notebooks. Notebook list:

- [00-images-integrity](https://github.com/mrtovsky/footvid/blob/master/notebooks/00-images-integrity.ipynb)
  focuses on getting familiarity with data, examines images size, resolution
  and relation of positive to negative frames. It also generates a presentation
  of an example input.
- [01-train-valid-split](https://github.com/mrtovsky/footvid/blob/master/notebooks/01-train-valid-split.ipynb)
  is dedicated to dividing the data set into an appropriately represented
  training and validation set to avoid consequences of _sampling bias_ like
  shown in the widely known _The Literary Digest_
  [Presidential poll](https://en.wikipedia.org/wiki/The_Literary_Digest#Presidential_poll).
  Implementation of an algorithm that works in a quadratic worst-case time
  complexity and finds best splits in a semi-greedy manner guided by
  `match_hash` is also provided in this notebook.
- [10-resnet](https://github.com/mrtovsky/footvid/blob/master/notebooks/10-resnet.ipynb)
  provides [ResNet50](https://arxiv.org/abs/1512.03385) experiments setup.
  Three methodologies of fine-tuning were tested. First, optimizing only the
  weights of the last, fully-connected, layer. Second, additionally unfreezing
  3rd and 4th ResNet layers. The last one, fine-tuning the whole neural net.
  Those operations are possible due to the recursive nature of the
  [`footvid.arena.freeze_layers`](https://github.com/mrtovsky/footvid/blob/master/footvid/arena.py#L59)
  function. The models were trained with the use of a safe SGD optimizer,
  avoiding the risk of unstable solutions obtained with
  [adaptive optimizers](https://arxiv.org/pdf/1705.08292.pdf).

## Installation

If only the **footvid** source package functionalities are of interest then it
is enough to run:

```bash
pip install git+https://github.com/mrtovsky/footvid.git
```

To interact with the notebooks e.g. rerun them, full project preparation is
necessary. It can be done in the following few steps. First of all, you need to
clone the repository:

```bash
git clone https://github.com/mrtovsky/footvid.git
```

Then, enter this directory and create a **.env** file that stores environment
variable with the cloned repository path:

```bash
cd footvid/
touch .env
printf "REPOSITORY_PATH=\"$(pwd)\"" >> .env
```

### Poetry

The recommended way of installing the full project is via
[Poetry](https://python-poetry.org/docs/#:~:text=Linux%20and%20OSX.-,Installation,recommended%20way%20of%20installing%20poetry%20.)
package. If Poetry is not installed already, follow the installation
instructions at the provided link. Then, assuming you have alreadt entered the
**footvid** directory, resolve and install dependencies using:

```bash
poetry install
```

Furthermore, you may want to attach a kernel with the already created virtual
environment to Jupyter Notebook. This can be done by calling:

```bash
poetry run python -m ipykernel install --name=footvid-venv
```

This will make **footvid-venv** available in your Jupyter Notebook kernels.

### pip

It is also possible to install the package in a traditional way, simply run:

```bash
pip install -e .
```

This will install the package in an editable mode. If you installed it inside
of the virtual environment, then attaching it to the Jupyter Notebook kernel is
the same as with the **Poetry** but the command is stripped from first two
elements (remember that the virtualenv needs to be activated beforehand):

```bash
python -m ipykernel install --name=footvid-venv
```

## Data

All files that were too big to be attached directly to the repository are
stored under the following
[Google Drive storage](https://drive.google.com/drive/folders/1EXcVoV8UPWVx2-2UQfnSZOFrw6Lk360K?usp=sharing).
Names and topology of files is matching the repository convention so whenever
given data is necessary (notebooks are informing about the required file
structure) just download and insert the files to the corresponding place in
your local repository.

## Results

| Method                              | Accuracy score (train / validation) | Average precision (train / validation) |
| ----------------------------------- | ----------------------------------: | -------------------------------------: |
| Fully-connected layer fine-tuning   |                     0.9676 / 0.9780 |                        0.9947 / 0.9949 |
| Top2 CNN layers and FCL fine-tuning |                     0.9826 / 0.9824 |                        0.9961 / 0.9970 |
| Full network fine-tuning            |                     0.9758 / 0.9802 |                        0.9956 / 0.9950 |

More detailed training results can be displayed by opening the **tensorboard**:

```bash
tensorboard --logdir ./logs/ --host localhost
```

## Related Publications

<!-- prettier-ignore -->
Karpathy et al., [Large-scale Video Classification with Convolutional Neural Networks](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42455.pdf)

**Key takeaways:**

- CNNs' expected behavior is to learn increasingly complex patterns with
  successive, deeper layers - first layers will learn more generic features, on
  the contrary, deeper layers will be responsible for learning intricate,
  dataset-specific features. In the case of transfer learning, this will have a
  significant impact on the choice of "defrosted" layers to be fine-tuned.

- Most sport videos are biased towards displaying the objects of interest in
  the central region. Thus, images preprocessing proposed in this article
  includes:

  - cropping to center region,
  - resizing images to 200 × 200 pixels,
  - randomly sampling a 170 × 170 region,
  - randomly flipping the images horizontally with 50% probability.

  The final absolute size of the images is too small for our problem because
  the pre-trained
  [torchvision.models](https://pytorch.org/docs/stable/torchvision/models.html)
  require the image height and width of at least 224 x 224 pixels but the idea
  is definitely worth attention, after appropriate adjustment to the problem
  under consideration.

<!-- prettier-ignore -->
Simeon Jackman, [Football Shot Detection using Convolutional Neural Networks](https://www.diva-portal.org/smash/get/diva2:1323791/FULLTEXT01.pdf)

**Key takeaways:**

- Confirmation of correctness of the appliaction of transfer learning in the
  context of football match TV broadcast analysis, with the use of pre-trained
  models trained on the ImageNet dataset.
