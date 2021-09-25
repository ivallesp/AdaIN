# Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization
This repository contains an implementation of the model described in the [Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization](https://arxiv.org/abs/1703.06868) paper. The model is trained on the [COCO dataset](http://cocodataset.org/) as content images, and the [abstract art dataset](https://www.kaggle.com/bryanb/abstract-art-gallery). The trained model can be used to generate images from a given style image. The model can be trained on a single GPU.

![example](img/sample_model_abstractdataset.png)

## Getting Started
If you are interested in run the code, please, follow the next steps.

1. Install [pyenv](https://github.com/pyenv/pyenv) and [poetry](https://python-poetry.org/) in your system following the linked official guides.
2. Open a terminal, clone this repository and `cd` to the cloned folder.
3. Run `pyenv install $(cat .python-version)` in your terminal to install the required python version.
4. Configure poetry with `poetry config virtualenvs.in-project true`.
5. Create the virtual environment with the required dependencies with `poetry install`.
6. Activate the environment with `source .venv/bin/activate`.

You must download the COCO dataset and the abstract art dataset from the [COCO dataset](http://cocodataset.org/) and [abstract art dataset](https://www.kaggle.com/bryanb/abstract-art-gallery) and place them in the `data` folder. The pytorch dataset loader requires the images to be in a folder per class, you can fix this by creating a `dummy` folder and placing the images in it.

You can run the `train.py` script to train the model and generate images (using the training dataset). The generated images will be dumped in tensorboard. To visualize the generated images, run `tensorboard --logdir=tb_logs`.

### Docker
To ease the process of running the code, you can use the [Docker](https://www.docker.com/) to run the code in a container. You just need to install docker, `cd` into the folder of the project and run `docker build -t adain .`

Once the build process finishes, you can start running code in the container. The following command shows an example of how to run inference inside the container.

```
 docker run \
    -v $(realpath ./models):/app/models \
    -v $(realpath ./data):/app/data \
    -v $(realpath out):/app/out \
    -it adain \
        .venv/bin/python infer.py \
            --content-dir data/samples/content \
            --style-dir data/samples/style \
            --output-dir out \
            --model-dir models/abstract_art/ \
            --epoch 45 --method cartesian
```


## Contribution
Pull requests and issues will be tackled upon availability.