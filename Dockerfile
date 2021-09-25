FROM nvidia/cuda:10.2-base
# Set locale
ENV LANG=C.UTF-8
# Avoid apt installations to require user interaction
ENV DEBIAN_FRONTEND=noninteractive

# Install pyenv
## Install dependencies for pyenv
RUN apt update && apt install --no-install-recommends -y \
    git curl unzip git make build-essential libssl-dev python3-pip python3-setuptools\
    libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev \
    libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python3-openssl zlib1g-dev
## Install pyenv from online script
RUN curl https://pyenv.run | bash
## Add bashrc lines to configure pyenv
RUN echo 'export PATH="~/.pyenv/bin:$PATH"' >> ~/.bashrc
RUN echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
## Configure the current shell
ENV PATH="/root/.pyenv/shims:/root/.pyenv/bin:${PATH}"
RUN eval $(pyenv init --path)

# Switch to the /app workdir
WORKDIR /app

# Install repository Python version and set-up virtual environment
COPY .python-version ./
RUN pyenv install

# Install poetry
RUN pip install poetry
RUN poetry config virtualenvs.in-project true
RUN pip install --upgrade pip
COPY poetry.lock  pyproject.toml ./
RUN poetry install

RUN mkdir -p ~/.cache/torch/hub/checkpoints/ && \
    wget --no-verbose --show-progress  --progress=bar:force:noscroll \
    https://download.pytorch.org/models/vgg19-dcbb9e9d.pth \
    -P ~/.cache/torch/hub/checkpoints/

# Copy the repository
COPY .git .git
COPY src src
COPY train.py ./
COPY download_data.py ./
COPY infer.py ./

RUN echo 'export VIRTUAL_ENV=/app/.venv' >> ~/.bashrc
RUN echo 'export PATH=$VIRTUAL_ENV/bin:$PATH' >> ~/.bashrc
