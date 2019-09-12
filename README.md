# A Practical Introduction to Automatic Audio Segmentation Using Deep Learning

_In Proceedings of: [**Summer School: Deep Learning for Language Analysis - September, 2019**](http://ml-school.uni-koeln.de/)_ 

> **Goal**
> 
> Train a deep neural network on [VGGish](https://github.com/tensorflow/models/tree/master/research/audioset) audio features 
> to automatically segment an audio file into speech and non-speech parts. 

## Setup
Please follow the instructions below in order.

1. Install and setup the following:
   1. [`docker`](https://docs.docker.com/install/)
   2. [`git`](https://git-scm.com/downloads)
2. Clone this repository.
    - Or, download it if you don't have git.
3. Start Docker on your machine.
    - Check in Docker settings if appropriate sharing of drive is setup
    - Check in Docker settings if appropriate CPU/RAM limits are setup
4. Build a docker image (Terminal on Linux/macOS, or **Powershell** on Windows):
    ```bash
    # change directory to the clone, if not already
    # replace ~ below with where you cloned this repository
    cd ~/UoC-ml-school-2019

    # Build a docker image
    docker build --tag uoc:2019 $PWD
    ```
5. Run a docker container with the built image (serving a [Jupyter Lab](https://jupyterlab.readthedocs.io/en/stable/) instance)
    ```bash
    # assuming you are in UoC-ml-school-2019 directory

    # run the created image, exposing Jupyter Lab port to 8888, 
    # and mounting the current directory inside the container
    docker run -it -p 8888:8888 -v $PWD/:/ml-school uoc:2019 jupyter lab
    ```
6. Open the served Jupyter Lab instance in the browser by following instructions in the terminal.
   - There will be a URL in the terminal that you can copy and then paste in the browser.
7. Check setup by opening and following the instructions `00-check-setup.ipynb`.