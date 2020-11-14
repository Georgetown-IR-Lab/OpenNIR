# This essentially gives you Python 3.6.12 on Ubuntu 18.04 according to their docs https://hub.docker.com/r/tensorflow/tensorflow/
FROM tensorflow/tensorflow:2.2.1-gpu-py3

# Create workspace folder and set it as working directory
WORKDIR /workspace

# Copy openNIR files into the container
COPY . .

# update apt-get
RUN apt-get update -y

# Install git
RUN apt-get install git -y

# Install java 11
RUN apt-get install openjdk-11-jdk -y

# Install python dependencies
RUN pip install -r requirements.txt

# Drop you into a bash shell
CMD ["/bin/bash"]
