## Docker

### Quick Start

If you have docker installed, you can run the image with the following command:

```bash
docker run -it --rm scottpchow23/open-nir:general
```

**NOTE**: This image is likely to be somewhat out of date in terms of features/changes; follow instructions below to build the image yourself to guarantee the most up-to-date changes.

This will drop you into a shell in a directory with all dependencies already installed. From there, you should be able to any of the training commands as provided in the [documentation](../README.md).

### Build it yourself

If you'd prefer to build the image yourself to guarantee the contents of the image, you can do so by running the following command in the root directory of this project:

```bash
docker build -t your-docker-username/open-nir:tag
```

This is also useful if you want to tweak the docker file to change/update dependencies.
