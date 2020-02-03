# mnist_classifiers docker guide

To build docker image:

docker build . -t mnist_classifer:version1

To run the docker with the API:
docker run --rm -v $PWD:/tmp -ti -p 5000:5000 mnist_classifier:version1 python3 api.py



