#!/bin/bash
docker build -t covidaid:latest .
docker run --shm-size 8G --volume ${PWD}/:/covidaid/ covidaid:latest
