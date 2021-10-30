#!/bin/bash
docker build -t covidaid:latest .
docker run --volume ${PWD}/:/covidaid/ covidaid:latest
