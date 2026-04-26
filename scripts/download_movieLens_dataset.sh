#!/bin/bash
mkdir -p data/movieLens/raw
wget -O data/movieLens/MovieLens.zip https://files.grouplens.org/datasets/movielens/ml-32m.zip
unzip -o data/movieLens/MovieLens.zip -d data/movieLens
rm -f data/movieLens/MovieLens.zip
mv data/movieLens/ml-32m/ratings.csv data/movieLens/ratings.csv
mv data/movieLens/ml-32m/tags.csv data/movieLens/tags.csv
mv data/movieLens/ml-32m/* data/movieLens/raw/
rmdir data/movieLens/ml-32m
