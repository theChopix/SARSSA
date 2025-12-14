#!/bin/bash
mkdir -p data/movieLens
wget -O data/movieLens/MovieLens.zip https://files.grouplens.org/datasets/movielens/ml-32m.zip
unzip -o data/movieLens/MovieLens.zip -d data/movieLens
rm -f data/movieLens/MovieLens.zip
mv data/movieLens/ml-32m/ratings.csv data/movieLens/MovieLensRatings.csv
mv data/movieLens/ml-32m/tags.csv data/movieLens/MovieLensTags.csv
# rm -rf data/movieLens/ml-32m