#!/bin/bash
mkdir -p data/lastFm1k
wget -O data/lastFm1k/LastFm1k.tar.gz http://mtg.upf.edu/static/datasets/last.fm/lastfm-dataset-1K.tar.gz
tar -xzf data/lastFm1k/LastFm1k.tar.gz -C data/lastFm1k --strip-components=1 \
    lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv
rm -f data/lastFm1k/LastFm1k.tar.gz
mv data/lastFm1k/userid-timestamp-artid-artname-traid-traname.tsv data/lastFm1k/ratings.tsv
