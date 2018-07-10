#!/bin/bash -e

echo "Unzipping compressed data sets..."
echo
unzip datasets/malariaDBLaNetworks2013.zip -d datasets/malariaDBLaNetworks2013 -x "__MACOSX/*"
echo
unzip datasets/BlogCatalog-dataset.zip -d datasets
echo
unzip datasets/Flickr-dataset.zip -d datasets
echo
unzip datasets/YouTube-dataset.zip -d datasets
echo
echo "All done!"
