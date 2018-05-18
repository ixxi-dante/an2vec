#!/bin/bash -e

echo "Unzipping compressed data sets..."
echo
unzip data/malariaDBLaNetworks2013.zip -d data/malariaDBLaNetworks2013 -x "__MACOSX/*"
echo
unzip data/BlogCatalog-dataset.zip -d data
echo
unzip data/Flickr-dataset.zip -d data
echo
unzip data/YouTube-dataset.zip -d data
echo
echo "All done!"
