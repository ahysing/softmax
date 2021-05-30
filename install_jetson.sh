#!/bin/bash

apt-get update --fix-missing
apt-get -yq install libcurl4-openssl-dev \
	                clang \
	                libatlas-dev \
					libatlas-base-dev \
					liblapack-dev \
					libblas-dev

/home/ubuntu/cuda-l4t/cuda-l4t.sh cuda-repo-l4t-r21.5-6-5-local_6.5-53_armhf.deb 6.5 6-5

sudo -u "$SUDO_USER" make -j 4