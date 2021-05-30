#!/bin/bash
PARALLELL=${1:-32}
make -j ${PARALLELL} DEBUG=-ggdb DEVICE_DEBUG=-G
