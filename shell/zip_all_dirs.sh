#!/bin/sh
for dir in *; do zip -r ${dir%.*}.zip $dir; done
