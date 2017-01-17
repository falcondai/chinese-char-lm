#! /bin/bash

awk '{print length, $0}' ./work/train.txt | sort -n | cut -d " " -f2- > ./work/new_train.txt