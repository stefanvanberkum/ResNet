#!/bin/bash

for dataset in cifar10 cifar100
do
	for n in 1 3 5 7 9
	do
		for p_L in 1 0.95 0.9 0.85 0.8
		do
			python ResNet.py $dataset $n $p_L
			wait
		done
	done
done