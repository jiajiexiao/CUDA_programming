#!/bin/bash
make
rm *txt
for n in 5 10 50 100 500 1000 2000 2718 3000
do
./MatrixMultTiled $n
done
