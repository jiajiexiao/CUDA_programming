#!/bin/bash
rm *txt
make
for n in 1000000000 100000000 10000000 1000000 100000 10000 1000 100 10
do
./pimc $n
sleep 5s
done
