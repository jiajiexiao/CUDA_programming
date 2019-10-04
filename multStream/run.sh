#!/bin/bash

rm *log
make
#./pihist pi-10million.txt 
./pihist pi-10million.txt >> out.log  
for n in 1000000 100000 1000 100 10
do
./pihist pi-10million.txt $n >> out.log 
done
