#!/bin/bash
for i in 0 1
do
for j in 0 2
do
make
make run
make clean
mv Train.cpp Train.txt;
let k=${j};
let ka=(${j}+2);
sed -i s/GAMA=$k/GAMA=$ka/ Train.txt;
mv Train.txt Train.cpp;
done

mv Train.cpp Train.txt;
sed -i s/GAMA=4/GAMA=0/ Train.txt;
mv Train.txt Train.cpp;

mv Cell.cpp Cell.txt;
let l=${i};
let la=(${i}+1)
sed -i s/NL_LTP=$l/NL_LTP=$la/ Cell.txt;
mv Cell.txt Cell.cpp;
done

mv Cell.cpp Cell.txt;
sed -i s/NL_LTP=2/NL_LTP=0/ Cell.txt;
mv Cell.txt Cell.cpp;