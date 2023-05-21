cd ../
mv *.zip ./dataset/all
cd ./dataset/all/  
rm -rf train
unzip *.zip
cd ../../utils
./apagaResultados.sh  
python geraDobras.py -folds=5 -valperc=0.3  
