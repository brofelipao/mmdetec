# Roda o experimento para diferentes datasets:
#
# Copie os datasets compactados (.zip) para a pasta ./zips 
# O nome do arquivo .zip deve ser igual ao da classe do
# problema (pois o script altera o arquivo experimento.py
# automaticamente usando o nome do arquivo para a variável
# CLASSES (ver abaixo comando sed)

rm -rf resultados
mkdir -p resultados
cd utils
./apagaResultados.sh
for file in ../zips/*.zip; do
    nomeArquivo=$(basename $file)
    nomeArquivoSemExtensao="${nomeArquivo%.*}"
    echo 'Rodando para ' $nomeArquivoSemExtensao
    rm -rf ../dataset/all/*
    unzip $file -d ../dataset/all/   
    ./removeCategoriaZero.sh
    python geraDobras.py -folds=5  
    cd ..
    sed -i "s/CLASSES=(.*)/CLASSES=('$nomeArquivoSemExtensao',)/" experimento.py
    ./roda.sh
    mkdir -p resultados/$nomeArquivoSemExtensao
    cp -R dataset ./resultados/$nomeArquivoSemExtensao
    cp nohup.out ./resultados/$nomeArquivoSemExtensao/
    cp experimento.py ./resultados/$nomeArquivoSemExtensao/
    cd utils
done


