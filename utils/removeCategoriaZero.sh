# Apagar as linhas 18..22 do arquivo de anotações do banco de imagens de
# ovos de Aedes

file=../dataset/all/train/_annotations.coco.json

sed -i -e '18,22d' $file



