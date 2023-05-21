# Quebra as linhas e deixa melhor organizado para
# visualização o arquivo de anotações gerado pelo Roboflow

python -m json.tool ../dataset/all/train/_annotations.coco.json > saida.json
mv saida.json ../dataset/all/train/_annotations.coco.json
