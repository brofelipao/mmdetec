# extraiRetangulos.py
# Autor: Hemerson Pistori 
# Descrição: A partir de anotações feitas no formato json COCO gera um novo
#    conjunto de treinamento para ser usado com o compara_classificadores_tf2
#    (transforma em um problema de classificação apenas)


from pycocotools.coco import COCO
import argparse
from PIL import Image
import os

parser = argparse.ArgumentParser(description='Extrai os retângulos anotados salvando como imagens independentes')

parser.add_argument('-annotations', default='../dataset/all/train/_annotations.coco.json',  type=str,  help='Caminho para o arquivo com as anotações',required=False)
parser.add_argument('-images', default='../dataset/all/train', type=str,                    help='Caminho para o arquivo com as imagens',required=False)
parser.add_argument('-classes', default='../dataset/classes',type=str, help='Pasta para os arquivos resultantes, separados por classe',required=False)

args = parser.parse_args()

anotacoes = COCO(args.annotations)

categorias_ID = anotacoes.getCatIds()
categorias = anotacoes.loadCats(categorias_ID)


print('Categorias: ',categorias)

i = 1

for categoria in categorias:

   nome_categoria = categoria['name']
   print('Processando categoria: ',nome_categoria)  
   
   # Pega todas as anotações de um categoria
   anotacoes_categoria_IDs = anotacoes.getAnnIds(catIds=[categoria['id']])  

   anotacoes_categoria = anotacoes.loadAnns(ids=anotacoes_categoria_IDs)

   for anotacao in anotacoes_categoria:
   
       x = anotacao['bbox'][0]
       y = anotacao['bbox'][1]
       w = anotacao['bbox'][2]
       h = anotacao['bbox'][3]

       img_id = anotacao['image_id']
       img_coco = anotacoes.loadImgs(ids=img_id)
       
       caminho_imagem=args.images+'/'+img_coco[0]['file_name']
       
       imagem = Image.open(caminho_imagem) 
       retangulo=imagem.crop((x, y, x+w, y+h))


       nome_pasta=args.classes+'/'+nome_categoria+'/'
       
       if not os.path.isdir(nome_pasta):
          os.makedirs(nome_pasta)
    
       nome_arquivo=str(i).zfill(5)+'.jpg'
       print('Salvando ',nome_pasta+nome_arquivo)
       
       retangulo.save(nome_pasta+nome_arquivo)
       
       i=i+1

