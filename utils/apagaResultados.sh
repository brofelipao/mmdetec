# Remove arquivos com resultados da última execução

find .. -iname "MModels" | xargs rm -rf
find .. -iname "prediction_*" | xargs rm -rf
rm ../dataset/*.csv
rm ../dataset/*.png
rm ../dataset/*.txt
rm -rf ../dataset/fold_*


