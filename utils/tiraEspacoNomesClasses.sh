# Troca no arquivo de anotações as palavras Ovos e Ovo por Corn (o código está usando Corn)

file=../dataset/all/train/_annotations.coco.json
sed -i 's/cheek and forehead/cheek_forehead/g' $file 
sed -i 's/cheek and nose/cheek_nose/g' $file 
sed -i 's/forehead and nose/forehead_nose/g' $file 
sed -i 's/side face/side_face/g' $file 



