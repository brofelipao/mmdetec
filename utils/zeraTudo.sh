find .. -iname "*.png" | xargs rm 
find .. -iname "*.jpg" | xargs rm 
find .. -iname "*.pth" | xargs rm 
rm ../dataset/filesJSON/*
rm -rf ../dataset/all/*
rm -rf ../mmdetection


