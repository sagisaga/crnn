#!/bin/bash -x

# usage
# ./collect.sh log.txt

LOG="/data1/qspace/sagazhou/crnn/model/crnn_demo/log.txt"
FILE="img_list.txt"

sed -n '/path/p' $LOG | awk '{print $5}' > $FILE

DIR="negative_sample"
if [ -d $DIR ]; then
	rm -rf $DIR
fi

if [ -f "$DIR.tar.gz" ]; then
	rm "$DIR.tar.gz"
fi

mkdir $DIR

while read LINE; do
	cp $LINE $DIR
done < $FILE

tar -zcvf "$DIR.tar.gz" ./$DIR

exit 0
