#!/bin/bash -x

rm ./lmdb_train/data.mdb
rm ./lmdb_train/lock.mdb

DEST_DIR=/data1/qspace/sagazhou/crnn/data/train
IMG_LIST=train_list.txt
LABEL_LIST=train_label.txt

listArray=(annotation_train_imgList.txt /data1/qspace/sagazhou/crnn/saoyisao/sample_0413/train_list.txt /data1/qspace/sagazhou/crnn/saoyisao/sample_0419/train_list.txt /data1/qspace/sagazhou/crnn/saoyisao/sample_0425/train_list.txt)

if [ -e $IMG_LIST ]; then
	rm $IMG_LIST
fi

# debug
#for i in ${listArray[@]}; do
#	echo "proc list $i"
#done

for i in ${listArray[@]}; do
	sed -i '/-/d; s/^M//g' $i
    cat $i >> $IMG_LIST 
done

if [ -e $IMG_LIST ]; then
	echo "create image list : $IMG_LIST"
fi

labelArray=(annotation_train_imgList_label.txt /data1/qspace/sagazhou/crnn/saoyisao/sample_0413/train_label.txt /data1/qspace/sagazhou/crnn/saoyisao/sample_0419/train_label.txt /data1/qspace/sagazhou/crnn/saoyisao/sample_0425/train_label.txt)

if [ -e $LABEL_LIST ]; then
	rm $LABEL_LIST
fi

# debug
#for i in ${labelArray[@]}; do
#	echo "proc label $i"
#done

for i in ${labelArray[@]}; do
	sed -i '/-/d; s/^M//g' $i
    cat $i >> $LABEL_LIST
done

if [ -e $LABEL_LIST ]; then
	echo "create label list : $LABEL_LIST"
fi

# python create_dataset_for_synth90k.py ./lmdb $IMG_LIST $LABEL_LIST $LEXICON
python create_dataset_for_synth90k.py ./lmdb_train $IMG_LIST $LABEL_LIST
echo "1 build dataset successfully."

cp ./lmdb_train/data.mdb $DEST_DIR
cp ./lmdb_train/lock.mdb $DEST_DIR
chmod 777 "${DEST_DIR}/data.mdb"
chmod 777 "${DEST_DIR}/lock.mdb"
echo "2 copy data.mdb lock.mdb to $DEST_DIR successfully."

exit 0

