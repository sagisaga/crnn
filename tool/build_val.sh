#!/bin/bash -x

if [ -e ./lmdb_val/data.mdb ]; then
	rm ./lmdb_val/data.mdb
	rm ./lmdb_val/lock.mdb
fi

DEST_DIR=/data1/qspace/sagazhou/crnn/data/val
IMG_LIST=val_list.txt
LABEL_LIST=val_label.txt

listArray=(annotation_val_imgList.txt /data1/qspace/sagazhou/crnn/saoyisao/sample_0413/val_list.txt /data1/qspace/sagazhou/crnn/saoyisao/sample_0419/val_list.txt /data1/qspace/sagazhou/crnn/saoyisao/sample_0425/val_list.txt) 

if [ -e $IMG_LIST ]; then
	rm $IMG_LIST
fi

for i in ${listArray[@]}; do
     sed -i '/-/d; s/^M//g' $i
     cat $i >> $IMG_LIST
done


labelArray=(annotation_val_imgList_label.txt /data1/qspace/sagazhou/crnn/saoyisao/sample_0413/val_label.txt /data1/qspace/sagazhou/crnn/saoyisao/sample_0419/val_label.txt /data1/qspace/sagazhou/crnn/saoyisao/sample_0425/val_label.txt)

if [ -e $LABEL_LIST ]; then
	rm $LABEL_LIST
fi

for i in ${labelArray[@]}; do
     sed -i '/-/d; s/^M//g' $i
     cat $i >> $LABEL_LIST
done

# python create_dataset_for_synth90k.py ./lmdb $IMG_LIST $LABEL_LIST $LEXICON
python create_dataset_for_synth90k.py ./lmdb_val $IMG_LIST $LABEL_LIST
echo "1 build dataset successfully."

cp ./lmdb_val/data.mdb $DEST_DIR
cp ./lmdb_val/lock.mdb $DEST_DIR
chmod 777 "${DEST_DIR}/data.mdb"
chmod 777 "${DEST_DIR}/lock.mdb"
echo "2 copy data.mdb to $DEST_DIR successfully."

exit 0
