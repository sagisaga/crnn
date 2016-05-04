import os
import lmdb
import cv2
import sys
import numpy as np
def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.fromstring(imageBin, dtype=np.uint8)
    if len(imageBuf)<10:
        return False    
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.iteritems():
            #txn.put(k, v)
            try:	
                if (k.find('ipath')>-1):
                    print 'k : %s v : %s' % (k, v)
                txn.put(k, v)
            except:
                print("txn put error")
                print("kv: %s,%s",k,v)
                print cache
                print env.info()


def createDataset(outputPath, imagePathList, labelList, lexiconList=None, checkValid=True):
    """
    Create LMDB dataset for CRNN training.

    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    assert(len(imagePathList) == len(labelList))
    nSamples = len(imagePathList)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    for i in xrange(nSamples):
        imagePath = imagePathList[i]
        label = labelList[i]
        #print 'imagePath',imagePath
        #print 'label',label
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'r') as f:
            imageBin = f.read()
        if checkValid:
            try: 
                if not checkImageIsValid(imageBin):
                    print('%s is not a valid image' % imagePath)
                    continue
            except:
                print('image read error: %s'%imagePath)

        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt
        pathKey  = 'ipath-%09d' % cnt		# add image path for debug

        # print 'key %s path %s' % (pathKey, imagePath)

        cache[imageKey] = imageBin
        cache[labelKey] = label
        cache[pathKey]  = imagePath

        #cache[labelKey] = lexiconList[int(label)]

        #if lexiconList:
        #    lexiconKey = 'lexicon-%09d' % cnt
        #    cache[lexiconKey] = ' '.join(lexiconList[i])
        if cnt % 100 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt-1
    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


if __name__ == '__main__':
   print sys.argv[1],sys.argv[2],sys.argv[3] #,sys.argv[4]
   
   f = open(sys.argv[2],"r")
   imageList =[] 
   for line in f.readlines():
       newLine = line.strip('\n')
       imageList.append(newLine)
       #print newLine
   f.close()

   labelsList=[]
   f = open(sys.argv[3],"r")
   for line in f.readlines():
       newLine = line.strip('\n')
       labelsList.append(newLine)
       #print newLine
   f.close()

   createDataset(sys.argv[1],imageList,labelsList)
'''
   lexiconList=[]
   f = open(sys.argv[4],"r")
   for line in f.readlines():
       newLine = line.strip('\n')
       lexiconList.append(newLine)
       #print newLine
   f.close()
'''

