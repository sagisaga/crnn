require('cutorch')
require('nn')
require('cunn')
require('cudnn')
require('optim')
require('paths')
require('nngraph')

require('libcrnn')
require('utilities')
require('inference')
require('CtcCriterion')
require('DatasetLmdb')
require('LstmLayer')
require('BiRnnJoin')
require('SharedParallelTable')


cutorch.setDevice(1)
torch.setnumthreads(4)
torch.setdefaulttensortype('torch.FloatTensor')

print('Loading model...')
local modelDir = '../model/crnn_demo/'
paths.dofile(paths.concat(modelDir, 'config.lua'))
-- local modelLoadPath = paths.concat(modelDir, 'model.t7')
local modelLoadPath = paths.concat(modelDir, 'snapshot_410000.t7')
gConfig = getConfig()
gConfig.modelDir = modelDir
gConfig.maxT = 0
local model, criterion = createModel(gConfig)
print('model path :', modelLoadPath)
local snapshot = torch.load(modelLoadPath)
loadModelState(model, snapshot)
model:evaluate()
print(string.format('Model loaded from %s', modelLoadPath))

-- recognize text image

-- [[
-- local imagePath = '../data/demo.png'
local imagePath = '../data/yellowknife.jpg'
-- local imagePath = '../data/Pasteurization.jpg'
local img = loadAndResizeImage(imagePath)
local text, raw = recognizeImageLexiconFree(model, img)
print(string.format('Recognized text: %s (raw: %s)', text, raw)) 
-- ]]

[[
local f = assert(io.open('./engWordList.txt', 'r'))
local fstring = f:read()
local tTotal = 0
local imageNum = 0
while fstring do
	    local t1 = os.clock()
		local imagePath = fstring
		print('proc image : ', imagePath)
		local img = loadAndResizeImage(imagePath)
		local text, raw = recognizeImageLexiconFree(model, img)
	    local t2=os.clock()
	    local tu=t2-t1
		tTotal =tTotal+ tu
		imageNum =imageNum+ 1
		print(string.format('Recognized text: %s (raw: %s),time:%.3f', text, raw,tu))
	    fstring = f:read()
end
print("total time:",tTotal,"image number:",imageNum,"avgTime",tTotal/imageNum)
f:close()
]]

