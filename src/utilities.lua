local Image = require('image')


function str2label(strs, maxLength)
    --[[ Convert a list of strings to integer label tensor (zero-padded).

    ARGS:
      - `strs`     : table, list of strings
      - `maxLength`: int, the second dimension of output label tensor

    RETURN:
      - `labels`   : tensor of shape [#(strs) x maxLength]
    ]]
    assert(type(strs) == 'table')

    function ascii2label(ascii)
        local label
        if ascii >= 48 and ascii <= 57 then -- '0'-'9' are mapped to 1-10
            label = ascii - 47
        elseif ascii >= 65 and ascii <= 90 then -- 'A'-'Z' are mapped to 11-36
            label = ascii - 64 + 10
        elseif ascii >= 97 and ascii <= 122 then -- 'a'-'z' are mapped to 11-36
            label = ascii - 96 + 10
        end
        return label
    end

    local nStrings = #strs
    local labels = torch.IntTensor(nStrings, maxLength):fill(0)
    for i, str in ipairs(strs) do
        for j = 1, string.len(str) do
            local ascii = string.byte(str, j)
            labels[i][j] = ascii2label(ascii)
        end
    end
    return labels
end


function label2str(labels, raw)
    --[[ Convert a label tensor to a list of strings.

    ARGS:
      - `labels`: int tensor, labels
      - `raw`   : boolean, if true, convert zeros to '-'

    RETURN:
      - `strs`  : table, list of strings
    ]]
    assert(labels:dim() == 2)
    raw = raw or false

    function label2ascii(label)
        local ascii
        if label >= 1 and label <= 10 then
            ascii = label - 1 + 48
        elseif label >= 11 and label <= 36 then
            ascii = label - 11 + 97
        elseif label == 0 then -- used when displaying raw predictions
            ascii = string.byte('-')
        end
        return ascii
    end

    local strs = {}
    local nStrings, maxLength = labels:size(1), labels:size(2)
    for i = 1, nStrings do
        local str = {}
        local labels_i = labels[i]
        for j = 1, maxLength do
            if raw then
                str[j] = label2ascii(labels_i[j])
            else
                if labels_i[j] == 0 then
                    break
                else
                    str[j] = label2ascii(labels_i[j])
                end
            end
        end
        str = string.char(unpack(str))
        strs[i] = str
    end
    return strs
end


function setupLogger(fpath)
    local fileMode = 'w'

    --[[if paths.filep(fpath) then
        local input = nil
        while not input do
            print('Logging file exits, overwrite(o)? append(a)? abort(q)?')
            input = io.read()
            if input == 'o' then
                fileMode = 'w'
            elseif input == 'a' then
                fileMode = 'a'
            elseif input == 'q' then
                os.exit()
            else
                fileMode = nil
            end
        end
    end
	--]]
    gLoggerFile = io.open(fpath, fileMode)
end


function tensorInfo(x, name)
    local name = name or ''
    local sizeStr = ''
    for i = 1, #x:size() do
        sizeStr = sizeStr .. string.format('%d', x:size(i))
        if i < #x:size() then
            sizeStr = sizeStr .. 'x'
        end
    end
    infoStr = string.format('[%15s] size: %12s, min: %+.2e, max: %+.2e', name, sizeStr, x:min(), x:max())
    return infoStr
end


function shutdownLogger()
    if gLoggerFile then
        gLoggerFile:close()
    end
end


function logging(message, mute)
    mute = mute or false
    local timeStamp = os.date('%x %X')
    local msgFormatted = string.format('[%s]  %s', timeStamp, message)
    if not mute then
        print(msgFormatted)
    end
    if gLoggerFile then
        gLoggerFile:write(msgFormatted .. '\n')
        gLoggerFile:flush()
    end
end


function modelSize(model)
    local params = model:parameters()
    local count = 0
    local countForEach = {}
    for i = 1, #params do
        local nParam = params[i]:numel()
        count = count + nParam
        countForEach[i] = nParam
    end
    return count, torch.LongTensor(countForEach)
end


function cloneList(tensors, fillZero)
    --[[ Clone a list of tensors, adapted from https://github.com/karpathy/char-rnn
    ARGS:
      - `tensors`  : table, list of tensors
      - `fillZero` : boolean, if true tensors are filled with zeros
    RETURNS:
      - `output`   : table, cloned list of tensors
    ]]
    local output = {}
    for k, v in pairs(tensors) do
        output[k] = v:clone()
        if fillZero then output[k]:zero() end
    end
    return output
end


function cloneManyTimes(net, T)
    --[[ Clone a network module T times, adapted from https://github.com/karpathy/char-rnn
    ARGS:
      - `net`    : network module to be cloned
      - `T`      : integer, number of clones
    RETURNS:
      - `clones` : table, list of clones
    ]]
    local clones = {}
    local params, gradParams = net:parameters()
    local mem = torch.MemoryFile("w"):binary()
    mem:writeObject(net)
    for t = 1, T do
        local reader = torch.MemoryFile(mem:storage(), "r"):binary()
        local clone = reader:readObject()
        reader:close()
        local cloneParams, cloneGradParams = clone:parameters()
        if params then
            for i = 1, #params do
                cloneParams[i]:set(params[i])
                cloneGradParams[i]:set(gradParams[i])
            end
        end
        clones[t] = clone
        collectgarbage()
    end
    mem:close()
    return clones
end


function diagnoseGradients(params, gradParams)
    --[[ Diagnose gradients by checking the value range and the ratio of the norms
    ARGS:
      - `params`     : first arg returned by net:parameters()
      - `gradParams` : second arg returned by net:parameters()
    ]]
    for i = 1, #params do
        local pMin = params[i]:min()
        local pMax = params[i]:max()
        local gpMin = gradParams[i]:min()
        local gpMax = gradParams[i]:max()
        local normRatio = gradParams[i]:norm() / params[i]:norm()
        --logging(string.format('%02d - params [%+.2e, %+.2e] gradParams [%+.2e, %+.2e], norm gp/p %+.2e',
        --    i, pMin, pMax, gpMin, gpMax, normRatio), true)

        logging(string.format('%02d - params [%10f, %10f] gradParams [%10f, %10f], norm gp/p %10f',
            i, pMin, pMax, gpMin, gpMax, normRatio), true)
    end
end


function modelState(model)
    --extract the state of model, fill the state with network parameters
    --[[ Get model state, including model parameters (weights and biases) and
         running mean/std in batch normalization layers
    ARGS:
      - `model` : network model
    RETURN:
      - `state` : table, model states
    ]]
    --model is nn.sequential, structure in torch
    local parameters = model:parameters()   -- parameter table
    local bnVars = {}                       -- empty table

    --findModules(typename)
    --Find all instances of modules in the network of a certain typename.
    --It returns a flattened list of the matching nodes, as well as a flattened list of the container modules for each matching node.
    --Modules that do not have a parent container (ie, a top level nn.Sequential for instance) will return their self as the container.
    --This function is very helpful for navigating complicated nested networks
    
    local bnLayers = model:findModules('nn.BatchNormalization')
    for i = 1, #bnLayers do
        --add tail
        bnVars[#bnVars+1] = bnLayers[i].running_mean
        -- bnVars[#bnVars+1] = bnLayers[i].running_std
        bnVars[#bnVars+1] = bnLayers[i].running_var
    end

    local bnLayers = model:findModules('nn.SpatialBatchNormalization')
    for i = 1, #bnLayers do
        --add tail
        bnVars[#bnVars+1] = bnLayers[i].running_mean
        -- bnVars[#bnVars+1] = bnLayers[i].running_std
        bnVars[#bnVars+1] = bnLayers[i].running_var
    end

    --zjc
    local scTable=model:findModules("cudnn.SpatialConvolution")
    print("spatial convolution\n", scTable)
    print("spatial batch normalization\n", bnLayers)
    local lstmTable=model:findModules("nn.LstmLayer")
    --print("lstm\n", lstmTable)
    
    -- output lstm layer parameters
    --[[
    file=io.open("lstmLayers.txt", "w")
    file:write("lstm layer nubmer : ", #lstmTable, "\n")
    for i=1, #lstmTable do
        local weight=lstmTable[i].lstmUnit.modules[2].weight
        file:write("\nlayer : ", i, " i2h weight ", weight:size(1), " * ", weight:size(2), "\n\n")
        for m=1, weight:size(1) do
            for n=1, weight:size(2) do
                file:write(string.format("% 6.4f ", weight[m][n]))
            end
            file:write("\n")
        end

        local bias=lstmTable[i].lstmUnit.modules[2].bias
        file:write("\nlayer : ", i, " i2h bias ", bias:size(1), "\n\n")
        for m=1, bias:size(1) do
            file:write(string.format("% 6.4f", bias[m]))
        end
        file:write("\n")

        local weight_2=lstmTable[i].lstmUnit.modules[4].weight
        file:write("\nlayer : ", i, " h2h weight ", weight_2:size(1), " * ", weight_2:size(2), "\n\n")
        for m=1, weight_2:size(1) do
            for n=1, weight_2:size(2) do
                file:write(string.format("% 6.4f ", weight_2[m][n]))
            end
            file:write("\n")
        end

        local bias_2=lstmTable[i].lstmUnit.modules[4].bias
        file:write("\nlayer : ", i, " h2h bias ", bias_2:size(1), "\n\n")
        for m=1, bias_2:size(1) do
            file:write(string.format("% 6.4f", bias_2[m]))
        end
        file:write("\n")

    end

    file:close()
    ]]

    --construct a tabel for torch.save
    local state = {parameters = parameters, bnVars = bnVars}
    return state
end


function loadModelState(model, stateToLoad)

    --zjc
    --stateToLoad is the snapshot
    print("stateToLoad\n")
    print(stateToLoad)
    --local modelSize, nParamsEachLayer=modelSize(stateToLoad)
    --io.write(string.format('Model size: %d\n%s', modelSize, nParamsEachLayer))

    local state = modelState(model)
    
    assert(#state.parameters == #stateToLoad.parameters)
	
	-- print('state.bnVars : ', state.bnVars)
	-- print('stateToLoad.bnVars : ', stateToLoad.bnVars)
    
	assert(#state.bnVars == #stateToLoad.bnVars)


    --44
    for i = 1, #state.parameters do
        state.parameters[i]:copy(stateToLoad.parameters[i])
    end

    --6
    for i = 1, #state.bnVars do
        state.bnVars[i]:copy(stateToLoad.bnVars[i])
    end

    --zjc
    print("state :", state)
    print("state.parameters number :", #state.parameters)
    print("state.bnVars number :", #state.bnVars)
    print("state.parameters : ", state.parameters)
	print("state.bnVars : ", state.bnVars)

    output_network(state)

end

function output_network(state)
    --output the whole network
    file=io.open("model_params.dat", "w")
    --cnn 1 weight
    local tensor=state.parameters[1]
    output_conv_table(tensor, file, "cnn 1 weight\n")
    --cnn 1 bias
    tensor=state.parameters[2]
    output_vector(tensor, file, "cnn 1 bias\n")
    --cnn 2 weight
    tensor=state.parameters[3]
    output_conv_table(tensor, file, "cnn 2 weight\n")
    --cnn 2 bias
    tensor=state.parameters[4]
    output_vector(tensor, file, "cnn 2 bias\n")
    --cnn 3 weight
    tensor=state.parameters[5]
    output_conv_table(tensor, file, "cnn 3 weight\n")
    --cnn 3 bias
    tensor=state.parameters[6]
    output_vector(tensor, file, "cnn 3 bias\n")
    local bnVars=state.bnVars
    --BatchNormalization
    output_vector(bnVars[1], file, "BatchNormalization mean\n")
    output_vector(bnVars[2], file, "BatchNormalization var\n")
    tensor=state.parameters[7]
    output_vector(tensor, file, "BatchNormalization gamma\n")
    tensor=state.parameters[8]
    output_vector(tensor, file, "BatchNormalization deta\n")
    --cnn 4 weight
    tensor=state.parameters[9]
    output_conv_table(tensor, file, "cnn 4 weight\n")
    --cnn 4 bias
    tensor=state.parameters[10]
    output_vector(tensor, file, "cnn 4 bias\n")
    --cnn 5 weight
    tensor=state.parameters[11]
    output_conv_table(tensor, file, "cnn 5 weight\n")
    --cnn 5 bias
    tensor=state.parameters[12]
    output_vector(tensor, file, "cnn 5 bias\n")
    --BatchNormalization
    output_vector(bnVars[3], file, "BatchNormalization mean\n")
    output_vector(bnVars[4], file, "BatchNormalization var\n")
    tensor=state.parameters[13]
    output_vector(tensor, file, "BatchNormalization gamma\n")
    tensor=state.parameters[14]
    output_vector(tensor, file, "BatchNormalization deta\n")
    --cnn 6 weight
    tensor=state.parameters[15]
    output_conv_table(tensor, file, "cnn 6 weight\n")
    --cnn 6 bias
    tensor=state.parameters[16]
    output_vector(tensor, file, "cnn 6 bias\n")
    --cnn 7 weight
    tensor=state.parameters[17]
    output_conv_table(tensor, file, "cnn 7 weight\n")
    --cnn 7 bias
    tensor=state.parameters[18]
    output_vector(tensor, file, "cnn 7 bias\n")
    --BatchNormalization
    output_vector(bnVars[5], file, "BatchNormalization mean\n")
    output_vector(bnVars[6], file, "BatchNormalization var\n")
    tensor=state.parameters[19]
    output_vector(tensor, file, "BatchNormalization gamma\n")
    tensor=state.parameters[20]
    output_vector(tensor, file, "BatchNormalization deta\n")
    --lstm 1 forward
    tensor=state.parameters[21]
    output_matrix(tensor, file, "lstm 1 forward i2h weight\n")
    tensor=state.parameters[22]
    output_vector(tensor, file, "lstm 1 forward i2h bias\n")
    tensor=state.parameters[23]
    output_matrix(tensor, file, "lstm 1 forward h2h weight\n")
    tensor=state.parameters[24]
    output_vector(tensor, file, "lstm 1 forward h2h bias\n")
    --lstm 1 backward
    tensor=state.parameters[25]
    output_matrix(tensor, file, "lstm 1 backward i2h weight\n")
    tensor=state.parameters[26]
    output_vector(tensor, file, "lstm 1 backward i2h bias\n")
    tensor=state.parameters[27]
    output_matrix(tensor, file, "lstm 1 backward h2h weight\n")
    tensor=state.parameters[28]
    output_vector(tensor, file, "lstm 1 backward h2h bias\n")
    --full connect
    tensor=state.parameters[29]
    output_matrix(tensor, file, "lstm 1 forward full connect weight\n")
    tensor=state.parameters[30]
    output_vector(tensor, file, "lstm 1 forward full connect bias\n")
    tensor=state.parameters[31]
    output_matrix(tensor, file, "lstm 1 backward full connect weight\n")
    tensor=state.parameters[32]
    output_vector(tensor, file, "lstm 1 backward full connect bias\n")
    --lstm 2 forward
    tensor=state.parameters[33]
    output_matrix(tensor, file, "lstm 2 forward i2h weight\n")
    tensor=state.parameters[34]
    output_vector(tensor, file, "lstm 2 forward i2h bias\n")
    tensor=state.parameters[35]
    output_matrix(tensor, file, "lstm 2 forward h2h weight\n")
    tensor=state.parameters[36]
    output_vector(tensor, file, "lstm 2 forward h2h bias\n")
    --lstm 2 backward
    tensor=state.parameters[37]
    output_matrix(tensor, file, "lstm 2 backward i2h weight\n")
    tensor=state.parameters[38]
    output_vector(tensor, file, "lstm 2 backward i2h bias\n")
    tensor=state.parameters[39]
    output_matrix(tensor, file, "lstm 2 backward h2h weight\n")
    tensor=state.parameters[40]
    output_vector(tensor, file, "lstm 2 backward h2h bias\n")
    --full connect
    tensor=state.parameters[41]
    output_matrix(tensor, file, "lstm 2 forward full connect weight\n")
    tensor=state.parameters[42]
    output_vector(tensor, file, "lstm 2 forward full connect bias\n")
    tensor=state.parameters[43]
    output_matrix(tensor, file, "lstm 2 backward full connect weight\n")
    tensor=state.parameters[44]
    output_vector(tensor, file, "lstm 2 backward full connect bias\n")


    file:close()
end

function output_matrix(tensor, file, str)
    file:write(str)
    local I=tensor:size(1)
    local J=tensor:size(2)
    file:write(string.format("Dim : %3d * %3d\n\n", I, J))
    for i=1, I do
        for j=1, J do
            file:write(string.format("% 6.2f ", tensor[i][j]))
        end
        file:write("\n")
    end
    file:write("\n")
    file:flush()
end

function output_vector(tensor, file, str)
    file:write(str)
    local num=tensor:size(1)
    file:write(string.format("Dim : %3d\n\n", num))
    for i=1, num do
        file:write(string.format("% 6.4f ", tensor[i]))
    end
    file:write("\n\n")
    file:flush()
end

function output_conv_table(tensor, file, str)
    --conv table MxNxIxJ
    file:write(str)
    local M=tensor:size(1)
    local N=tensor:size(2)
    local I=tensor:size(3)
    local J=tensor:size(4)
    file:write(string.format("Dim : %3d * %3d * %3d * %3d\n\n", M, N, I, J))
    for m=1, M do
        for n=1, N do
            for i=1, I do
                for j=1, J do
                    file:write(string.format("% 6.4f ", tensor[m][n][i][j]))
                end
                file:write("\n")
            end
            file:write("\n")
        end
    end

    file:flush()
end

function loadAndResizeImage(imagePath)
    local img = Image.load(imagePath, 3, 'byte')
    img = Image.rgb2y(img)
    img = Image.scale(img, 100, 32)[1]
    return img
end


function recognizeImageLexiconFree(model, image)
    --[[ Lexicon-free text recognition.
    ARGS:
      - `model`   : CRNN model
      - `image`   : single-channel image, byte tensor
    RETURN:
      - `str`     : recognized string
      - `rawStr`  : raw recognized string
    ]]
    assert(image:dim() == 2 and image:type() == 'torch.ByteTensor',
        'Input image should be single-channel byte tensor')
    image = image:view(1, 1, image:size(1), image:size(2))
	-- print("recognizeImageLexiconFree")
    local output = model:forward(image)

    function label2ascii(label)
        local ascii
        if label >= 1 and label <= 10 then
            ascii = label - 1 + 48
        elseif label >= 11 and label <= 36 then
            ascii = label - 11 + 97
        elseif label == 0 then -- used when displaying raw predictions
            ascii = string.byte('-')
        end
        return ascii
    end

	--print('output : ', output)
    --output prob matrix in file
    local file=io.open("cnn_mid_result.txt", "a")

    for i=0, 36 do
        file:write(string.format("    %c   ", label2ascii(i)))
        --print(string.format("%c ", label2ascii(i)))
    end

    file:write("\n")

    for i=1, 26 do
        maxp=tonumber(output[1][i][1])
        prop=0
        maxI=1

        for k=2, 37 do
            prop=tonumber(output[1][i][k])
            if prop==0.0 or prop==-0.0 then
                maxI=k
                break
            end

            if (prop>maxp) then
                 --print(prop, ">", maxp, " maxI : ", k)
                 maxp=prop
                 maxI=k
            end
        end
        
        print(string.format("%2d : %c", i, label2ascii(maxI-1)))

        for j=1, 37 do
            if j==maxI then
                file:write(string.format("*%6.2f ", output[1][i][j]))
            else
                file:write(string.format(" %6.2f ", output[1][i][j]))
            end
        end
        file:write("\n")
    end

    file:close()
    -- end output

    local pred, predRaw = naiveDecoding(output)
    local str = label2str(pred)[1]
    local rawStr = label2str(predRaw, true)[1]
    return str, rawStr
end


function recognizeImageWithLexicion(model, image, lexicon)
    --[[ Text recognition with a lexicon.
    ARGS:
      - `imagePath` : string, image path
      - `lexicon`   : list of string, lexicon words
    RETURN:
      - `str`       : recognized string
    ]]
    assert(image:dim() == 2 and image:type() == 'torch.ByteTensor',
        'Input image should be single-channel byte tensor')
    image = image:view(1, 1, image:size(1), image:size(2))
    local output = model:forward(image)
    local str = decodingWithLexicon(output, lexicon)
    return str
end
