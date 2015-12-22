function createModelHuge()
   local features = nn.Sequential()
   features:add(cudnn.SpatialConvolution(3,64,11,11,4,4,2,2))       -- 224 -> 55
   features:add(nn.SpatialBatchNormalization(64,1e-3))
   features:add(cudnn.ReLU(true))
   features:add(cudnn.SpatialMaxPooling(3,3,2,2))                   -- 55 ->  27
   features:add(cudnn.SpatialConvolution(64,192,5,5,1,1,2,2))       --  27 -> 27
   features:add(nn.SpatialBatchNormalization(192,1e-3))
   features:add(cudnn.ReLU(true))
   features:add(cudnn.SpatialMaxPooling(3,3,2,2))                   --  27 ->  13
   features:add(cudnn.SpatialConvolution(192,384,3,3,1,1,1,1))      --  13 ->  13
   features:add(nn.SpatialBatchNormalization(384,1e-3))
   features:add(cudnn.ReLU(true))
   features:add(cudnn.SpatialConvolution(384,256,3,3,1,1,1,1))      --  13 ->  13
   features:add(nn.SpatialBatchNormalization(256,1e-3))
   features:add(cudnn.ReLU(true))
   features:add(cudnn.SpatialConvolution(256,256,3,3,1,1,1,1))      --  13 ->  13
   features:add(nn.SpatialBatchNormalization(256,1e-3))
   features:add(cudnn.ReLU(true))
   features:add(cudnn.SpatialMaxPooling(3,3,2,2))                   -- 13 -> 6

   features:cuda()

   local classifier = nn.Sequential()
   classifier:add(nn.View(256*6*6))

   classifier:add(nn.Dropout(0.5))
   classifier:add(nn.Linear(256*6*6, 4096))
   classifier:add(nn.BatchNormalization(4096, 1e-3))
   classifier:add(nn.ReLU())

   classifier:add(nn.Dropout(0.5))
   classifier:add(nn.Linear(4096, 4096))
   classifier:add(nn.BatchNormalization(4096, 1e-3))
   classifier:add(nn.ReLU())

   classifier:add(nn.Linear(4096, nClasses))
   classifier:add(nn.LogSoftMax())

   classifier:cuda()

   local model = nn.Sequential():add(features):add(classifier)
   model.imageSize = 256
   model.imageCrop = 224

   return model
end

function createModel()
    local features = nn.Sequential()
    features:add(cudnn.SpatialConvolution(3, 32, 3, 3, 2, 2))
    features:add(nn.SpatialBatchNormalization(32,1e-3))
    features:add(cudnn.ReLU(true))
    features:add(cudnn.SpatialMaxPooling(3,3,2,2))
    features:add(cudnn.SpatialConvolution(32,16,2,2))
    features:add(nn.SpatialBatchNormalization(16,1e-3))
    features:add(cudnn.ReLU(true))
    
    features:cuda()

    local classifier = nn.Sequential()
    classifier:add(nn.View(16*2*2))

    classifier:add(nn.Dropout(0.5))
    classifier:add(nn.Linear(16*2*2, 16))
    classifier:add(nn.BatchNormalization(16, 1e-3))
    classifier:add(nn.ReLU())

    classifier:add(nn.Dropout(0.5))
    classifier:add(nn.Linear(16, 4))
    classifier:add(nn.LogSoftMax())
    
    classifier:cuda()

    local model = nn.Sequential():add(features):add(classifier)
    return model
    --ret = sq:forward(torch.ones(4,3,15,15))
end
