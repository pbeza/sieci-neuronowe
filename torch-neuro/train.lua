require 'optim'

local batchNumber
local top1_epoch, loss_epoch

local optimState = {
    learningRate = opt.learningRate,
    learningRateDecay = 0.0,
    momentum = opt.momentum,
    dampening = 0.0,
    weightDecay = opt.weightDecay
}

function train()
   batchNumber = 0
   cutorch.synchronize()

   model:training()

   top1_epoch = 0
   loss_epoch = 0
   for i=1,opt.epochSize do
        local inputs, labels = getSamples(opt.batchSize)
        trainBatch(inputs, labels)
   end

   cutorch.synchronize()

   top1_epoch = top1_epoch * 100 / (opt.batchSize * opt.epochSize)
   loss_epoch = loss_epoch / opt.epochSize

   collectgarbage()
end

criterion = nn.ClassNLLCriterion()
criterion:cuda()

local inputs = torch.CudaTensor()
local labels = torch.CudaTensor()

function trainBatch(inputsCPU, labelsCPU)
   cutorch.synchronize()
   collectgarbage()

   -- transfer over to GPU
   inputs:resize(inputsCPU:size()):copy(inputsCPU)
   labels:resize(labelsCPU:size()):copy(labelsCPU)
   
   local err, outputs
   feval = function(x)
      model:zeroGradParameters()
      outputs = model:forward(inputs)
      err = criterion:forward(outputs, labels)
      local gradOutputs = criterion:backward(outputs, labels)
      model:backward(inputs, gradOutputs)
      return err, gradParameters
   end
   optim.sgd(feval, parameters, optimState)

   model:apply(function(m) if m.syncParameters then m:syncParameters() end end)

   cutorch.synchronize()
   batchNumber = batchNumber + 1
   loss_epoch = loss_epoch + err
   -- top-1 error
   local top1 = 0
   do
      local _,prediction_sorted = outputs:float():sort(2, true) -- descending
      for i=1,opt.batchSize do
	 if prediction_sorted[i][1] == labelsCPU[i] then
	    top1_epoch = top1_epoch + 1;
	    top1 = top1 + 1
	 end
      end
      top1 = top1 * 100 / opt.batchSize;
   end
end
