require 'torch'
require 'cunn'
require 'cutorch'
require 'image'
require 'optim'
require 'paths'

torch.setdefaulttensortype('torch.FloatTensor')

opt = {
	learningRate = 1e-3,
	momentum = 0.9 * learningRate,
	batchSize = 100,
	weightDecay = 0.0
}

paths.dofile('samples.lua')
paths.dofile('model.lua')
paths.dofile('train.lua')

dataset={};
function dataset:size()
	return batchSize
end

for i=1,dataset:size() do 
	local input = lena:sub(1, 3, i, i + 2, i, i + 2)[1]
	input = input:contiguous()
	input = input:view(input:nElement())
	--input = input:contiguous()
	local output = torch.Tensor({input[2] - input[4] + input[3] - input[1]})
	--local output = torch.rand(1, 1)
	--local input = torch.FloatTensor({torch.uniform(), torch.uniform()});
	--local output = torch.Tensor(1);
	--if input[1]*input[2]>0 then
	--	output[1]=-1;
	--else
	--	output[1]=1;
	--end
	dataset[i] = {input, output};
end

mlp=nn.Sequential();  -- make a multi-layer perceptron
inputs=9; outputs=1; HUs=20;
mlp:add(nn.Linear(inputs,HUs))
mlp:add(nn.Tanh(HUs, HUs))
mlp:add(nn.Linear(HUs,outputs))

criterion = nn.MSECriterion()  
trainer = nn.StochasticGradient(mlp, criterion)
trainer.learningRate = learningRate
trainer.maxIteration = 10
trainer.shuffleIndices = false
trainer:train(dataset)

if itorch then
	local zrs = torch.zeros(lena:size(2), lena:size(3))
	local mutantlena = torch.cat( torch.cat(zrs, zrs, 1), zrs, 1 )
	local w = zrs:size(2)
	local h = zrs:size(3)
	for i=1, w-2 do
		for j=1, h-2 do
			local input = lena:sub(1, 3, i, i + 2, j, j + 2)[1]
			input = input:contiguous()
			input = input:view(input:nElement())
			local output = mlp:forward(input)
			mutantlena[{ {1}, {i}, {j} }] = output
		end
	end
	mutantlena = image.yuv2rgb(mutantlena)
	itorch.image(mutantlena)
end


