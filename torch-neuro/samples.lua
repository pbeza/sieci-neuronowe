require 'image'

function convertSample(sample)
    conv = image.scale(sample, 512)
    conv = image.scale(conv, 256)
    conv = image.rgb2yuv(conv)
    return conv
end

lena = image.lena()
lena = image.scale(lena, 256)
lena = image.rgb2yuv(lena)

sampleSet = {}


function getSamples(batchSize)
    local w = lena:size(2)
	local h = lena:size(3)
    local samples = nn.Tensor(batchSize, 3, 15, 15)
    for i=1,batchSize do
        
    end
end
