local VGGNet, parent = torch.class("dp.VGGNet", "dp.Layer")
VGGNet.isVGGNet = true

function VGGNet:__init(config)
  assert(type(config) == 'table', "Constructor requires key-value arguments")
  local args, inputSize, inputHeight, inputWidth, usingCudnn, typename = xlua.unpack(
      {config},
      'VGGNet', 
      '2x conv3-64 -> maxpool -> 2x conv3-128 -> maxpool-> 2x conv3-256 -> maxpool -> 3x conv3-512 -> maxpool -> 3x conv3-512 -> maxpool '..
      '-> fc-4096 -> fc-4096 -> fc-1000 -> logsoftmax',
      {arg='inputSize', type='number', req=true,
       help='Number of input channels or colors'},
      {arg='inputHeight', type='number', req=true,
       help='Image height'},
      {arg='inputWidth', type='number', req=true,
       help='Image width'},
      {arg='usingCudnn', type='boolean', req=true,
       help='using cudnn'},
      {arg='typename', type='string', default='VGGNet', 
       help='identifies Model type in reports.'}
    )
   
  self._input_size = inputSize
  self._input_height = inputHeight
  self._input_width = inputWidth
  
  self._param_modules = {}
  self._module = nn.Sequential()
  self._transfer = nn.ReLU()
  self._SpatialConvolution = nn.SpatialConvolutionMM
  self._SpatialMaxPooling = nn.SpatialMaxPooling
  
  if usingCudnn then
     require 'cudnn'
     print('Using cudnn')
     self._transfer = cudnn.ReLU()
     self._SpatialConvolution = cudnn.SpatialConvolution
     self._SpatialMaxPooling = cudnn.SpatialMaxPooling
  end

  self._config = {}
  -- 16-layer config

  -- Here we use 'A' configuration instead 'D' configuration to reduce memory usage
  --[[
  self._config.conv = {
    -- 2x conv3-64 
    {type = "CONV", nInputPlane = 3, nOutputPlane = 64, kernel_size = 3, pad = 1, kernel_stride = 1, transfer = self._transfer},
    {type = "CONV", nInputPlane = 64, nOutputPlane = 64, kernel_size = 3, pad = 1, kernel_stride = 1, transfer = self._transfer},
    {type = "MAXPOOL", pool_size = 2, pool_stride = 2},
    
    -- 2x conv3-128 
    {type = "CONV", nInputPlane = 64, nOutputPlane = 128, kernel_size = 3, pad = 1, kernel_stride = 1, transfer = self._transfer},
    {type = "CONV", nInputPlane = 128, nOutputPlane = 128, kernel_size = 3, pad = 1, kernel_stride = 1, transfer = self._transfer},
    {type = "MAXPOOL", pool_size = 2, pool_stride = 2},
    
    -- 2x conv3-256 
    {type = "CONV", nInputPlane = 128, nOutputPlane = 256, kernel_size = 3, pad = 1, kernel_stride = 1, transfer = self._transfer},
    {type = "CONV", nInputPlane = 256, nOutputPlane = 256, kernel_size = 3, pad = 1, kernel_stride = 1, transfer = self._transfer},
    {type = "MAXPOOL", pool_size = 2, pool_stride = 2},
    
    -- 3x conv3-512
    {type = "CONV", nInputPlane = 256, nOutputPlane = 512, kernel_size = 3, pad = 1, kernel_stride = 1, transfer = self._transfer},
    {type = "CONV", nInputPlane = 512, nOutputPlane = 512, kernel_size = 3, pad = 1, kernel_stride = 1, transfer = self._transfer},
    {type = "CONV", nInputPlane = 512, nOutputPlane = 512, kernel_size = 3, pad = 1, kernel_stride = 1, transfer = self._transfer},
    {type = "MAXPOOL", pool_size = 2, pool_stride = 2},
    
    -- 3x conv3-512
    {type = "CONV", nInputPlane = 512, nOutputPlane = 512, kernel_size = 3, pad = 1, kernel_stride = 1, transfer = self._transfer},
    {type = "CONV", nInputPlane = 512, nOutputPlane = 512, kernel_size = 3, pad = 1, kernel_stride = 1, transfer = self._transfer},
    {type = "CONV", nInputPlane = 512, nOutputPlane = 512, kernel_size = 3, pad = 1, kernel_stride = 1, transfer = self._transfer},
    {type = "MAXPOOL", pool_size = 2, pool_stride = 2},
    
  }
  --]]
  self._config.conv = {
    -- 1x conv3-64 
    {type = "CONV", nInputPlane = 3, nOutputPlane = 64, kernel_size = 3, pad = 1, kernel_stride = 1, transfer = self._transfer},
    {type = "MAXPOOL", pool_size = 2, pool_stride = 2},
    
    -- 1x conv3-128 
    {type = "CONV", nInputPlane = 64, nOutputPlane = 128, kernel_size = 3, pad = 1, kernel_stride = 1, transfer = self._transfer},
    {type = "MAXPOOL", pool_size = 2, pool_stride = 2},
    
    -- 1x conv3-256 
    {type = "CONV", nInputPlane = 128, nOutputPlane = 256, kernel_size = 3, pad = 1, kernel_stride = 1, transfer = self._transfer},
    {type = "MAXPOOL", pool_size = 2, pool_stride = 2},
    
    -- 2x conv3-512
    {type = "CONV", nInputPlane = 256, nOutputPlane = 512, kernel_size = 3, pad = 1, kernel_stride = 1, transfer = self._transfer},
    {type = "CONV", nInputPlane = 512, nOutputPlane = 512, kernel_size = 3, pad = 1, kernel_stride = 1, transfer = self._transfer},
    {type = "MAXPOOL", pool_size = 2, pool_stride = 2},
    
    -- 2x conv3-512
    {type = "CONV", nInputPlane = 512, nOutputPlane = 512, kernel_size = 3, pad = 1, kernel_stride = 1, transfer = self._transfer},
    {type = "CONV", nInputPlane = 512, nOutputPlane = 512, kernel_size = 3, pad = 1, kernel_stride = 1, transfer = self._transfer},
    {type = "MAXPOOL", pool_size = 2, pool_stride = 2},
    
  }
  self._config.fc={
    
    -- FC-4096
    {type = "FC", output_size = 4096, transfer = self._transfer},
    {type = "DROPOUT", dropout_prob = 0.5},
    
     -- FC-4096
    {type = "FC", output_size = 4096, transfer = self._transfer},
    {type = "DROPOUT", dropout_prob = 0.5},
    
    -- FC-1000
    {type = "FC", output_size = 1000},
    
    -- LOGSOFTMAX
    {type = "LOGSOFTMAX"},
  }
  
  -- Builds the conv part first
  for k,v in pairs(self._config.conv) do

    if v.type == "CONV" then
      local conv = self._SpatialConvolution(
         v.nInputPlane, v.nOutputPlane, 
         v.kernel_size, v.kernel_size, 
         v.kernel_stride, v.kernel_stride,
         v.pad
      )
      table.insert(self._param_modules, conv)
      self._module:add(conv)
      self._module:add(v.transfer:clone())
      
    elseif v.type == "MAXPOOL" then
      
      local max_pool = self._SpatialMaxPooling(
         v.pool_size, v.pool_size, 
         v.pool_stride, v.pool_stride
      )
      self._module:add(max_pool)
      
    end
  end
  -- Testing
  local output = self._module:forward(torch.Tensor(2, self._input_size, self._input_height, self._input_width))
  -- Then we need to get the output size of the conv part
  inputSize, height, width = output:size(2),output:size(3),output:size(4)
  inputSize = inputSize*height*width
  
  -- Reshapes the conv part
  self._module:add(nn.Reshape(inputSize))

  -- Builds the fc part
  for k,v in pairs(self._config.fc) do
    
    if v.type == "FC" then
      self._module:add(nn.Linear(inputSize,v.output_size))
      if v.transfer then
        self._module:add(v.transfer:clone())
      end
      inputSize = v.output_size
    elseif v.type == "DROPOUT" then
      self._module:add(nn.Dropout(v.dropout_prob))
    end
    
  end
  
  config.typename = typename
  config.input_view = 'bchw'
  config.output_view = 'bf'
  config.output = dp.ClassView()
  parent.__init(self, config)
end

function VGGNet:reset(stdv)
   self._module:reset(stdv)
   if self._sparse_init then
      for i, modula in ipairs(self._param_modules) do
         local W = modula.weight
         self._sparseReset(W:view(W:size(1), -1))
      end
   end
end

function VGGNet:maxNorm(max_out_norm, max_in_norm)
   assert(self.backwarded, "Should call maxNorm after a backward pass")
   max_out_norm = self.mvstate.max_out_norm or max_out_norm
   max_in_norm = self.mvstate.max_in_norm or max_in_norm
   for i, modula in ipairs(self._param_modules) do
      local W = modula.weight
      W = W:view(W:size(1), -1)
      if max_out_norm then
         W:renorm(1, 2, max_out_norm)
      end
      if max_in_norm then
         W:renorm(2, 2, max_in_norm)
      end
   end
end

function VGGNet:share(vgg, ...)
   assert(VGGNet.isVGGNet)
   return parent.share(self, vgg, ...)
end

-- output size of the model (excluding batch dim)
function VGGNet:outputSize(inputHeight, inputWidth, view)
   local input = torch.Tensor(2, self._input_size, inputHeight, inputWidth)
   local inputView = dp.ImageView('bchw', input)
   -- just propagate this dummy input through to know the output size
   local output = self:forward(inputView, dp.Carry{nSample=2}):forward(view or 'bchw')
   self:zeroStatistics()
   return output:size(2), output:size(3), output:size(4)
end