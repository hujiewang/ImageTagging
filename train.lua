require 'dp'

--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Image Classification using VGGNet')
cmd:text('Example:')
cmd:text('$> th train.lua --batchSize 128 --momentum 0.5')
cmd:text('Options:')
cmd:option('--learningRate', 0.1, 'learning rate at t=0')
cmd:option('--decayPoint', 10, 'epoch at which learning rate is decayed')
cmd:option('--decayFactor', 0.0005, 'factory by which learning rate is decayed at decay point')
cmd:option('--maxOutNorm', 1, 'max norm each layers output neuron weights')
cmd:option('--maxNormPeriod', 2, 'Applies MaxNorm Visitor every maxNormPeriod batches')
cmd:option('--momentum', 0, 'momentum')
cmd:option('--batchSize', 128, 'number of examples per batch')
cmd:option('--cuda', true, 'use CUDA')
cmd:option('--useDevice', 1, 'sets the device (GPU) to use')
cmd:option('--maxEpoch', 100, 'maximum number of epochs to run')
cmd:option('--maxTries', 30, 'maximum number of epochs to try to find a better local minima for early-stopping')
cmd:option('--dataset', 'Mnist', 'which dataset to use : Mnist | NotMnist | Cifar10 | Cifar100 | Svhn')
cmd:option('--standardize', false, 'apply Standardize preprocessing')
cmd:option('--zca', false, 'apply Zero-Component Analysis whitening')
cmd:option('--lecunlcn', false, 'apply Yann LeCun Local Contrast Normalization (recommended)')
cmd:option('--normalInit', false, 'initialize inputs using a normal distribution (as opposed to sparse initialization)')
cmd:option('--progress', true, 'print progress bar')
cmd:option('--silent', false, 'dont print anything to stdout')
cmd:text()
opt = cmd:parse(arg or {})

if not opt.silent then
  table.print(opt)
end

--[[preprocessing]]--
local input_preprocess = {}
if opt.standardize then
  table.insert(input_preprocess, dp.Standardize())
end
if opt.zca then
  table.insert(input_preprocess, dp.ZCA())
end
if opt.lecunlcn then
  table.insert(input_preprocess, dp.GCN())
  table.insert(input_preprocess, dp.LeCunLCN{progress=true})
end

--[[data]]--

local datasource
if opt.dataset == 'Mnist' then
  datasource = dp.Mnist{input_preprocess = input_preprocess}
elseif opt.dataset == 'NotMnist' then
  datasource = dp.NotMnist{input_preprocess = input_preprocess}
elseif opt.dataset == 'Cifar10' then
  datasource = dp.Cifar10{input_preprocess = input_preprocess}
elseif opt.dataset == 'Cifar100' then
  datasource = dp.Cifar100{input_preprocess = input_preprocess}
elseif opt.dataset == 'Svhn' then
  datasource = dp.Svhn{input_preprocess = input_preprocess}
else
  error("Unknown Dataset")
end

--[[model]]--

mlp = dp.Sequential{
  models = {
    VGGNet
  }
}

local visitor = {
  dp.Momentum{momentum_factor = opt.momentum},
  dp.Learn{
    learning_rate = opt.learningRate, 
    observer = dp.LearningRateSchedule{
      schedule = {[opt.decayPoint]=opt.learningRate*opt.decayFactor}
    }
  },
  dp.MaxNorm{max_out_norm = opt.maxOutNorm, period=opt.maxNormPeriod}
}

--[[Propagators]]--
train = dp.Optimizer{
  loss = dp.NLL(),
  visitor = visitor,
  feedback = dp.Confusion(),
  sampler = dp.ShuffleSampler{batch_size = opt.batchSize},
  progress = opt.progress
}
valid = dp.Evaluator{
  loss = dp.NLL(),
  feedback = dp.Confusion(),  
  sampler = dp.Sampler{batch_size = opt.batchSize}
}
test = dp.Evaluator{
  loss = dp.NLL(),
  feedback = dp.Confusion(),
  sampler = dp.Sampler{batch_size = opt.batchSize}
}

--[[Experiment]]--
xp = dp.Experiment{
  model = mlp,
  optimizer = train,
  validator = valid,
  tester = test,
  observer = {
    dp.FileLogger(),
    dp.EarlyStopper{
      error_report = {'validator','feedback','confusion','accuracy'},
      maximize = true,
      max_epochs = opt.maxTries
    }
  },
  random_seed = os.time(),
  max_epoch = opt.maxEpoch
}

--[[GPU or CPU]]--
if opt.cuda then
  require 'cutorch'
  require 'cunn'
  cutorch.setDevice(opt.useDevice)
  xp:cuda()
end

if not opt.silent then
  print"dp.Models :"
  print(cnn)
  print"nn.Modules :"
  print(mlp:toModule(datasource:trainSet():sub(1,32)))
end
xp:verbose(not opt.silent)

xp:run(datasource)