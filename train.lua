require 'dp'
require 'VGGNet'
require 'SaveModel'
--require('mobdebug').start()
--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Image Classification using VGGNet')
cmd:text('Example:')
cmd:text('$> th train.lua --batchSize 256 --momentum 0.9')
cmd:text('Options:')
cmd:option('--dataPath', paths.concat(dp.DATA_DIR, 'ImageNet'), 'path to ImageNet')
cmd:option('--trainPath', '', 'Path to train set. Defaults to --dataPath/ILSVRC2012_img_train')
cmd:option('--validPath', '', 'Path to valid set. Defaults to --dataPath/ILSVRC2012_img_val')
cmd:option('--metaPath', '', 'Path to metadata. Defaults to --dataPath/metadata')
cmd:option('--learningRate', 0.01, 'learning rate at t=0')
cmd:option('--maxOutNorm', -1, 'max norm each layers output neuron weights')
cmd:option('-weightDecay', 5e-4, 'weight decay')
cmd:option('--maxNormPeriod', 1, 'Applies MaxNorm Visitor every maxNormPeriod batches')
cmd:option('--momentum', 0.9, 'momentum') 
cmd:option('--batchSize', 18, 'number of examples per batch')
cmd:option('--cuda', true, 'use CUDA')
cmd:option('--useDevice', 1, 'sets the device (GPU) to use')
cmd:option('--trainEpochSize', -1, 'number of train examples seen between each epoch')
cmd:option('--maxEpoch', 100, 'maximum number of epochs to run')
cmd:option('--maxTries', 30, 'maximum number of epochs to try to find a better local minima for early-stopping')
cmd:option('--accUpdate', false, 'accumulate gradients inplace')
cmd:option('--verbose', false, 'print verbose messages')
cmd:option('--progress', true, 'print progress bar')
cmd:option('--nThread', 4, 'allocate threads for loading images from disk. Requires threads-ffi.')
cmd:text()
cmd:option('--usingCudnn', true, 'use cudnn instead of cunn')
opt = cmd:parse(arg or {})

opt.trainPath = (opt.trainPath == '') and paths.concat(opt.dataPath, 'ILSVRC2012_img_train') or opt.trainPath
opt.validPath = (opt.validPath == '') and paths.concat(opt.dataPath, 'ILSVRC2012_img_val') or opt.validPath
opt.metaPath = (opt.metaPath == '') and paths.concat(opt.dataPath, 'metadata') or opt.metaPath

if not opt.silent then
  table.print(opt)
end

--[[data]]--
datasource = dp.ImageNet{
   train_path=opt.trainPath, valid_path=opt.validPath, 
   meta_path=opt.metaPath, verbose=opt.verbose
}

--[[preprocessing]]--
ppf = datasource:normalizePPF()

--[[GPU or CPU]]--
if opt.cuda then
  require 'cutorch'
  if usingCudnn then
     require 'cudnn'
  else
     require 'cunn'
  end
  torch.setdefaulttensortype('torch.CudaTensor')
  cutorch.setDevice(opt.useDevice)
end

--[[model]]--
mlp = dp.Sequential()
mlp:add(dp.VGGNet{
      inputSize = 3,
      inputHeight = 224,
      inputWidth = 224,
      usingCudnn = opt.usingCudnn,
    })
local visitor = {
  dp.Momentum{momentum_factor = opt.momentum},
  dp.Learn{
    learning_rate = opt.learningRate, 
    observer = dp.LearningRateSchedule{
         schedule={[1]=1e-2,[19]=5e-3,[30]=1e-3,[44]=5e-4,[53]=1e-4}
    }
  },
  dp.MaxNorm{max_out_norm = opt.maxOutNorm, period=opt.maxNormPeriod}
}

--[[Propagators]]--
train = dp.Optimizer{
  loss = dp.NLL(),
  visitor = visitor,
  feedback = dp.Confusion(),
  sampler = dp.RandomSampler{
      batch_size = opt.batchSize, epoch_size = opt.trainEpochSize, ppf = ppf
   },
  progress = opt.progress
}
valid = dp.Evaluator{
   loss = dp.NLL(),
   feedback = dp.TopCrop{n_top={1,5,10},n_crop=10,center=2},  
   sampler = dp.Sampler{
      batch_size=math.round(opt.batchSize/10),
      ppf=ppf
   }
}
test = dp.Evaluator{
   loss = dp.NLL(),
   feedback = dp.TopCrop{n_top={1,5,10},n_crop=10,center=2},  
   sampler = dp.Sampler{
      batch_size=math.round(opt.batchSize/10),
      ppf=ppf
   }
}

--[[multithreading]]--
if opt.nThread > 0 then
   datasource:multithread(opt.nThread)
   train:sampler():async()
   valid:sampler():async()
   test:sampler():async()
end

--[[Experiment]]--
xp = dp.Experiment{
   model = mlp,
   optimizer = train,
   validator = valid,
   tester = test,
   observer = {
      dp.FileLogger(),
      dp.EarlyStopper{
         save_strategy = dp.SaveModel(),
         start_epoch = 1,
         error_report = {'validator','feedback','topcrop','all',5},
         maximize = true,
         max_epochs = opt.maxTries
      }
   },
   random_seed = os.time(),
   max_epoch = opt.maxEpoch
}

--[[GPU or CPU]]--
if opt.cuda then
  xp:cuda()
end

--[[
if not opt.silent then
  print"dp.Models :"
  print(cnn)
  print"nn.Modules :"
  print(mlp:toModule(datasource:trainSet():sub(1,32)))
end
--]]
xp:verbose(not opt.silent)

xp:run(datasource)