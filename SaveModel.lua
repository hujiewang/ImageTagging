require 'dp'
local SaveModel, parent = torch.class("dp.SaveModel","dp.SaveToFile")
SaveModel.isSaveModel = true

function SaveModel:__init(config)
  parent.__init(self)
end

function SaveModel:save(subject, current_error)
  assert(subject, "SaveModel: subject is nil")
  local subject_path = subject:id():toPath()
  local save_dir = paths.concat(self._save_dir, subject_path)
  local model_dir = paths.concat(save_dir, 'model')
  --creates directories if required
  os.execute('mkdir -p ' .. model_dir)
  model_name = model_name or 'model.dat'
  local filename = paths.concat(model_dir, 'model.dat')
  dp.vprint(self._verbose, 'SaveModel: saving to '.. filename..' with validation error of '..current_error)
  torch.save(filename, {validation_error = current_error, model = subject._model})
end

--[[
function SaveModel:doneExperiment(report)
    dp.vprint(self._verbose, 'Saving the model of the last epoch...')
    self:save(self._subject,0,"final_model.dat")
end
--]]
