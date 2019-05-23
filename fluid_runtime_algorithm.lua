-- Copyright 2016 Google Inc, NYU.
-- 
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
-- 
--     http://www.apache.org/licenses/LICENSE-2.0
-- 
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

-- Top level training and validation script for FluidNet.
--
-- Usage:
-- Global options can be set from the command line, ie:
-- >> qlua fluid_net_train.lua -gpu 1 -train_preturb.rotation 20 
--
-- To print a list of options (and their defaults) use:
-- >> qlua fluid_net_train.lua -help

dofile('lib/include.lua')
local cudnn = torch.loadPackageSafe('cudnn')
local cutorch = torch.loadPackageSafe('cutorch')
local paths = require('paths')
local optim = require('optim')
local  mattorch = torch.loadPackageSafe('mattorch')
--local mattorch = require('mattorch')
print(mattorch)
local gnuplot = torch.loadPackageSafe('gnuplot')

-- ****************************** Define Config ********************************
local conf = torch.defaultConf()  -- Table with configuration and model params.
conf = torch.parseArgs(conf)  -- Overwrite conf params from the command line.
torch.makeGlobal('_conf', conf)

--wq: neeed to set batchSize to 1 in default_conf
assert(conf.batchSize == 1, 'The batch size must be one')


-- ****************************** Select the GPU *******************************
cutorch.setDevice(conf.gpu)
print("GPU That will be used:")
print(cutorch.getDeviceProperties(conf.gpu))

-- **************************** Load data from Disk ****************************
local tr = torch.loadSet(conf, 'tr') --Instance of DataBinary
torch.makeGlobal('_tr', tr)
local te = torch.loadSet(conf, 'te') --Instance of DataBinary
torch.makeGlobal('_te', te)

-- ***************************** Create the model ******************************

local mpath
local mconf = {}
local model = {}
local tframes = 0
local ttime =0

--conf.modelFilename = {'dongss2D','tog2D',}
conf.modelFilename = {'default_881','default_3131','default_8881','default_1232D','default_1681','default_331','default_sub22D','default_sub12D','default_lay12D','dongss2D','dongequal2D','default2D','dongsmall2D','dongdefault2D'} 

local netarch = {}
--[[
netarch['tog2D'] = 		   '   7,'.. 										   --valid layer numer
						   '  16,  32,  32,  64,  64,  32,   1,   0,   0, '..  --osize
						   '   5, 	5, 	 5,   5,   1,   1,   3,   0,   0, '..  --ksize
						   '   2, 	1, 	 1,   1,   1,   1,   1,   0,   0, '..  --psize
						   '   1,   1,   1,   1,   1,   1,   2,   0,   0, '..  --upsampling size (1 == no upsampling).
						   '   0,   0,   0,   0,   0,   0,   0,   0,   0  ' --residual connection
--]]

netarch['default_1232D'] = '   4,'.. 										   --valid layer numer
						   '   8,   8,   8,   1,   0,   0,   0,   0,   0, '..  --osize
						   '   3, 	3, 	 3,   1,   0,   0,   0,   0,   0, '..  --ksize
						   '   1, 	1, 	 1,   1,   0,   0,   0,   0,   0, '..  --psize
						   '   1,   1,   1,   1,   0,   0,   0,   0,   0, '..  --upsampling size (1 == no upsampling).
						   '   0,   0,   0,   0,   0,   0,   0,   0,   0  ' --residual connection

						   --wq: if you have a residual connection, the valid conv layer should be added by 1 
netarch['dongss2D'] = 	   '   4,'.. 										   --valid conv layer numer
						   '  16,  16,   1,   0,   0,   0,   0,   0,   0, '..  --osize
						   '   3, 	3, 	 1,   0,   0,   0,   0,   0,   0, '..  --ksize
						   '   1, 	1, 	 1,   0,   0,   0,   0,   0,   0, '..  --psize
						   '   1,   1,   1,   0,   0,   0,   0,   0,   0, '..  --upsampling size (1 == no upsampling).
						   '   0,   1,   0,   0,   0,   0,   0,   0,   0  ' --residual connection

netarch['default2D'] = 	   '   5,'.. 										   --valid layer numer
						   '  16,  16,  16,  16,   1,   0,   0,   0,   0, '..  --osize
						   '   3, 	3, 	 3,   3,   1,   0,   0,   0,   0, '..  --ksize
						   '   1, 	1, 	 1,   1,   1,   0,   0,   0,   0, '..  --psize
						   '   1,   1,   1,   1,   1,   0,   0,   0,   0, '..  --upsampling size (1 == no upsampling).
						   '   0,   0,   0,   0,   0,   0,   0,   0,   0  ' --residual connection

						   --wq: if you have a residual connection, the valid conv layer should be added by 1 
netarch['dongequal2D'] =   '   5,'.. 										   --valid layer numer
						   '  16,  16,  15,   1,   0,   0,   0,   0,   0, '..  --osize
						   '   3, 	3, 	 3,   1,   0,   0,   0,   0,   0, '..  --ksize
						   '   1, 	1, 	 1,   1,   0,   0,   0,   0,   0, '..  --psize
						   '   1,   1,   1,   1,   0,   0,   0,   0,   0, '..  --upsampling size (1 == no upsampling).
						   '   0,   1,   0,   0,   0,   0,   0,   0,   0  ' --residual connection

						   --wq: if you have a residual connection, the valid conv layer should be added by 1 
netarch['dongdefault2D'] = '   8,'.. 										   --valid layer numer
						   '  16,  16,  16,  16,   1,   0,   0,   0,   0, '..  --osize
						   '   3, 	3, 	 3,   3,   1,   0,   0,   0,   0, '..  --ksize
						   '   1, 	1, 	 1,   1,   1,   0,   0,   0,   0, '..  --psize
						   '   1,   1,   1,   1,   1,   0,   0,   0,   0, '..  --upsampling size (1 == no upsampling).
						   '   0,   1,   1,   1,   0,   0,   0,   0,   0  ' --residual connection
--[[
						   --wq: if you have a residual connection, the valid conv layer should be added by 1 						   
netarch['dong2D'] = 	   '   9,'.. 										   --valid layer numer
						   '  16,  32,  32,  64,  64,  32,   1,   0,   0, '..  --osize
						   '   5, 	5, 	 5,   5,   1,   1,   3,   0,   0, '..  --ksize
						   '   2, 	1, 	 1,   1,   1,   1,   1,   0,   0, '..  --psize
						   '   1,   1,   1,   1,   1,   1,   2,   0,   0, '..  --upsampling size (1 == no upsampling).
						   '   0,   0,   1,   0,   1,   0,   0,   0,   0  ' --residual connection
						   
						   --wq: if you have a residual connection, the valid conv layer should be added by 1 						   
--]]
netarch['dongsmall2D'] =   '   6,'.. 										   --valid layer numer
						   '  16,  16,  16,   1,   0,   0,   0,   0,   0, '..  --osize
						   '   3, 	3, 	 3,   3,   0,   0,   0,   0,   0, '..  --ksize
						   '   1, 	1, 	 1,   1,   0,   0,   0,   0,   0, '..  --psize
						   '   1,   1,   1,   1,   0,   0,   0,   0,   0, '..  --upsampling size (1 == no upsampling).
						   '   0,   1,   1,   0,   0,   0,   0,   0,   0  ' --residual connection
netarch['default_lay12D'] =   '   4,'.. 										   --valid layer numer
						   '   16,  16,  16,   1,   0,   0,   0,   0,   0, '..  --osize
						   '   3, 	3, 	 3,   1,   0,   0,   0,   0,   0, '..  --ksize
						   '   1, 	1, 	 1,   1,   0,   0,   0,   0,   0, '..  --psize
						   '   1,   1,   1,   1,   0,   0,   0,   0,   0, '..  --upsampling size (1 == no upsampling).
						   '   0,   0,   0,   0,   0,   0,   0,   0,   0  ' --residual connection
netarch['default_sub12D'] =   '   4,'.. 										   --valid layer numer
						   '   8,  16,  16,   1,   0,   0,   0,   0,   0, '..  --osize
						   '   3, 	3, 	 3,   1,   0,   0,   0,   0,   0, '..  --ksize
						   '   1, 	1, 	 1,   1,   0,   0,   0,   0,   0, '..  --psize
						   '   1,   1,   1,   1,   0,   0,   0,   0,   0, '..  --upsampling size (1 == no upsampling).
						   '   0,   0,   0,   0,   0,   0,   0,   0,   0  ' --residual connection
netarch['default_sub22D'] =   '   4,'.. 										   --valid layer numer
						   '   16,  8,  16,   1,   0,   0,   0,   0,   0, '..  --osize
						   '   3, 	3, 	 3,   1,   0,   0,   0,   0,   0, '..  --ksize
						   '   1, 	1, 	 1,   1,   0,   0,   0,   0,   0, '..  --psize
						   '   1,   1,   1,   1,   0,   0,   0,   0,   0, '..  --upsampling size (1 == no upsampling).
						   '   0,   0,   0,   0,   0,   0,   0,   0,   0  ' --residual connection

netarch['default_8881'] =   '   4,'.. 										   --valid layer numer
						   '  8,  8,  8,   1,   0,   0,   0,   0,   0, '..  --osize
						   '   3, 	3, 	 1,   1,   0,   0,   0,   0,   0, '..  --ksize
						   '   1, 	1, 	 1,   1,   0,   0,   0,   0,   0, '..  --psize
						   '   1,   1,   1,   1,   0,   0,   0,   0,   0, '..  --upsampling size (1 == no upsampling).
						   '   0,   0,   0,   0,   0,   0,   0,   0,   0  ' --residual connection
netarch['default_3131'] =   '   4,'.. 										   --valid layer numer
						   '  8,  8,  8,   1,   0,   0,   0,   0,   0, '..  --osize
						   '   3, 	1, 	 3,   1,   0,   0,   0,   0,   0, '..  --ksize
						   '   1, 	1, 	 1,   1,   0,   0,   0,   0,   0, '..  --psize
						   '   1,   1,   1,   1,   0,   0,   0,   0,   0, '..  --upsampling size (1 == no upsampling).
						   '   0,   0,   0,   0,   0,   0,   0,   0,   0  ' --residual connection
netarch['default_331'] =   '   3,'.. 										   --valid layer numer
						   '  16,  16,  1,   0,   0,   0,   0,   0,   0, '..  --osize
						   '   3, 	3, 	 1,   0,   0,   0,   0,   0,   0, '..  --ksize
						   '   1, 	1, 	 1,   0,   0,   0,   0,   0,   0, '..  --psize
						   '   1,   1,   1,   0,   0,   0,   0,   0,   0, '..  --upsampling size (1 == no upsampling).
						   '   0,   0,   0,   0,   0,   0,   0,   0,   0  ' --residual connection
netarch['default_1681'] =   '   3,'.. 										   --valid layer numer
						   '  16,  8,  1,   0,   0,   0,   0,   0,   0, '..  --osize
						   '   3, 	3, 	 1,   0,   0,   0,   0,   0,   0, '..  --ksize
						   '   1, 	1, 	 1,   0,   0,   0,   0,   0,   0, '..  --psize
						   '   1,   1,   1,   0,   0,   0,   0,   0,   0, '..  --upsampling size (1 == no upsampling).
						   '   0,   0,   0,   0,   0,   0,   0,   0,   0  ' --residual connection
netarch['default_881'] =   '   3,'.. 										   --valid layer numer
						   '  8,  8,  1,   0,   0,   0,   0,   0,   0, '..  --osize
						   '   3, 	3, 	 1,   0,   0,   0,   0,   0,   0, '..  --ksize
						   '   1, 	1, 	 1,   0,   0,   0,   0,   0,   0, '..  --psize
						   '   1,   1,   1,   0,   0,   0,   0,   0,   0, '..  --upsampling size (1 == no upsampling).
						   '   0,   0,   0,   0,   0,   0,   0,   0,   0  ' --residual connection
						   
print(netarch)
count = 0
for k,v in pairs(netarch) do
	--print(k..' : '.. v)
     count = count + 1
end
print('There are '..count..' available model archs') 
print('There are '..#conf.modelFilename..' available model names') 
assert(count == #conf.modelFilename,  'the network architecture should be the same length with models')




local netFlops = {}
netFlops['default_881'] =27.26;

netFlops['default_3131'] = 29.62 ;

netFlops['default_8881'] = 29.62 ;

netFlops['default_1232D'] = 46.4;

netFlops['default_1681'] = 53.48;

netFlops['default_331'] = 91.75;

netFlops['default_sub22D'] = 92.01;

netFlops['default_sub12D'] = 122.68;

netFlops['default_lay12D'] = 167.77;

netFlops['dongss2D'] = 168.03;

netFlops['dongequal2D'] = 239.27;

netFlops['default2D'] = 243.79;

netFlops['dongsmall2D'] = 320.34;

netFlops['dongdefault2D'] = 472.65;

--netFlops['dong2D'] = 1080;

local netPosbi = {} 

-------------***************load MLP model and select top-k models********************-------------
--[[
local upscale_MLP = 1
local Time_user = 0.2391
local Quality_user = 0.0096
local Div_end = 50


local upscale_MLP = 4
local Time_user = 2.9727
local Quality_user = 0.0134


local Quality_user = 0.0094

local upscale_MLP = 2
local Time_user = 1.4483
local Quality_user = 0.01
local upscale_MLP = 6
local Time_user = 3.8184
local Quality_user = 0.0117
--]]

local upscale_MLP = 8
local Time_user = 6.6363
local Quality_user = 0.010


MLP_mpath = '../torch_DivNorm_wq/upscale_'..upscale_MLP..'/MLP_'..upscale_MLP.. '_lastEpoch'
print('Loading model from ' .. MLP_mpath)
MLP_mconf, MLP_model = torch.loadModel(MLP_mpath)

print(MLP_mconf)

--os.exit()

--iput size '128*128'
local Model_posibility = {}-- {'0.324', '0.514', '0.875', '0.154', '0.754', '0.46', '0.547', '0.855', '0.542', '0.782',  '0.347', '0.478', '0.247', '0.886','0.523',}-- wq:MLP predict model success rate
--local Model_flops = {27.26,29.62,29.62,46.4,53.48,91.75,92.01,122.68,167.77,168.03,239.27,243.79,320.34,472.65,1080}


local Model_flops_topk = {}
--
--local Model_posibility_topk = {}


print("")
--local feature_user = {}
---feature_user[1] = Time_user
--feature_user[2] = Quality_user
function generate_feature(str)
	local feature = {} 

feature[1] = Time_user
feature[2] = Quality_user
	--print(#feature)
	--print(str)
	for word in string.gmatch(str, '([^,]+)') do
		word = tonumber(word)
		feature[#feature+1] = word
		--print(word)
	end
	return feature
end
for j =  1, #conf.modelFilename do 
	print("*********** j = "..j..", model: "..conf.modelFilename[j].." *******************")
	str =  netarch[conf.modelFilename[j]]

	local feature = generate_feature( str)
	--print('concatenated feature length is :'..#feature_user)
	--print(type(feature))
	--=torch.Tensor( 1,#feature)
 	--feature_tensor[{ {}, 1}] = 
	feature_tensor = torch.Tensor(feature):reshape(1, #feature)
	--print(feature_tensor)
	--print(feature_tensor:size())
	
	local output = MLP_model:forward(feature_tensor)
	--print(output:size())
	--print(output)
	print(output[1][1]) --only one single number representing the possibily in (0,1)
	Model_posibility[j] = output[1][1] 
end
--os.exit()

function printList(list)
	local str = ""
	for i = 1,#list do
		str = str .. list[i] .. ","
	end
	str = str .. "\n"
	print(str)
end
	

function sort(list, modelFilename, top_k)

local modelFilename_topk = {}
	for i = 1,#list do
		for j = 1,#list - i do
			if list[j + 1] and list[j] < list[j + 1] then
				list[j + 1],list[j] = list[j],list[j + 1]
				modelFilename[j + 1],modelFilename[j] = modelFilename[j],modelFilename[j + 1]
			end
		end
	end

	for i =1, top_k do
		modelFilename_topk[i] = modelFilename[i]
		netPosbi[modelFilename_topk[i]] = list[i]
		--print(list[i])
	end
	return modelFilename_topk, netPosbi
end
printList(conf.modelFilename)
conf.modelFilename, netPosbi = sort(Model_posibility,conf.modelFilename,5)

printList(conf.modelFilename)



-------------*********Run-time algorithm******************---------------

for i=1, #conf.modelFilename do
  conf.modelDirname = conf.modelDir .. '/' .. conf.modelFilename[i] --wq:reload
  local mpath = conf.modelDirname
  if conf.resumeTraining then
    mpath = mpath .. '_lastEpoch'
  end
  print('Loading model from ' .. mpath)
  mconf[i], model[i] = torch.loadModel(mpath)
 -- mconf.simMethod = 'pcg'
  conf.newModel = nil
end

torch.makeGlobal('_mconf', mconf)
torch.makeGlobal('_model', model)
torch.makeGlobal('_netarch', netarch)
torch.makeGlobal('_netPosbi',netPosbi)
torch.makeGlobal('_netFlops',netFlops)
torch.makeGlobal('_netDiv', netDiv)

-- Now do a more detailed analysis of the test and training sets (including
-- long term divergence prediction). This is quite slow.
function tfluids.CalcAndDumpStats(data, dataStr)
  for i= 1, #mconf do 
	  mconf[i].simMethod = mconf[i].simMethod or 'convnet'  -- For legacy models.
	  mconf[i].maxIter = 34  -- Match timing performance of our ConvNet.
	  local oldSimMethod = mconf[i].simMethod
	  if conf.statsSimMethod:len() > 0 then
		-- We might want to collect stats using the jacobi or pcg solvers.
		mconf[i].simMethod = conf.statsSimMethod
	  end
  end
  
  --print('Stats run using simMethod: ' .. mconf[1].simMethod)
  local nSteps = 128  -- Use 128 for paper.
  local stats = torch.calcStats(
      {data = data, conf = _conf, mconf = _mconf, model = _model,
	  netarch = _netarch, nSteps = nSteps, netPosbi = _netPosbi, netFlops = _netFlops, netDiv = _netDiv})
end
--tfluids.CalcAndDumpStats(_te, 'te') --train
tfluids.CalcAndDumpStats(_tr, 'tr') --test




