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
--local mconf, model
--if conf.loadModel then

--[[
   if mconf.modelType == 'tog' then
      -- Small model.
      osize = {16, 32, 32, 64, 64, 32, 1}  -- Conv # output features.
      ksize = {5, 5, 5, 5, 1, 1, 3}  -- Conv filter size.
      psize = {2, 1, 1, 1, 1, 1, 1}  -- pooling decimation size (1: no pooling)
      usize = {1, 1, 1, 1, 1, 1, 2}  -- upsampling size (1 == no upsampling).
      rank = {2, 2, 2, 2, 2, 2, 2}
      gatedConv = {false, false, false, false, false, false, false}
      interFeats = {nil, nil, nil, nil, nil, nil, nil}

      -- Note: upsampling is done WITHIN the conv layer (using
      -- SpatialConvolutionUpsampling or VolumetricConvolutionUpsampling).
      -- Therefore you "can" have both upsampling AND pooling in the same layer,
      -- however this would be silly and so I assert against it (because it's
      -- probably a mistake).
    elseif mconf.modelType == 'dong' then
     -- Small model.
      osize = {16, 32, 32, 64, 64, 32, 1}  -- Conv # output features.
      ksize = {5, 5, 5, 5, 1, 1, 3}  -- Conv filter size.
      psize = {2, 1, 1, 1, 1, 1, 1}  -- pooling decimation size (1: no pooling)
      usize = {1, 1, 1, 1, 1, 1, 2}  -- upsampling size (1 == no upsampling).
      rank = {2, 2, 2, 2, 2, 2, 2}
      gatedConv = {false, false, false, false, false, false, false}
      interFeats = {nil, nil, nil, nil, nil, nil, nil}
     elseif mconf.modelType == 'dongdefault' then
      osize = {16, 16, 16, 16, 1}  -- Conv # output features.
      ksize = {3, 3, 3, 3, 1}  -- Conv filter size.
      psize = {1, 1, 1, 1, 1}  -- pooling decimation size (1: no pooling)
      usize = {1, 1, 1, 1, 1}  -- upsampling size (1 == no upsampling).
      rank = {2, 2, 2, 2, 2}
      gatedConv = {false, false, false, false, false}
      interFeats = {nil, nil, nil, nil, nil}
     elseif mconf.modelType == 'dongsmall' then
      osize = {16, 16, 16, 1}  -- Conv # output features.
      ksize = {3, 3, 3 , 1}  -- Conv filter size.
      psize = {1, 1, 1, 1}  -- pooling decimation size (1: no pooling)
      usize = {1, 1, 1, 1}  -- upsampling size (1 == no upsampling).
      rank = {2, 2, 2, 2}
      gatedConv = {false, false, false, false}
      interFeats = {nil, nil, nil, nil}	  
     elseif mconf.modelType == 'dongss' then
      osize = {16, 16, 1}  -- Conv # output features.
      ksize = {3, 3 , 1}  -- Conv filter size.
      psize = {1, 1, 1}  -- pooling decimation size (1: no pooling)
      usize = {1, 1, 1}  -- upsampling size (1 == no upsampling).
      rank = {2, 2, 2}
      gatedConv = { false, false, false}
      interFeats = {nil, nil, nil}	  	  
     elseif mconf.modelType == 'dongequal' then
      osize = {16, 16, 15, 1}  -- Conv # output features.
      ksize = {3, 3, 3 , 1}  -- Conv filter size.
      psize = {1, 1, 1, 1}  -- pooling decimation size (1: no pooling)
      usize = {1, 1, 1, 1}  -- upsampling size (1 == no upsampling).
      rank = {2, 2, 2, 2}
      gatedConv = {false, false, false, false}
      interFeats = {nil, nil, nil, nil}	  	  
    elseif mconf.modelType == 'default' then
      osize = {16, 16, 16, 16, 1}  -- Conv # output features.
      ksize = {3, 3, 3, 3, 1}  -- Conv filter size.
      psize = {1, 1, 1, 1, 1}  -- pooling decimation size (1: no pooling)
      usize = {1, 1, 1, 1, 1}  -- upsampling size (1 == no upsampling).
      rank = {2, 2, 2, 2, 2}
      gatedConv = {false, false, false, false, false}
      interFeats = {nil, nil, nil, nil, nil}
	elseif mconf.modelType == 'default_lay1' then
      osize = {16, 16, 16, 1}  -- Conv # output features.
      ksize = {3, 3, 3, 1}  -- Conv filter size.
      psize = {1, 1, 1, 1}  -- pooling decimation size (1: no pooling)
      usize = {1, 1, 1, 1}  -- upsampling size (1 == no upsampling).
      rank = {2, 2, 2, 2}
      gatedConv = {false, false, false, false, false}
      interFeats = {nil, nil, nil, nil, nil}
	elseif mconf.modelType == 'default_sub1' then
      osize = {8, 16, 16, 1}  -- Conv # output features.
      ksize = {3, 3, 3, 1}  -- Conv filter size.
      psize = {1, 1, 1, 1}  -- pooling decimation size (1: no pooling)
      usize = {1, 1, 1, 1}  -- upsampling size (1 == no upsampling).
      rank = {2, 2, 2, 2}
      gatedConv = {false, false, false, false, false}
      interFeats = {nil, nil, nil, nil, nil}
	elseif mconf.modelType == 'default_sub2' then
      osize = {16, 8, 16, 1}  -- Conv # output features.
      ksize = {3, 3, 3, 1}  -- Conv filter size.
      psize = {1, 1, 1, 1}  -- pooling decimation size (1: no pooling)
      usize = {1, 1, 1, 1}  -- upsampling size (1 == no upsampling).
      rank = {2, 2, 2, 2}
      gatedConv = {false, false, false, false}
      interFeats = {nil, nil, nil, nil}
	elseif mconf.modelType == 'default_sub3' then
      osize = {16, 16, 8, 1}  -- Conv # output features.
      ksize = {3, 3, 3, 1}  -- Conv filter size.
      psize = {1, 1, 1, 1}  -- pooling decimation size (1: no pooling)
      usize = {1, 1, 1, 1}  -- upsampling size (1 == no upsampling).
      rank = {2, 2, 2, 2}
      gatedConv = {false, false, false, false}
      interFeats = {nil, nil, nil, nil}
	elseif mconf.modelType == 'default_123' then
      osize = {8, 8, 8, 1}  -- Conv # output features.
      ksize = {3, 3, 3, 1}  -- Conv filter size.
      psize = {1, 1, 1, 1}  -- pooling decimation size (1: no pooling)
      usize = {1, 1, 1, 1}  -- upsampling size (1 == no upsampling).
      rank = {2, 2, 2, 2}
      gatedConv = {false, false, false, false}
      interFeats = {nil, nil, nil, nil}
	elseif mconf.modelType == 'default_8881' then
      osize = {8, 8, 8, 1}  -- Conv # output features.
      ksize = {3, 3, 1, 1}  -- Conv filter size.
      psize = {1, 1, 1, 1}  -- pooling decimation size (1: no pooling)
      usize = {1, 1, 1, 1}  -- upsampling size (1 == no upsampling).
      rank = {2, 2, 2, 2}
      gatedConv = {false, false, false, false}
      interFeats = {nil, nil, nil, nil}
	elseif mconf.modelType == 'default_3131' then
      osize = {8, 8, 8, 1}  -- Conv # output features.
      ksize = {3, 1, 3, 1}  -- Conv filter size.
      psize = {1, 1, 1, 1}  -- pooling decimation size (1: no pooling)
      usize = {1, 1, 1, 1}  -- upsampling size (1 == no upsampling).
      rank = {2, 2, 2, 2}
      gatedConv = {false, false, false, false}
      interFeats = {nil, nil, nil, nil}
	elseif mconf.modelType == 'default_6681' then
      osize = {6, 6, 8, 1}  -- Conv # output features.
      ksize = {3, 3, 3, 1}  -- Conv filter size.
      psize = {1, 1, 1, 1}  -- pooling decimation size (1: no pooling)
      usize = {1, 1, 1, 1}  -- upsampling size (1 == no upsampling).
      rank = {2, 2, 2, 2}
      gatedConv = {false, false, false, false}
      interFeats = {nil, nil, nil, nil}
	elseif mconf.modelType == 'default_331' then
      osize = { 16, 16, 1}  -- Conv # output features.
      ksize = { 3, 3, 1}  -- Conv filter size.
      psize = { 1, 1, 1}  -- pooling decimation size (1: no pooling)
      usize = { 1, 1, 1}  -- upsampling size (1 == no upsampling).
      rank = { 2, 2, 2}
      gatedConv = {false, false, false}
      interFeats = {nil, nil, nil}
	elseif mconf.modelType == 'default_1681' then
      osize = { 16, 8, 1}  -- Conv # output features.
      ksize = { 3, 3, 1}  -- Conv filter size.
      psize = { 1, 1, 1}  -- pooling decimation size (1: no pooling)
      usize = { 1, 1, 1}  -- upsampling size (1 == no upsampling).
      rank = { 2, 2, 2}
      gatedConv = {false, false, false}
      interFeats = {nil, nil, nil}
	elseif mconf.modelType == 'default_881' then
      osize = { 8, 8, 1}  -- Conv # output features.
      ksize = { 3, 3, 1}  -- Conv filter size.
      psize = { 1, 1, 1}  -- pooling decimation size (1: no pooling)
      usize = { 1, 1, 1}  -- upsampling size (1 == no upsampling).
      rank = { 2, 2, 2}
      gatedConv = {false, false, false}
      interFeats = {nil, nil, nil}
    elseif mconf.modelType == 'yang' then
      -- From the paper: Data-driven projection method in fluid
      -- simulation, Yang et al., 2016.
      -- "In this paper, the neural network has three hidden layers and six
      -- neurons per hidden layer."      

      -- Note: the Yang paper defines a per-patch network, however this can
      -- be exactly mplemented as a "fully-convolutional" network with 1x1x1
      -- stages for the remaining hidden convolution layers.
      -- They also use only the surrounding neighbor pixels as input context,
      -- with p, divergence and flags as input.
      torch.checkYangSettings(mconf)
      osize = {6, 6, 6, 1}
      ksize = {3, 1, 1, 1}  -- They define a per patch network, whic
      psize = {1, 1, 1, 1}  -- They do not pool or upsample
      usize = {1, 1, 1, 1}  -- They do not pool or upsample
      rank = {2, 2, 2, 2}  -- Always full rank.
      gatedConv = {false, false, false, false}
      interFeats = {nil, nil, nil, nil}
]]
local mpath
local mconf = {}
local model = {}
local tframes = 0
local ttime =0
--conf.modelFilename = {'dongss2D'}
conf.modelFilename = {'tog2D','default_1232D', 'dongss2D', 'default2D', 'dongequal2D', 'dongdefault2D', 'dong2D', 'dongsmall2D', 'default_1681', 'default_3131', 'default_331',  'default_881', 'default_8881', 'default_lay12D', 'default_sub12D','default_sub22D',} 

local netarch = {}

netarch['tog2D'] = 		   '   7,'.. 										   --valid layer numer
						   '  16,  32,  32,  64,  64,  32,   1,   0,   0, '..  --osize
						   '   5, 	5, 	 5,   5,   1,   1,   3,   0,   0, '..  --ksize
						   '   2, 	1, 	 1,   1,   1,   1,   1,   0,   0, '..  --psize
						   '   1,   1,   1,   1,   1,   1,   2,   0,   0, '..  --upsampling size (1 == no upsampling).
						   '   0,   0,   0,   0,   0,   0,   0,   0,   0  ' --residual connection


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

						   --wq: if you have a residual connection, the valid conv layer should be added by 1 						   
netarch['dong2D'] = 	   '   9,'.. 										   --valid layer numer
						   '  16,  32,  32,  64,  64,  32,   1,   0,   0, '..  --osize
						   '   5, 	5, 	 5,   5,   1,   1,   3,   0,   0, '..  --ksize
						   '   2, 	1, 	 1,   1,   1,   1,   1,   0,   0, '..  --psize
						   '   1,   1,   1,   1,   1,   1,   2,   0,   0, '..  --upsampling size (1 == no upsampling).
						   '   0,   0,   1,   0,   1,   0,   0,   0,   0  ' --residual connection
						   
						   --wq: if you have a residual connection, the valid conv layer should be added by 1 						   
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

netarch['default_1232D'] =   '   4,'.. 										   --valid layer numer
						   '  8,  8,  8,   1,   0,   0,   0,   0,   0, '..  --osize
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

--[[print(netarch)
count = 0
for k,v in pairs(netarch) do
	print(k..' : '.. v)
     count = count + 1
end
print('There are '..count..' available model archs') 
print('There are '..#conf.modelFilename..' available model names') 

assert(count == #conf.modelFilename,  'the network architecture should be the same length with models')
 
--]]

for i=1, #conf.modelFilename do
  conf.modelDirname = conf.modelDir .. '/' .. conf.modelFilename[i] --wq:reload
  local mpath = conf.modelDirname
  if conf.resumeTraining then
    mpath = mpath .. '_lastEpoch'
  end
  print('Loading model from ' .. mpath)
  mconf[i], model[i] = torch.loadModel(mpath)
 -- mconf.simMethod = 'pcg'
  if conf.resumeTraining then
    mconf[i].optimState.bestPerf = math.huge  -- We might change loss params.
    -- We might also want to change loss function parameters, so copy over
    -- some mconf parameters that DO NOT pertain to the model architecture
    -- (which is fixed if we're loading a model).
    print('Overwriting some conf.newModel params into loaded mconf:')
    torch.copyTrainingMconfParams(mconf[i], conf.newModel)
  end

  conf.newModel = nil


end


torch.makeGlobal('_mconf', mconf)
torch.makeGlobal('_model', model)
torch.makeGlobal('_netarch', netarch)


-- ********************* Define Criterion (loss) function **********************
 
-- ***************************** Get the parameters ****************************
local parameters = {}
local gradParameters = {}
for i=1, #conf.modelFilename do
print '==> Extracting model parameters'
parameters[i], gradParameters[i] = model[i]:getParameters()
torch.makeGlobal('_parameters', parameters[i])
torch.makeGlobal('_gradParameters', gradParameters[i])
collectgarbage()
end



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
	  netarch = _netarch, nSteps = nSteps})
  --if mconf.simMethod == 'pcg' then
    -- local fn = mconf.simMethod .. '_Stats.bin'
  --else 
    -- local fn = conf.modelDirname .. '_Stats.bin'
  --end
  --torch.save(fn, stats)
  --print('Saved ' .. fn)
  --[[if mattorch ~= nil then
    fn = fn .. '.mat'
    matStats = {}
    matStats['normDiv'] = stats.normDiv
    mattorch.save(fn, matStats)
    print('Saved ' .. fn)
  end
  mconf.simMethod = oldSimMethod]]
end
tfluids.CalcAndDumpStats(_te, 'te')
-- tfluids.CalcAndDumpStats(_tr, 'tr')
