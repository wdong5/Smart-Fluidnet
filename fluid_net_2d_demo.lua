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

-- This is a real-time OpenGL demo for fluid models.

local sys = require('sys')
--print("sys ok")
local gl = require('libLuaGL')
--print("gl ok")
local glu = require('libLuaGlu')
--print("glu ok")
local glut = require('libLuaGlut')
--print("glut ok")
local tfluids = require('tfluids')
--print('tfluids ok')
--require 'image' --Wenqian to save images
--local  mattorch = torch.loadPackageSafe('mattorch')
local mattorch = require('mattorch')
print(mattorch)

dofile("lib/include.lua")
--print("include.lua ok")
local emitter = dofile("lib/emitter.lua")
local downscale = 1
-- ****************************** Define Config ********************************
local conf = torch.defaultConf()
conf.batchSize = 1
conf.loadModel = true
conf = torch.parseArgs(conf)  -- Overwrite conf params from the command line.
assert(conf.batchSize == 1, 'The batch size must be one')
assert(conf.loadModel == true, 'You must load a pre-trained model')

-- ****************************** Select the GPU *******************************
cutorch.setDevice(conf.gpu)


-- **************************** Load data from Disk ****************************
-- We use this in the visualization demo to seed the velocity and flag fields.
local tr = torch.loadSet(conf, 'tr') -- Instance of DataBinary.
local te = torch.loadSet(conf, 'te') -- Instance of DataBinary.

-- ***************************** Create the model ******************************
local mpath
local mconf
local mconf_tog
local mconf_yang
local model={}
local tframes = 0
local ttime = 0
--conf.modelFilename = {'tog2D','default2D','dongdefault2D'}
conf.modelFilename = {'dongdefault2D','default2D','dongsmall2D','dongss2D'}

for i=1, #conf.modelFilename do
	conf.modelDirname = conf.modelDir ..'/'.. conf.modelFilename[i] --wq:reload model path
	mpath = conf.modelDirname
	--[[
	if i==1 then
	  mconf_tog, model[i] = torch.loadModel(mpath)
	elseif i==2 then
	  mconf, model[i] = torch.loadModel(mpath)
	elseif i==3 then
	  mconf_yang, model[i]= torch.loadModel(mpath)
	end
	--]]
	mconf, model[i] = torch.loadModel(mpath)
	model[i]:cuda()
	--print('==> Loaded model from: ' .. conf.modelDirname)
	torch.setDropoutTrain(model[i], false)
	--assert(mconf.is3D == tr.is3D, 'Model data dimension mismatch')
end

--[[conf.modelDirname = conf.modelDir .. '/' .. conf.modelFilename
local mconf, model = torch.loadModel(conf.modelDirname)
model:cuda()
print('==> Loaded model from: ' .. conf.modelDirname)
torch.setDropoutTrain(model, false)
assert(mconf.is3D == tr.is3D, 'Model data dimension mismatch')
--]]

-- *************************** Define some variables ***************************
-- These variables are global at FILE scope only.
local upscale = 1.5

local batchCPU, batchGPU
local mouseDown = {false, false}  -- {left, right}
local mouseDragging = {false, false}
local mouseLastPos = emitter.vec3.create(0, 0, 0)
local mouseInputRadiusInGridCells = 5 * upscale
local mouseInputAmplitude = 20
local mouseInputSphere = emitter.Sphere.create(emitter.vec3.create(0, 0, 0),
                                               mouseInputRadiusInGridCells)

------******************wenqian defind variables***********************************
local frameCounter = 1
local lastFrameCount = 0
local tSimulate = 0
local change = 32
local change2 = 64
local endframe = 513
local saveDir = 'test'
--local statDir = '20'
local simMethod = 'convnet'
local divNet = tfluids.VelocityDivergence():cuda() --wq
local normDiv = torch.zeros(2, endframe):double() --wq
local wq_imageQuality = {}
  wq_imageQuality[0] = 0
local div
local p, U, flags, density 
local cumDivNorm = {}
local switchStep = 20
local model_tag 

local time = sys.clock()
local elapsed = 0
local im = torch.FloatTensor()  -- Temporary render buffer.
local renderVelocity = false
local renderPressure = false
local renderDivergence = false
local renderObstacles = false --true
local texGLIDs = {}
local windowResolutionX = 512
local windowResolutionY = 512
local maxDivergence = 0.0
local filterTexture = true
local densityType = 3
local flipRendering = true  -- Required to have the 0 grid index on the bottom.

-- Colors for tracer paint:
local curColor = 0;
local colors = {{1, 0, 0}, {0, 0, 1}, {0, 1, 0}, {1, 0.2, 1}}
local numColors = #colors


-- ****************************** DATA FUNCTIONS *******************************
function tfluids.loadData()
  local imgList = {torch.random(1, tr:nsamples())}
  print('tr:nsamples:'..tr:nsamples())
  print('Using image: ' .. imgList[1])
  batchCPU = tr:AllocateBatchMemory(conf.batchSize)
  print('conf.batchSize:'..conf.batchSize)
  tr:CreateBatch(batchCPU, torch.IntTensor(imgList), conf.batchSize,
                 conf.dataDir)

  -- Upscale the data. TODO(tompson): Handle upscaling flags properly)
  -- wq:the order(pDiv,densityTarget,densityDiv,pTarget,UTarget,UDiv,flags)
  -- wq:dimession(batchSize, 1, d, h, w)
  for key, value in pairs(batchCPU) do
	print("batchCPU(key):"..key)
    if torch.isTensor(value) then
      assert(value:size(1) == 1 and value:size(3) == 1)  -- Should be bs=1, 2D
	  --print("value:size(1),value:size(2),value:size(3), value:size(4), value:size(5):"..value:size(1)..","..value:size(2)..","..value:size(3)..","..value:size(4)..","..value:size(5))
      value = value:resize(value:size(2), value:size(4), value:size(5))
      local up = value:clone():resize(value:size(1), value:size(2) * upscale,
                                      value:size(3) * upscale)
--print("value:size(1),value:size(2),value:size(3):"..value:size(1)..","..value:size(2)..","..value:size(3))
	  --print("up:size(1),up:size(2),up:size(3):"..up:size(1)..","..up:size(2)..","..up:size(3))
	  for i = 1, value:size(1) do
        image.scale(up[i], value[i])
      end
	  batchCPU[key] = up:resize(conf.batchSize, up:size(1), 1, up:size(2),
                                up:size(3))

    end
  end

  batchCPU.flags = batchCPU.flags:round()--wq:round to the nearest integers.

  batchGPU = torch.deepClone(batchCPU, 'torch.CudaTensor')
  assert(not tr.is3D, 'Density needs updating to 3D')
  -- Pick a new density each time.
  densityType = math.fmod(densityType + 1, 5)
  local im_org
  local im
  if densityType == 0 then
    im = image.fabio()
    im = torch.repeatTensor(im:view(1, im:size(1), im:size(2)), 3, 1, 1)
    im = im:contiguous()
  elseif densityType == 1 then
    im = image.lena()
  elseif densityType == 2 then
    im = image.load('../data/kitteh.jpg', 3, 'float')
  elseif densityType == 3 then
    im = torch.ones(3, tr.ydim, tr.xdim):mul(0.5)
  elseif densityType == 4 then
    im = image.load('../data/kitten.jpg', 3, 'float')
    z = torch.DoubleTensor(3, 4,5)
  else
    error('Bad densityType')
  end

   
  if flipRendering then
    im = image.vflip(im) --wq:flip the 
  end

  local h = batchCPU.flags:size(4)
  local w = batchCPU.flags:size(5)
  local density = image.scale(im, h, w)
  -- All values (U, p, flags, etc) need to have a batch dimension and a unary
  -- depth dimension.
  print("density:size(1),density:size(2),density:size(3):"..density:size(1)..","..density:size(2)..","..density:size(3)) --3
  density = density:resize(1, density:size(1), 1, h, w)

  -- We actually need either a scalar density or a table of densities.
  batchCPU.density = {}
  batchGPU.density = {}
  --batchGPU_target.density = {}
  for i = 1, 3 do
    batchCPU.density[i] = density[{{}, {i}, {}, {}, {}}]:contiguous() --wq:将内存中整块存储模式变成连续分布的模式
    batchGPU.density[i] = batchCPU.density[i]:cuda()
	--batchGPU_target.density[i] = batchCPU.density[i]:cuda()
  end
  --wq：return pDiv,uDiv,flags
  local _, UGPU, flagsGPU = tfluids.getPUFlagsDensityReference(batchGPU)

-- *******************************calculate normDiv ****************************

	--torch.syncBatchToGPU(batchCPU, batchGPU)
	local input = torch.getModelInput(batchGPU)
	local target = torch.getModelTarget(batchGPU)
	local output = model[3]:forward(input)
	local pPred, UPred = torch.parseModelOutput(output)
	local pTarget, UTarget, flags = torch.parseModelTarget(target)
	local pErr = pPred - pTarget
	local UErr = UPred - UTarget

	 
	-- Now record divergence stability vs time.
	-- Restart the sim from the target frame.
	local p, U, flags, density = tfluids.getPUFlagsDensityReference(batchGPU)
	U:copy(batchGPU.UTarget)
	p:copy(batchGPU.pTarget)
	p:copy(pTarget)
	U:copy(UTarget)

	-- Record the divergence of the start frame.
	div = divNet:forward({U, flags})
	normDiv[{1, 1}] = div[1]:norm()
end

local densityToRGBNet = nn.JoinTable(2):cuda()
local divergenceNet = tfluids.VelocityDivergence():cuda()
local flagsToOccupancyNet = tfluids.FlagsToOccupancy():cuda()

-- Remove buoyancy for this demo (can be toggled on later).

mconf.buoyancyScale = 0
mconf.vorticityConfinementAmp = 0
mconf.gravityScale = 0
mconf.dt = 4 / 60
if (simMethod == 'pcg' or simMethod == 'jacobi') then
  mconf.simMethod = simMethod
end
mconf.advectionMethod = 'maccormackOurs'
mconf.maccormackStrength = 0.75


-- ******************************** OpenGL Funcs *******************************
local function convertMousePosToGrid(x, y)
  local xdim = batchCPU.flags:size(5)
  local ydim = batchCPU.flags:size(4)
  local zdim = 1
  local gridX = x / windowResolutionX
  local gridY = y / windowResolutionY
  local gridZ = 1
  gridX = math.max(math.min(math.floor(gridX * xdim) + 1, xdim), 1)
  gridY = math.max(math.min(math.floor(gridY * ydim) + 1, ydim), 1)
  return gridX, gridY, gridZ
end

function tfluids.getGLError()
  local err = gl.GetError()
  while err ~= "NO_ERROR" do
    print(err)
    err = gl.GetError()
  end
end


function tfluids.printDebugInfo()
  print("----------------------------------------------------------")
  print("gl: ")
  for k, v in torch.orderedPairs(gl) do
    print(k, v)
  end

  print("----------------------------------------------------------")
  print("glut: ")
  for k, v in torch.orderedPairs(glut) do
    print(k, v)
  end
  print("----------------------------------------------------------")
end

function tfluids.keyboardFunc(key, x, y)
  if key == 27 then  -- ESC key.
    os.exit(0)
  elseif key == 118 then  -- 'v'
    renderVelocity = not renderVelocity
  elseif key == 112 then  -- 'p'
    renderPressure = not renderPressure
    if renderPressure then
      renderDivergence = false
    end
  elseif key == 100 then  -- 'd'
    renderDivergence = not renderDivergence
    if renderDivergence then
      renderPressure = false
    end
  elseif key == 114 then  -- 'r'
    print("Re-Loading Data!")
    tfluids.loadData()
  elseif key == 46 then  -- '.'
    mconf.vorticityConfinementAmp = (mconf.vorticityConfinementAmp + 0.5)
    print("mconf.vorticityConfinementAmp = " .. mconf.vorticityConfinementAmp)
  elseif key == 44 then -- ','
    mconf.vorticityConfinementAmp =
        math.max(mconf.vorticityConfinementAmp - 0.5, 0)
    print("mconf.vorticityConfinementAmp = " .. mconf.vorticityConfinementAmp)
  elseif key == 97 then  -- 'a'
    local methods = {'maccormackOurs', 'maccormack', 'euler', 'eulerOurs'}
    for i = 1, #methods do
      if methods[i] == mconf.advectionMethod then
        if i == #methods then
          mconf.advectionMethod = methods[1]
        else
          mconf.advectionMethod = methods[i + 1]
        end
        break
      end
    end
    print('Using Advection method: ' .. mconf.advectionMethod)
  elseif key == 43 then  -- '+'
    mconf.dt = mconf.dt * 1.25
    print('mconf.dt = ' .. mconf.dt)
  elseif key == 45 then  -- '-'
    mconf.dt = mconf.dt / 1.25
    print('mconf.dt = ' .. mconf.dt)
  elseif key == 103 then  -- 'g'
    renderObstacles = not renderObstacles
  elseif key == 98 then  -- 'b'
    if batchGPU.UBC ~= nil then
      tfluids.removeBCs(batchGPU)
      print('Plume BCs OFF')
    else
      curColor = math.mod(curColor + 1, numColors)
      local densityVal = colors[curColor]
      local uScale = 10
      local rad = 0.05  -- Fraction of xdim
      tfluids.createPlumeBCs(batchGPU, densityVal, uScale, rad)
      print('Plume BCs ON')
    end
  elseif key == 109 then  -- 'm'
    if mconf.gravityScale <= 0 then
      mconf.gravityScale = 1
      print('gravity ON')
    else
      mconf.gravityScale = 0
      print('gravity OFF')
    end
  elseif key == 110 then  -- 'n'
    if mconf.buoyancyScale == 0 then
      mconf.buoyancyScale = 1
      print('buoyancy ON')
    else
      mconf.buoyancyScale = 0
      print('buoyancy OFF')
    end
  elseif key == 115 then -- 's'
    if mconf.simMethod == 'convnet' then
      mconf.simMethod = 'jacobi'
    elseif mconf.simMethod == 'jacobi' then
      mconf.simMethod = 'pcg'
    elseif mconf.simMethod == 'pcg' then
      mconf.simMethod = 'convnet'
    end
    print('mconf.simMethod = ' .. mconf.simMethod)
  end
end
-- LuaGL needs keyboardFunc to be global.
torch.makeGlobal('keyboardFunc', tfluids.keyboardFunc)
-- Also print the keyboard bindings.
print('Press:')
print('  ESC exit')
print('\n  RENDER SETTINGS:')
print('  "v" render velocity ON/OFF')
print('  "p" render pressure ON/OFF')
print('  "d" render divergence ON/OFF')
print('  "r" reload the data')
print('  "g" render obstacles ON/OFF')
print('\n  MODEL SETTINGS:')
print('  "." / "," increase or decrease vorticity confinement')
print('  "a" cycle first / second order advection methods')
print('  "+" / "-" increase or decrease timestep')
print('  "b" toggle "plume" boundary condition ON/OFF')
print('  "n" toggle buoyancy ON/OFF')
print('  "m" toggle gravity ON/OFF')
print('  "s" toggle simulation method (convnet / jacobi / pcg)')
print('')

function tfluids.drawFullscreenQuad(blend, color, flip)
  if flip == nil then
    flip = false
  end
  gl.Enable("TEXTURE_2D")

  if blend == nil then 
   blend = false
  end
  if color == nil then
    color = {1, 1, 1, 1}
  end

  local blendSrc, blendDst
  if blend then
    blendSrc = gl.GetConst('BLEND_SRC')
    blendDst = gl.GetConst('BLEND_DST')
    gl.Enable("BLEND")
    -- Normal compositing when texture is black and white.
    gl.BlendFunc("SRC_COLOR", "ONE_MINUS_SRC_COLOR")
  else
    gl.Disable("BLEND")
  end
  gl.Color(color)
  gl.Begin('QUADS')              -- Draw A Quad
  local y1, y2
  if flip then
    y1 = 0
    y2 = 1
  else
    y1 = 1
    y2 = 0
  end
  gl.TexCoord(0, y1)
  gl.Vertex(0, 1, 0)         -- Top Left

  gl.TexCoord(1, y1)
  gl.Vertex(1, 1, 0)         -- Top Right

  gl.TexCoord(1, y2)
  gl.Vertex(1, 0, 0)         -- Bottom Right

  gl.TexCoord(0, y2)
  gl.Vertex(0, 0, 0)         -- Bottom Left
  gl.End()
  tfluids.getGLError()

  if blend then
    gl.Disable("BLEND")
    gl.BlendFunc(blendSrc, blendDst)
    gl.Color({1, 1, 1, 1})
 end

  gl.Disable("TEXTURE_2D")
end

function dong_mapping(x2, x1)
	--print('x1 '..x1)
	--print('x2 '..x2)
	local y
	
	if x1 == nil then 
		y = 2.14e-5*x2^1.198 
	else
		y = 2.14e-5*x2^1.198 - 2.14e-5*x1^1.198
	end
	return y
end

function default_mapping(x2, x1)
	--print('x1 '..x1)
	--print('x2 '..x2)
	local y
	
	if x1 == nil then 
		y = 1.002e-05*x2^1.27
	else
		y = 1.002e-05*x2^1.27 - 1.002e-05*x1^1.27
	end
	return y
end

function dongsmall_mapping(x2, x1)
	
	--print('x1 '..x1)
	--print('x2 '..x2)
	local y
	
	if x1 == nil then 
		y = 3.501e-05*x2^1.023 
	else
		y = 3.501e-05*x2^1.023 - 3.501e-05*x1^1.023
	end
	return y
end

function dongss_mapping(x2, x1)
	
	--print('x1 '..x1)
	--print('x2 '..x2)
	local y
	
	if x1 == nil then 
		y = 7.048e-06*x2^1.296 
	else
		y = 7.048e-06*x2^1.296 - 7.048e-06*x1^1.296
	end
	return y
end

function tfluids.displayFunc()
    print("\n\n\n")
    gl.Clear("COLOR_BUFFER_BIT")

  gl.Enable("TEXTURE_2D")

  print('simulation method '..mconf.simMethod)
  do
    print("tframes is "..(tframes+1))
    -- Update data to next time step.
    local t0 = sys.clock()
    sys.tic()  --wq: high precision clock (us precision)
	if (tframes+1) % (switchStep*4) < switchStep+1 then
	  tfluids.simulate(conf, mconf, batchGPU, model[1])
	  model_tag = 'dong2D'
	elseif (tframes+1) % (switchStep*4)  < switchStep * 2+1 then
	  tfluids.simulate(conf, mconf, batchGPU, model[2])
	  model_tag = 'default2D'
	elseif (tframes+1) % (switchStep*4)  < switchStep * 3+1 then 
	  tfluids.simulate(conf, mconf, batchGPU, model[3])
	  model_tag = 'dongsmall2D'
	else 
	  tfluids.simulate(conf, mconf, batchGPU, model[4])
	  model_tag = 'dongss2D'
    end
	p, U, flags, density =
				tfluids.getPUFlagsDensityReference(batchGPU)
	div = divNet:forward({U, flags})
	normDiv[{1,tframes+1}] = div[1]:norm()
	cutorch.synchronize()
	--print('line 517') 
	if tframes+1 > 1 then 
	  cumDivNorm[tframes+1] = cumDivNorm[tframes] + normDiv[{1,tframes+1}]
	else
	  cumDivNorm[tframes+1] = normDiv[{1,tframes+1}]
	end
	print('normDiv, cumDivNorm: '..normDiv[{1, tframes+1}]..','..cumDivNorm[tframes+1])

	
	print("model tag is : "..model_tag)
	if (tframes+1) % switchStep == 0 then
	print("enter the if")
	  if model_tag == 'dong2D' then
	    wq_imageQuality[tframes+1] = wq_imageQuality[tframes+1 - switchStep] +
		dong_mapping(cumDivNorm[tframes+1], cumDivNorm[tframes+1 - switchStep])
	  elseif model_tag == 'default2D' then
	    wq_imageQuality[tframes+1] = wq_imageQuality[tframes+1- switchStep ] +
		default_mapping(cumDivNorm[tframes+1], cumDivNorm[tframes+1 - switchStep])
	  elseif model_tag == 'dongsmall2D' then
	    wq_imageQuality[tframes+1] = wq_imageQuality[tframes+1 - switchStep] +
		dongsmall_mapping(cumDivNorm[tframes+1], cumDivNorm[tframes+1 - switchStep])
	  elseif model_tag == 'dongss2D' then 
	    wq_imageQuality[tframes+1] = wq_imageQuality[tframes+1 - switchStep] +
		dongss_mapping(cumDivNorm[tframes+1], cumDivNorm[tframes+1 - switchStep])
	  end 
	  print("wq_imageQuality: "..wq_imageQuality[tframes+1])
	  
	if tframes+1 == switchStep then
		fd = io.open(saveDir..'/predict_statistics_'..switchStep..'.txt', 'w')
		--fd = io.open('pcg_to_convnet/statistics_'..change..'.txt', 'w')
	else
		fd = io.open(saveDir..'/predict_statistics_'..switchStep..'.txt', 'a+')
		--fd = io.open('pcg_to_convnet/statistics_'..change..'.txt', 'a+')
	end
	fd:write((tframes+1)..', '..wq_imageQuality[tframes+1]..'\n')
	fd:close()
	
	end

	  

      --[[if tframes %2 ==0  then
		mconf.simMethod = 'convnet'
		--tfluids.simulate(conf, mconf, batchGPU, model[2])
		--print("using method:"..modelFilename[2])
	  else 
		mconf.simMethod = 'pcg'
		--tfluids.simulate(conf, mconf, batchGPU, model[3])
		--print("using method:"..modelFilename[3])
      end
	tfluids.simulate(conf, mconf, batchGPU, model[2])--]]
    local simulate_time = sys.toc()
    print('current frame simulate Time: ' .. 1000 * simulate_time .. ' ms ')
    -- The simulate output is left on the GPU.
    local t1 = sys.clock()
	tframes = tframes + 1;
	ttime = ttime + (t1 - t0)
    tSimulate = tSimulate + (t1 - t0)
	print("frame "..tframes..", total time "..ttime .. ",    Avg Simulate time is:"..(ttime/tframes))


	--wenqian: calculate the normDiv here
--[[
	if tframes+1 < endframe+1 then --wenqian: tframes counts from zero instead of 1.
		local p, U, flags, density =
				tfluids.getPUFlagsDensityReference(batchGPU)
		local div = divNet:forward({U, flags})
		--print(div)
		--print(type(div))
		normDiv[{1,tframes+1}] = div[1]:norm()
		cumDivNorm = cumDivNorm + normDiv[{1,tframes+1}]
		
		print('normDiv, cumDivNorm:'..normDiv[{1, tframes}]..','..cumDivNorm)
		
		if tframes+1 == endframe then
		  local fn = saveDir..'/'..statDir..'_Stats.bin'
		  torch.save(fn, normDiv)
			if mattorch ~= nil then
			  local matStats = {}
			  matStats['normDiv'] = normDiv --This is correct 
			  mattorch.save(fn..'.mat', matStats)
			  print('Saved ' ..fn..'.mat')
			end
		end
		cutorch.synchronize()	
	end 
	--]]
	
    -- Calculate some frame stats.
    frameCounter = frameCounter + 1
    local t = sys.clock()
    if t - time > 3.0 then
      elapsed = t - time
      time = t
      local frames = frameCounter - lastFrameCount
      lastFrameCount = frameCounter
      local ms = string.format('%3.0f ms total',
                               (elapsed / frames) * 1000)
      local msSim = string.format('%3.0f ms tfluids.simulate() only',
                                  (tSimulate / frames) * 1000)
      local fps = string.format('FPS: %3.3f', (frames / elapsed))
	  print("elapsed,  frames, tSimulate: "..elapsed.."      ,   "..frames.."   ,   "..tSimulate)
      print(fps .. ' / ' .. ms .. ' / ' .. msSim)
      tSimulate = 0.0
    end

	-- So that our profiling is correct we need to flush the GPU buffer, however
    -- this could be at the cost of total render time.

	
    if math.fmod(frameCounter, endframe+2) == 0 then
      -- Don't collect too often.
      collectgarbage()
    end
  end

  -- We also want to visualize and plot divergence.
  local maxDivergence = 0
  local divergenceGPU
  do
    local _, UGPU, flagsGPU = tfluids.getPUFlagsDensityReference(batchGPU)
    divergenceGPU = divergenceNet:forward({UGPU, flagsGPU})
    maxDivergence = divergenceGPU:max()
  end

  if mouseDown[2] then
    -- Splat down some paint.
    local depth = 1
    local y = mouseLastPos.y
    if flipRendering then
      y = windowResolutionY - y + 1
    end
    local gridX, gridY = convertMousePosToGrid(mouseLastPos.x, y)
    local _, _, _, densityGPU = tfluids.getPUFlagsDensityReference(batchGPU)
    -- TODO(tompson): This is ugly and slow.
    for i = 1, 3 do
      densityGPU[i][{1, 1, depth, gridY, gridX}] = colors[curColor + 1][i]
    end
  end

  -- Visualize the scalar background.
  assert(not tr.is3D, 'Only 2D visualization is supported')
  local function VisualizeScalarTensor(tensor, rescale, filter, file_name_prefix)
    im:resize(unpack(tensor:size():totable()))
    im:copy(tensor)  -- Copy to temporary buffer.
    if rescale then
      local normVal = math.max(-im:min(), im:max())
      im:add(-normVal)  -- Normalize to [-1, 1] (symmetrically)
      im:div(normVal)
      im:mul(0.5):add(1):clamp(0, 1)  -- Normalize to [0 to 1].
    end
    im:clamp(0, 1)
    tfluids.loadTensorTexture(im, texGLIDs[1], filter)
    --image.save(file_name_prefix..frameCounter..'.png', im) --wenqian

	end

    
  if renderPressure then
    local pGPU = tfluids.getPUFlagsDensityReference(batchGPU)
    VisualizeScalarTensor(pGPU:squeeze(), true, filterTexture,'pressure_')
  elseif renderDivergence then
    -- Calculated above.
    VisualizeScalarTensor(divergenceGPU:squeeze(), true, filterTexture,'divergence_')
  else
    local _, _, _, densityGPU = tfluids.getPUFlagsDensityReference(batchGPU)
    densityGPU = densityToRGBNet:forward(densityGPU)
    VisualizeScalarTensor(densityGPU:squeeze(), false, filterTexture,'density_')
  end
  tfluids.drawFullscreenQuad(nil, nil, flipRendering)

  if renderObstacles then
    local _, _, flagsGPU = tfluids.getPUFlagsDensityReference(batchGPU)
    local occupancy = flagsToOccupancyNet:forward(flagsGPU)
    VisualizeScalarTensor(occupancy:squeeze(), false, false,'occupancy_')
    tfluids.drawFullscreenQuad(true, {1, 1, 1, 1}, flipRendering)
  end

  -- Render velocity arrows on top.
  if renderVelocity then
    local _, UGPU = tfluids.getPUFlagsDensityReference(batchGPU)
    local _, UCPU = tfluids.getPUFlagsDensityReference(batchCPU)
    UCPU:copy(UGPU)  -- GPU --> CPU copy.
    tfluids.drawVelocityField(UCPU, flipRendering)
  end

  tfluids.getGLError()

  -- At this point we're done with signed divergence, so it's OK to take in-
  -- place abs.
  --local str = string.format('MaxDivergence modelType: %3.4f, %s', maxDivergence,mconf.simMethod)
  --tfluids.drawString(5, windowResolutionY - 15, str)
  tfluids.getGLError()
  glut.SwapBuffers()
  glut.PostRedisplay()

	gl.ReadBuffer("FRONT")

    --local vp
    local width = windowResolutionX
    local height = windowResolutionY
    local x = 0
    local y = 0

    --print('begin ReadPixels')
    vp = gl.ReadPixels(x,y,width,height,"RGB")
	--print(type(vp))
	
	--print('die or not')
	local my_target = {}
    my_im = torch.FloatTensor(3, height,width)
    for c=1,3 do
        for i = 1,height do
            for j = 1, width do
                     --becuase the image is up side down, i need to use 512-i instead of i
                my_im[c][windowResolutionX+1-i][j] = vp[(i-1)* width * 3+ (j-1)* 3  + c] 
            end                    
        end
    end
    print("frame Counter is "..frameCounter)
    if frameCounter < endframe+1 then 
		my_target[frameCounter] = my_im
		--[[
		if mconf.simMethod=='pcg' then
			--image.save('pcg_images/target_'..frameCounter..'.png', my_im) --wenqian
			diff = 0.0
		elseif mconf.simMethod =='jacobi' then
			image.save('jacobi_images/target_'..frameCounter..'.png', my_im) --wenqian
			pcg_image = image.load('pcg_images/target_'..frameCounter..'.png', 3, 'float') --wenqian
			diff = pcg_image - my_im
			diff = diff * diff
			diff = torch.mean(diff)
			diff = torch.sqrt(diff)
			print(diff)
		elseif mconf.simMethod == 'convnet' then	
			--]]
			image.save(saveDir..'/target_'..frameCounter..'.png', my_im)
			--image.save('pcg_to_convnet/target_'..frameCounter..'.png', my_im) --wenqian
			print('load pcg image target_'..frameCounter..'.png')
			pcg_image = image.load('pcg_images/target_'..frameCounter..'.png', 3, 'float') --wenqian
			--print('load sucess')
			pcg_image:csub(my_im) 
			--print('subtract sucess') 
			pcg_image:abs()
			--print('abs sucess')
			diff = torch.mean(pcg_image)
			--print('mean sucess') 
			print("diff compared with pcg: "..diff)
			

			if frameCounter == 2 then
				fd = io.open(saveDir..'/statistics_'..switchStep..'.txt', 'w')
				--fd = io.open('pcg_to_convnet/statistics_'..change..'.txt', 'w')
			else
				fd = io.open(saveDir..'/statistics_'..switchStep..'.txt', 'a+')
				--fd = io.open('pcg_to_convnet/statistics_'..change..'.txt', 'a+')
			end
			fd:write(frameCounter..', '..diff..'\n')
			fd:close()
	elseif frameCounter > endframe +2 then 
		os.exit(0)
		
    end
    	
	print('***************************')

end
-- LuaGL needs displayFunc to be global.
torch.makeGlobal('displayFunc', tfluids.displayFunc)

function tfluids.reshapeFunc(width, height)
  windowResolutionX = width
  windowResolutionY = height
  gl.Viewport(0, 0, windowResolutionX, windowResolutionY)
end

-- LuaGL needs keyboardFunc to be global.
torch.makeGlobal('reshapeFunc', tfluids.reshapeFunc)

function tfluids.mouseButtonFunc(button, state, x, y)
  -- Note: x and y are in pixels.
  if button == 0 then
    -- Button 0 is left mouse.
    if state == 0 then
      -- State 0 is down, 1 is up.
      mouseDown[1] = true
    else
      mouseDown[1] = false
    end
    mouseDragging[1] = mouseDown[1]
    mouseLastPos.x = x
    mouseLastPos.y = y
  elseif button == 2 then
    -- Button 2 is right mouse.
    if state == 0 then
      mouseDown[2] = true
    else
      mouseDown[2] = false
      curColor = math.mod(curColor + 1, numColors)
    end
    mouseDragging[2] = mouseDown[2]
    mouseLastPos.x = x
    mouseLastPos.y = y
  end
end

-- LuaGL needs keyboardFunc to be global.
torch.makeGlobal('mouseButtonFunc', tfluids.mouseButtonFunc)

function tfluids.mouseMotionFunc(x, y)
  -- x and y are in pixels
  if mouseDown[1] then
    local velX = x - mouseLastPos.x
    local velY = y - mouseLastPos.y
    tfluids.addMouseVelocityInput(x, y, velX, velY, false)

    mouseLastPos.x = x
    mouseLastPos.y = y
  elseif mouseDown[2] then
    local velX = x - mouseLastPos.x
    local velY = y - mouseLastPos.y
    tfluids.addMouseVelocityInput(x, y, velX, velY, true)
    mouseLastPos.x = x
    mouseLastPos.y = y
  end
end

-- LuaGL needs keyboardFunc to be global.
torch.makeGlobal('mouseMotionFunc', tfluids.mouseMotionFunc)

function tfluids.mousePassiveMotionFunc(x, y)
  -- x and y are in pixels
end

-- LuaGL needs keyboardFunc to be global.
torch.makeGlobal('mousePassiveMotionFunc', tfluids.mousePassiveMotionFunc)

function tfluids.addMouseVelocityInput(x, y, velX, velY, rightButton)
  if flipRendering then
    y = windowResolutionY - y + 1
    velY = velY * -1
  end

  -- translate x,y from pixel space to grid cell indices
  -- assuming grid is from [0, 1].
  local _, UGPU = tfluids.getPUFlagsDensityReference(batchGPU)
  local _, UCPU = tfluids.getPUFlagsDensityReference(batchCPU)
  -- TODO(tompson): This is slow.  We sync from GPU --> CPU --> GPU just to
  -- fill in a few values. We should fill in the amplitude values to a CPU
  -- buffer (a small one). Sync this to the GPU then apply the accumulation on
  -- the GPU.
  UCPU:copy(UGPU)  -- GPU --> CPU sync.
  assert(UCPU:dim() == 5)
  local depth = 1
  local dims = emitter.vec3.create(UCPU:size(5), UCPU:size(4), UCPU:size(3))
  local gridX, gridY = convertMousePosToGrid(x, y)
  mouseInputSphere.center:set(gridX, gridY, 0)
  local pp = emitter.Vec3Utils.clone(mouseInputSphere.center)
  local r = math.floor(mouseInputSphere.radius)
  for xx = -r, r do
    for yy = -r, r do
      pp.x = gridX + xx
      pp.y = gridY + yy
      if pp.x > 0 and pp.x <= dims.x and pp.y > 0 and pp.y <= dims.y then
        local t = mouseInputAmplitude *
            emitter.MathUtils.sphereForceFalloff(mouseInputSphere, pp)
        UCPU[{1, {1}, {depth}, {pp.y}, {pp.x}}]:add(t * mconf.dt * velX)
        UCPU[{1, {2}, {depth}, {pp.y}, {pp.x}}]:add(t * mconf.dt * velY)
      end
    end
  end
  UGPU:copy(UCPU)  -- CPU --> GPU sync.
end

function tfluids.drawString(pixelX, pixelY, string)
  assert(pixelX >= 0 and pixelX < windowResolutionX)
  assert(pixelY >= 0 and pixelY < windowResolutionY)

  gl.Disable("BLEND")
  gl.MatrixMode("PROJECTION")
  gl.PushMatrix()
  gl.LoadIdentity()
  gl.Ortho(0, windowResolutionX, 0, windowResolutionY, -1.0, 1.0)
  gl.MatrixMode("MODELVIEW")
  gl.PushMatrix()
  gl.LoadIdentity()
  gl.PushAttrib("DEPTH_TEST")
  gl.Disable("DEPTH_TEST")
  gl.Color({0, 0, 0.66, 0})
  gl.RasterPos({pixelX, pixelY})
  glut.BitmapCharacter(string)
  gl.PopAttrib()
  gl.MatrixMode("PROJECTION")
  gl.PopMatrix()
  gl.MatrixMode("MODELVIEW")
  gl.PopMatrix()
  gl.Enable("BLEND")
end

function tfluids.initgl(output)
  gl.MatrixMode("PROJECTION")
  gl.LoadIdentity()
  tfluids.getGLError()

  -- This makes the screen go from [-1,1] in each dim.
  gl.Ortho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
  gl.Viewport(0, 0, windowResolutionX, windowResolutionY)
  gl.ClearColor(0.0, 0.0, 0.0, 0.0)
  gl.Clear("COLOR_BUFFER_BIT")
  gl.Disable("DEPTH_TEST")

  gl.MatrixMode("MODELVIEW")
  gl.LoadIdentity()

  texGLIDs[#texGLIDs + 1] = gl.GenTextures(1)[1]
  local _, _, _, densityGPU = tfluids.getPUFlagsDensityReference(batchGPU)
  densityGPU = densityToRGBNet:forward(densityGPU)
  tfluids.loadTensorTexture(densityGPU[{1, 1, 1}]:float():clamp(0, 1):squeeze(), --tfluids/init.lua
                            texGLIDs[#texGLIDs], filterTexture)

  tfluids.getGLError()
end

function tfluids.startOpenGL(output)
  glut.InitWindowSize(windowResolutionX, windowResolutionY)
  glut.Init()
  glut.InitDisplayMode("RGB,DOUBLE")
  local window = glut.CreateWindow("example")

  glut.MouseFunc("mouseButtonFunc")
  glut.MotionFunc("mouseMotionFunc")
  glut.PassiveMotionFunc("mousePassiveMotionFunc")
  glut.DisplayFunc("displayFunc")
  glut.KeyboardFunc("keyboardFunc")
  glut.ReshapeFunc("reshapeFunc")
  glut.PostRedisplay()
  tfluids.loadData()


  tfluids.initgl(output)
  print("OpenGL Version: " .. gl.GetString("VERSION"))

  -- Do a simulate step JUST before running the loop. Once we enter the GL loop
  -- any errors will not have a stack trace, so simulating one loop here helps
  -- with debugging.
  tfluids.simulate(conf, mconf, batchGPU, model[1])

  glut.MainLoop()
end

-- ******************************** ENTER LOOP *********************************
tfluids.startOpenGL()
