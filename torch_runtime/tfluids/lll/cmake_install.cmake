# Install script for directory: /tmp4/wenbobao_data/FluidNet/torch/tfluids

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/wenbobao/torch/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/tfluids/1-00/lib/libtfluids.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/tfluids/1-00/lib/libtfluids.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/tfluids/1-00/lib/libtfluids.so"
         RPATH "$ORIGIN/../lib:/home/wenbobao/torch/install/lib:/usr/local/cuda-9.0/lib64:/home/wenbobao/anaconda3_new/lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/tfluids/1-00/lib" TYPE MODULE FILES "/tmp4/wenbobao_data/FluidNet/torch/tfluids/build/libtfluids.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/tfluids/1-00/lib/libtfluids.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/tfluids/1-00/lib/libtfluids.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/tfluids/1-00/lib/libtfluids.so"
         OLD_RPATH "/home/wenbobao/torch/install/lib:/usr/local/cuda-9.0/lib64:/home/wenbobao/anaconda3_new/lib:::::::::::::::"
         NEW_RPATH "$ORIGIN/../lib:/home/wenbobao/torch/install/lib:/usr/local/cuda-9.0/lib64:/home/wenbobao/anaconda3_new/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/tfluids/1-00/lib/libtfluids.so")
    endif()
  endif()
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/tfluids/1-00/lua/tfluids" TYPE FILE FILES
    "/tmp4/wenbobao_data/FluidNet/torch/tfluids/init.lua"
    "/tmp4/wenbobao_data/FluidNet/torch/tfluids/flags_to_occupancy.lua"
    "/tmp4/wenbobao_data/FluidNet/torch/tfluids/set_wall_bcs.lua"
    "/tmp4/wenbobao_data/FluidNet/torch/tfluids/test_tfluids.lua"
    "/tmp4/wenbobao_data/FluidNet/torch/tfluids/velocity_divergence.lua"
    "/tmp4/wenbobao_data/FluidNet/torch/tfluids/velocity_update.lua"
    "/tmp4/wenbobao_data/FluidNet/torch/tfluids/volumetric_up_sampling_nearest.lua"
    )
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/tmp4/wenbobao_data/FluidNet/torch/tfluids/build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
