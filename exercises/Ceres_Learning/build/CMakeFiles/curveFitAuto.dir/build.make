# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/mt/Miller/VSLAM/project/exercises/Ceres_Learning

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/mt/Miller/VSLAM/project/exercises/Ceres_Learning/build

# Include any dependencies generated for this target.
include CMakeFiles/curveFitAuto.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/curveFitAuto.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/curveFitAuto.dir/flags.make

CMakeFiles/curveFitAuto.dir/curveFittingAuto.cpp.o: CMakeFiles/curveFitAuto.dir/flags.make
CMakeFiles/curveFitAuto.dir/curveFittingAuto.cpp.o: ../curveFittingAuto.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/mt/Miller/VSLAM/project/exercises/Ceres_Learning/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/curveFitAuto.dir/curveFittingAuto.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/curveFitAuto.dir/curveFittingAuto.cpp.o -c /home/mt/Miller/VSLAM/project/exercises/Ceres_Learning/curveFittingAuto.cpp

CMakeFiles/curveFitAuto.dir/curveFittingAuto.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/curveFitAuto.dir/curveFittingAuto.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/mt/Miller/VSLAM/project/exercises/Ceres_Learning/curveFittingAuto.cpp > CMakeFiles/curveFitAuto.dir/curveFittingAuto.cpp.i

CMakeFiles/curveFitAuto.dir/curveFittingAuto.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/curveFitAuto.dir/curveFittingAuto.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/mt/Miller/VSLAM/project/exercises/Ceres_Learning/curveFittingAuto.cpp -o CMakeFiles/curveFitAuto.dir/curveFittingAuto.cpp.s

# Object files for target curveFitAuto
curveFitAuto_OBJECTS = \
"CMakeFiles/curveFitAuto.dir/curveFittingAuto.cpp.o"

# External object files for target curveFitAuto
curveFitAuto_EXTERNAL_OBJECTS =

curveFitAuto: CMakeFiles/curveFitAuto.dir/curveFittingAuto.cpp.o
curveFitAuto: CMakeFiles/curveFitAuto.dir/build.make
curveFitAuto: /usr/local/lib/libceres.a
curveFitAuto: /usr/lib/x86_64-linux-gnu/libglog.so
curveFitAuto: /usr/lib/x86_64-linux-gnu/libgflags.so.2.2.2
curveFitAuto: /usr/lib/x86_64-linux-gnu/libspqr.so
curveFitAuto: /usr/lib/x86_64-linux-gnu/libcholmod.so
curveFitAuto: /usr/lib/x86_64-linux-gnu/libamd.so
curveFitAuto: /usr/lib/x86_64-linux-gnu/libcamd.so
curveFitAuto: /usr/lib/x86_64-linux-gnu/libccolamd.so
curveFitAuto: /usr/lib/x86_64-linux-gnu/libcolamd.so
curveFitAuto: /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so
curveFitAuto: /usr/lib/x86_64-linux-gnu/libtbb.so.2
curveFitAuto: /usr/lib/x86_64-linux-gnu/libcxsparse.so
curveFitAuto: /usr/lib/x86_64-linux-gnu/liblapack.so
curveFitAuto: /usr/lib/x86_64-linux-gnu/libf77blas.so
curveFitAuto: /usr/lib/x86_64-linux-gnu/libatlas.so
curveFitAuto: CMakeFiles/curveFitAuto.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/mt/Miller/VSLAM/project/exercises/Ceres_Learning/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable curveFitAuto"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/curveFitAuto.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/curveFitAuto.dir/build: curveFitAuto

.PHONY : CMakeFiles/curveFitAuto.dir/build

CMakeFiles/curveFitAuto.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/curveFitAuto.dir/cmake_clean.cmake
.PHONY : CMakeFiles/curveFitAuto.dir/clean

CMakeFiles/curveFitAuto.dir/depend:
	cd /home/mt/Miller/VSLAM/project/exercises/Ceres_Learning/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/mt/Miller/VSLAM/project/exercises/Ceres_Learning /home/mt/Miller/VSLAM/project/exercises/Ceres_Learning /home/mt/Miller/VSLAM/project/exercises/Ceres_Learning/build /home/mt/Miller/VSLAM/project/exercises/Ceres_Learning/build /home/mt/Miller/VSLAM/project/exercises/Ceres_Learning/build/CMakeFiles/curveFitAuto.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/curveFitAuto.dir/depend

