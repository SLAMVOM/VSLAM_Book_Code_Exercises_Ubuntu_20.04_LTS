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
CMAKE_SOURCE_DIR = /home/mt/Miller/VSLAM/project/ch3/examples

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/mt/Miller/VSLAM/project/ch3/examples/build

# Include any dependencies generated for this target.
include CMakeFiles/transformingCoordinate.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/transformingCoordinate.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/transformingCoordinate.dir/flags.make

CMakeFiles/transformingCoordinate.dir/coordinateTransform.cpp.o: CMakeFiles/transformingCoordinate.dir/flags.make
CMakeFiles/transformingCoordinate.dir/coordinateTransform.cpp.o: ../coordinateTransform.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/mt/Miller/VSLAM/project/ch3/examples/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/transformingCoordinate.dir/coordinateTransform.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/transformingCoordinate.dir/coordinateTransform.cpp.o -c /home/mt/Miller/VSLAM/project/ch3/examples/coordinateTransform.cpp

CMakeFiles/transformingCoordinate.dir/coordinateTransform.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/transformingCoordinate.dir/coordinateTransform.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/mt/Miller/VSLAM/project/ch3/examples/coordinateTransform.cpp > CMakeFiles/transformingCoordinate.dir/coordinateTransform.cpp.i

CMakeFiles/transformingCoordinate.dir/coordinateTransform.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/transformingCoordinate.dir/coordinateTransform.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/mt/Miller/VSLAM/project/ch3/examples/coordinateTransform.cpp -o CMakeFiles/transformingCoordinate.dir/coordinateTransform.cpp.s

# Object files for target transformingCoordinate
transformingCoordinate_OBJECTS = \
"CMakeFiles/transformingCoordinate.dir/coordinateTransform.cpp.o"

# External object files for target transformingCoordinate
transformingCoordinate_EXTERNAL_OBJECTS =

transformingCoordinate: CMakeFiles/transformingCoordinate.dir/coordinateTransform.cpp.o
transformingCoordinate: CMakeFiles/transformingCoordinate.dir/build.make
transformingCoordinate: CMakeFiles/transformingCoordinate.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/mt/Miller/VSLAM/project/ch3/examples/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable transformingCoordinate"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/transformingCoordinate.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/transformingCoordinate.dir/build: transformingCoordinate

.PHONY : CMakeFiles/transformingCoordinate.dir/build

CMakeFiles/transformingCoordinate.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/transformingCoordinate.dir/cmake_clean.cmake
.PHONY : CMakeFiles/transformingCoordinate.dir/clean

CMakeFiles/transformingCoordinate.dir/depend:
	cd /home/mt/Miller/VSLAM/project/ch3/examples/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/mt/Miller/VSLAM/project/ch3/examples /home/mt/Miller/VSLAM/project/ch3/examples /home/mt/Miller/VSLAM/project/ch3/examples/build /home/mt/Miller/VSLAM/project/ch3/examples/build /home/mt/Miller/VSLAM/project/ch3/examples/build/CMakeFiles/transformingCoordinate.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/transformingCoordinate.dir/depend

