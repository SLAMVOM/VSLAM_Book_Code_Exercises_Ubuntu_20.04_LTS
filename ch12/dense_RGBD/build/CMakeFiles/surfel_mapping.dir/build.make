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
CMAKE_SOURCE_DIR = /home/mt/Miller/VSLAM/project/ch12/dense_RGBD

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/mt/Miller/VSLAM/project/ch12/dense_RGBD/build

# Include any dependencies generated for this target.
include CMakeFiles/surfel_mapping.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/surfel_mapping.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/surfel_mapping.dir/flags.make

CMakeFiles/surfel_mapping.dir/surfel_mapping.cpp.o: CMakeFiles/surfel_mapping.dir/flags.make
CMakeFiles/surfel_mapping.dir/surfel_mapping.cpp.o: ../surfel_mapping.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/mt/Miller/VSLAM/project/ch12/dense_RGBD/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/surfel_mapping.dir/surfel_mapping.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/surfel_mapping.dir/surfel_mapping.cpp.o -c /home/mt/Miller/VSLAM/project/ch12/dense_RGBD/surfel_mapping.cpp

CMakeFiles/surfel_mapping.dir/surfel_mapping.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/surfel_mapping.dir/surfel_mapping.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/mt/Miller/VSLAM/project/ch12/dense_RGBD/surfel_mapping.cpp > CMakeFiles/surfel_mapping.dir/surfel_mapping.cpp.i

CMakeFiles/surfel_mapping.dir/surfel_mapping.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/surfel_mapping.dir/surfel_mapping.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/mt/Miller/VSLAM/project/ch12/dense_RGBD/surfel_mapping.cpp -o CMakeFiles/surfel_mapping.dir/surfel_mapping.cpp.s

# Object files for target surfel_mapping
surfel_mapping_OBJECTS = \
"CMakeFiles/surfel_mapping.dir/surfel_mapping.cpp.o"

# External object files for target surfel_mapping
surfel_mapping_EXTERNAL_OBJECTS =

surfel_mapping: CMakeFiles/surfel_mapping.dir/surfel_mapping.cpp.o
surfel_mapping: CMakeFiles/surfel_mapping.dir/build.make
surfel_mapping: /usr/local/lib/libopencv_gapi.so.4.5.5
surfel_mapping: /usr/local/lib/libopencv_stitching.so.4.5.5
surfel_mapping: /usr/local/lib/libopencv_alphamat.so.4.5.5
surfel_mapping: /usr/local/lib/libopencv_aruco.so.4.5.5
surfel_mapping: /usr/local/lib/libopencv_barcode.so.4.5.5
surfel_mapping: /usr/local/lib/libopencv_bgsegm.so.4.5.5
surfel_mapping: /usr/local/lib/libopencv_bioinspired.so.4.5.5
surfel_mapping: /usr/local/lib/libopencv_ccalib.so.4.5.5
surfel_mapping: /usr/local/lib/libopencv_dnn_objdetect.so.4.5.5
surfel_mapping: /usr/local/lib/libopencv_dnn_superres.so.4.5.5
surfel_mapping: /usr/local/lib/libopencv_dpm.so.4.5.5
surfel_mapping: /usr/local/lib/libopencv_face.so.4.5.5
surfel_mapping: /usr/local/lib/libopencv_freetype.so.4.5.5
surfel_mapping: /usr/local/lib/libopencv_fuzzy.so.4.5.5
surfel_mapping: /usr/local/lib/libopencv_hdf.so.4.5.5
surfel_mapping: /usr/local/lib/libopencv_hfs.so.4.5.5
surfel_mapping: /usr/local/lib/libopencv_img_hash.so.4.5.5
surfel_mapping: /usr/local/lib/libopencv_intensity_transform.so.4.5.5
surfel_mapping: /usr/local/lib/libopencv_line_descriptor.so.4.5.5
surfel_mapping: /usr/local/lib/libopencv_mcc.so.4.5.5
surfel_mapping: /usr/local/lib/libopencv_quality.so.4.5.5
surfel_mapping: /usr/local/lib/libopencv_rapid.so.4.5.5
surfel_mapping: /usr/local/lib/libopencv_reg.so.4.5.5
surfel_mapping: /usr/local/lib/libopencv_rgbd.so.4.5.5
surfel_mapping: /usr/local/lib/libopencv_saliency.so.4.5.5
surfel_mapping: /usr/local/lib/libopencv_stereo.so.4.5.5
surfel_mapping: /usr/local/lib/libopencv_structured_light.so.4.5.5
surfel_mapping: /usr/local/lib/libopencv_superres.so.4.5.5
surfel_mapping: /usr/local/lib/libopencv_surface_matching.so.4.5.5
surfel_mapping: /usr/local/lib/libopencv_tracking.so.4.5.5
surfel_mapping: /usr/local/lib/libopencv_videostab.so.4.5.5
surfel_mapping: /usr/local/lib/libopencv_viz.so.4.5.5
surfel_mapping: /usr/local/lib/libopencv_wechat_qrcode.so.4.5.5
surfel_mapping: /usr/local/lib/libopencv_xfeatures2d.so.4.5.5
surfel_mapping: /usr/local/lib/libopencv_xobjdetect.so.4.5.5
surfel_mapping: /usr/local/lib/libopencv_xphoto.so.4.5.5
surfel_mapping: /usr/lib/x86_64-linux-gnu/libpcl_apps.so
surfel_mapping: /usr/lib/x86_64-linux-gnu/libpcl_outofcore.so
surfel_mapping: /usr/lib/x86_64-linux-gnu/libpcl_people.so
surfel_mapping: /usr/lib/x86_64-linux-gnu/libboost_system.so
surfel_mapping: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
surfel_mapping: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
surfel_mapping: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
surfel_mapping: /usr/lib/x86_64-linux-gnu/libboost_regex.so
surfel_mapping: /usr/lib/x86_64-linux-gnu/libqhull.so
surfel_mapping: /usr/lib/x86_64-linux-gnu/libvtkChartsCore-7.1.so.7.1p.1
surfel_mapping: /usr/lib/x86_64-linux-gnu/libvtkInfovisCore-7.1.so.7.1p.1
surfel_mapping: /usr/lib/x86_64-linux-gnu/libfreetype.so
surfel_mapping: /usr/lib/x86_64-linux-gnu/libz.so
surfel_mapping: /usr/lib/x86_64-linux-gnu/libjpeg.so
surfel_mapping: /usr/lib/x86_64-linux-gnu/libpng.so
surfel_mapping: /usr/lib/x86_64-linux-gnu/libtiff.so
surfel_mapping: /usr/lib/x86_64-linux-gnu/libexpat.so
surfel_mapping: /usr/lib/x86_64-linux-gnu/libvtkIOGeometry-7.1.so.7.1p.1
surfel_mapping: /usr/lib/x86_64-linux-gnu/libvtkIOLegacy-7.1.so.7.1p.1
surfel_mapping: /usr/lib/x86_64-linux-gnu/libvtkIOPLY-7.1.so.7.1p.1
surfel_mapping: /usr/lib/x86_64-linux-gnu/libvtkRenderingLOD-7.1.so.7.1p.1
surfel_mapping: /usr/lib/x86_64-linux-gnu/libvtkViewsContext2D-7.1.so.7.1p.1
surfel_mapping: /usr/lib/x86_64-linux-gnu/libvtkViewsCore-7.1.so.7.1p.1
surfel_mapping: /usr/lib/x86_64-linux-gnu/libvtkRenderingContextOpenGL2-7.1.so.7.1p.1
surfel_mapping: /usr/lib/x86_64-linux-gnu/libflann_cpp.so
surfel_mapping: /usr/local/lib/libopencv_shape.so.4.5.5
surfel_mapping: /usr/local/lib/libopencv_highgui.so.4.5.5
surfel_mapping: /usr/local/lib/libopencv_datasets.so.4.5.5
surfel_mapping: /usr/local/lib/libopencv_plot.so.4.5.5
surfel_mapping: /usr/local/lib/libopencv_text.so.4.5.5
surfel_mapping: /usr/local/lib/libopencv_ml.so.4.5.5
surfel_mapping: /usr/local/lib/libopencv_phase_unwrapping.so.4.5.5
surfel_mapping: /usr/local/lib/libopencv_optflow.so.4.5.5
surfel_mapping: /usr/local/lib/libopencv_ximgproc.so.4.5.5
surfel_mapping: /usr/local/lib/libopencv_video.so.4.5.5
surfel_mapping: /usr/local/lib/libopencv_videoio.so.4.5.5
surfel_mapping: /usr/local/lib/libopencv_imgcodecs.so.4.5.5
surfel_mapping: /usr/local/lib/libopencv_objdetect.so.4.5.5
surfel_mapping: /usr/local/lib/libopencv_calib3d.so.4.5.5
surfel_mapping: /usr/local/lib/libopencv_dnn.so.4.5.5
surfel_mapping: /usr/local/lib/libopencv_features2d.so.4.5.5
surfel_mapping: /usr/local/lib/libopencv_flann.so.4.5.5
surfel_mapping: /usr/local/lib/libopencv_photo.so.4.5.5
surfel_mapping: /usr/local/lib/libopencv_imgproc.so.4.5.5
surfel_mapping: /usr/local/lib/libopencv_core.so.4.5.5
surfel_mapping: /usr/lib/x86_64-linux-gnu/libpcl_surface.so
surfel_mapping: /usr/lib/x86_64-linux-gnu/libpcl_keypoints.so
surfel_mapping: /usr/lib/x86_64-linux-gnu/libpcl_tracking.so
surfel_mapping: /usr/lib/x86_64-linux-gnu/libpcl_recognition.so
surfel_mapping: /usr/lib/x86_64-linux-gnu/libpcl_registration.so
surfel_mapping: /usr/lib/x86_64-linux-gnu/libpcl_stereo.so
surfel_mapping: /usr/lib/x86_64-linux-gnu/libpcl_segmentation.so
surfel_mapping: /usr/lib/x86_64-linux-gnu/libpcl_features.so
surfel_mapping: /usr/lib/x86_64-linux-gnu/libpcl_filters.so
surfel_mapping: /usr/lib/x86_64-linux-gnu/libpcl_sample_consensus.so
surfel_mapping: /usr/lib/x86_64-linux-gnu/libpcl_ml.so
surfel_mapping: /usr/lib/x86_64-linux-gnu/libpcl_visualization.so
surfel_mapping: /usr/lib/x86_64-linux-gnu/libpcl_search.so
surfel_mapping: /usr/lib/x86_64-linux-gnu/libpcl_kdtree.so
surfel_mapping: /usr/lib/x86_64-linux-gnu/libpcl_io.so
surfel_mapping: /usr/lib/x86_64-linux-gnu/libpcl_octree.so
surfel_mapping: /usr/lib/x86_64-linux-gnu/libpcl_common.so
surfel_mapping: /usr/lib/x86_64-linux-gnu/libvtkInteractionWidgets-7.1.so.7.1p.1
surfel_mapping: /usr/lib/x86_64-linux-gnu/libvtkFiltersModeling-7.1.so.7.1p.1
surfel_mapping: /usr/lib/x86_64-linux-gnu/libvtkInteractionStyle-7.1.so.7.1p.1
surfel_mapping: /usr/lib/x86_64-linux-gnu/libvtkFiltersExtraction-7.1.so.7.1p.1
surfel_mapping: /usr/lib/x86_64-linux-gnu/libvtkFiltersStatistics-7.1.so.7.1p.1
surfel_mapping: /usr/lib/x86_64-linux-gnu/libvtkImagingFourier-7.1.so.7.1p.1
surfel_mapping: /usr/lib/x86_64-linux-gnu/libvtkalglib-7.1.so.7.1p.1
surfel_mapping: /usr/lib/x86_64-linux-gnu/libvtkFiltersHybrid-7.1.so.7.1p.1
surfel_mapping: /usr/lib/x86_64-linux-gnu/libvtkImagingGeneral-7.1.so.7.1p.1
surfel_mapping: /usr/lib/x86_64-linux-gnu/libvtkImagingSources-7.1.so.7.1p.1
surfel_mapping: /usr/lib/x86_64-linux-gnu/libvtkImagingHybrid-7.1.so.7.1p.1
surfel_mapping: /usr/lib/x86_64-linux-gnu/libvtkRenderingAnnotation-7.1.so.7.1p.1
surfel_mapping: /usr/lib/x86_64-linux-gnu/libvtkImagingColor-7.1.so.7.1p.1
surfel_mapping: /usr/lib/x86_64-linux-gnu/libvtkRenderingVolume-7.1.so.7.1p.1
surfel_mapping: /usr/lib/x86_64-linux-gnu/libvtkIOXML-7.1.so.7.1p.1
surfel_mapping: /usr/lib/x86_64-linux-gnu/libvtkIOXMLParser-7.1.so.7.1p.1
surfel_mapping: /usr/lib/x86_64-linux-gnu/libvtkIOCore-7.1.so.7.1p.1
surfel_mapping: /usr/lib/x86_64-linux-gnu/libvtkRenderingContext2D-7.1.so.7.1p.1
surfel_mapping: /usr/lib/x86_64-linux-gnu/libvtkRenderingFreeType-7.1.so.7.1p.1
surfel_mapping: /usr/lib/x86_64-linux-gnu/libfreetype.so
surfel_mapping: /usr/lib/x86_64-linux-gnu/libvtkRenderingOpenGL2-7.1.so.7.1p.1
surfel_mapping: /usr/lib/x86_64-linux-gnu/libvtkImagingCore-7.1.so.7.1p.1
surfel_mapping: /usr/lib/x86_64-linux-gnu/libvtkRenderingCore-7.1.so.7.1p.1
surfel_mapping: /usr/lib/x86_64-linux-gnu/libvtkCommonColor-7.1.so.7.1p.1
surfel_mapping: /usr/lib/x86_64-linux-gnu/libvtkFiltersGeometry-7.1.so.7.1p.1
surfel_mapping: /usr/lib/x86_64-linux-gnu/libvtkFiltersSources-7.1.so.7.1p.1
surfel_mapping: /usr/lib/x86_64-linux-gnu/libvtkFiltersGeneral-7.1.so.7.1p.1
surfel_mapping: /usr/lib/x86_64-linux-gnu/libvtkCommonComputationalGeometry-7.1.so.7.1p.1
surfel_mapping: /usr/lib/x86_64-linux-gnu/libvtkFiltersCore-7.1.so.7.1p.1
surfel_mapping: /usr/lib/x86_64-linux-gnu/libvtkIOImage-7.1.so.7.1p.1
surfel_mapping: /usr/lib/x86_64-linux-gnu/libvtkCommonExecutionModel-7.1.so.7.1p.1
surfel_mapping: /usr/lib/x86_64-linux-gnu/libvtkCommonDataModel-7.1.so.7.1p.1
surfel_mapping: /usr/lib/x86_64-linux-gnu/libvtkCommonTransforms-7.1.so.7.1p.1
surfel_mapping: /usr/lib/x86_64-linux-gnu/libvtkCommonMisc-7.1.so.7.1p.1
surfel_mapping: /usr/lib/x86_64-linux-gnu/libvtkCommonMath-7.1.so.7.1p.1
surfel_mapping: /usr/lib/x86_64-linux-gnu/libvtkCommonSystem-7.1.so.7.1p.1
surfel_mapping: /usr/lib/x86_64-linux-gnu/libvtkCommonCore-7.1.so.7.1p.1
surfel_mapping: /usr/lib/x86_64-linux-gnu/libvtksys-7.1.so.7.1p.1
surfel_mapping: /usr/lib/x86_64-linux-gnu/libvtkDICOMParser-7.1.so.7.1p.1
surfel_mapping: /usr/lib/x86_64-linux-gnu/libvtkmetaio-7.1.so.7.1p.1
surfel_mapping: /usr/lib/x86_64-linux-gnu/libz.so
surfel_mapping: /usr/lib/x86_64-linux-gnu/libGLEW.so
surfel_mapping: /usr/lib/x86_64-linux-gnu/libSM.so
surfel_mapping: /usr/lib/x86_64-linux-gnu/libICE.so
surfel_mapping: /usr/lib/x86_64-linux-gnu/libX11.so
surfel_mapping: /usr/lib/x86_64-linux-gnu/libXext.so
surfel_mapping: /usr/lib/x86_64-linux-gnu/libXt.so
surfel_mapping: CMakeFiles/surfel_mapping.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/mt/Miller/VSLAM/project/ch12/dense_RGBD/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable surfel_mapping"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/surfel_mapping.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/surfel_mapping.dir/build: surfel_mapping

.PHONY : CMakeFiles/surfel_mapping.dir/build

CMakeFiles/surfel_mapping.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/surfel_mapping.dir/cmake_clean.cmake
.PHONY : CMakeFiles/surfel_mapping.dir/clean

CMakeFiles/surfel_mapping.dir/depend:
	cd /home/mt/Miller/VSLAM/project/ch12/dense_RGBD/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/mt/Miller/VSLAM/project/ch12/dense_RGBD /home/mt/Miller/VSLAM/project/ch12/dense_RGBD /home/mt/Miller/VSLAM/project/ch12/dense_RGBD/build /home/mt/Miller/VSLAM/project/ch12/dense_RGBD/build /home/mt/Miller/VSLAM/project/ch12/dense_RGBD/build/CMakeFiles/surfel_mapping.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/surfel_mapping.dir/depend

