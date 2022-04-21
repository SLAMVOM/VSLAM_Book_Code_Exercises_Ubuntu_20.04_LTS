# Note: in order to successfully compile this file on ubuntu 20.04, 
# the prerequisitie is: installing libfmt-dev (if it has not been done)
# Also, the <example> folder should not be compiled alone.
# It should be complied with the parent folder <ch4>,
# otherwise the Sophus package will not be included, hence causing errors

option(USE_UBUNTU_20 "Set to ON if you are using Ubuntu 20.04" ON)

find_package(Pangolin REQUIRED)

if(USE_UBUNTU_20)
	message("You are using Ubuntu 20.04, fmt::fmt will be linked")
	find_package(fmt REQUIRED)
	set(FMT_LIBRARIES fmt::fmt)
endif()

include_directories(${Pangolin_INCLUDE_DIRS})

add_executable( trajectoryError trajectoryError.cpp )

target_link_libraries(trajectoryError ${Pangolin_LIBRARIES} ${FMT_LIBRARIES} fmt)
