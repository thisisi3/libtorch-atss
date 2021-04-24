

add_library(
	frcnn
	STATIC
	data.cpp
	utils.cpp
	backbones.cpp
	necks.cpp
	anchor.cpp
	bbox.cpp
	trainer.cpp
	losses.cpp
	atss.cpp)
	
target_include_directories(frcnn PUBLIC ${PROJECT_SOURCE_DIR}/frcnn ${OpenCV_INCLUDE_DIRS})
target_link_libraries(frcnn PUBLIC cvops ${OpenCV_LIBRARIES})

