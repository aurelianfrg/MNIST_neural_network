# CMakeList.txt : projet CMake pour MNIST_neural_network, incluez la source et définissez
# la logique spécifique au projet ici.
#

set(NAME "MNIST_neural_network")


# Trouver les dépendances AVANT la déclaration de l'exécutable
find_package(glfw3 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(glad CONFIG REQUIRED)
find_package(OpenGL REQUIRED)

add_executable(${NAME}
    "${NAME}.cpp"
    "gpu_matrix_op.cpp"
)

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET MNIST_neural_network PROPERTY CXX_STANDARD 20)
endif()

target_link_libraries(${NAME} PRIVATE
    glfw
    glm::glm
    glad::glad
    OpenGL::GL
)

file(COPY ${CMAKE_SOURCE_DIR}/${NAME}/shaders DESTINATION ${CMAKE_BINARY_DIR}/${NAME})
file(COPY ${CMAKE_SOURCE_DIR}/${NAME}/mnist DESTINATION ${CMAKE_BINARY_DIR}/${NAME})