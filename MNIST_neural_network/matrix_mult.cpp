#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <chrono>
#include <iomanip>

#include "matrix_mult.hpp"


// Utility: load shader source
std::string loadShaderSource(const char* path) {
    std::ifstream file(path);
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

// Utility: compile shader
GLuint compileComputeShader(const std::string& src) {
    GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
    const char* csrc = src.c_str();
    glShaderSource(shader, 1, &csrc, nullptr);
    glCompileShader(shader);

    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char log[512];
        glGetShaderInfoLog(shader, 512, nullptr, log);
        std::cerr << "Compute Shader Compilation Failed:\n" << log << std::endl;
    }
    GLuint program = glCreateProgram();
    glAttachShader(program, shader);
    glLinkProgram(program);
    glDeleteShader(shader);
    return program;
}


void fillMatrix(GLdouble mat[n][n]) {
    static bool seeded = false;
    if (!seeded) {
        srand(1.414);
        seeded = true;
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            mat[i][j] = 2. * double(rand())/RAND_MAX;
        }
    }
}
void printMatrix(GLdouble mat[n][n]) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << std::fixed << std::setprecision(2) << mat[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

int init_gl() {

    if (!glfwInit()) return -1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE); // headless

    GLFWwindow* window = glfwCreateWindow(640, 480, "Compute Shader", nullptr, nullptr);
    //glfwMakeContextCurrent(window);
    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);

    std::cout << "GL version: " << glGetString(GL_VERSION) << "\n";
    std::cout << "GLSL version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << "\n";

	return 0;
}


void fillMatrix(array<array<GLdouble, n>, n> mat) {
    static bool seeded = false;
    if (!seeded) {
        srand(1.414);
        seeded = true;
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            mat[i][j] = 2. * double(rand()) / RAND_MAX;
        }
    }
}
}


//int main() {
//    // --- Initialize GLFW + OpenGL ---
//    if (!glfwInit()) return -1;
//    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
//    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
//    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
//    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE); // headless
//
//    GLFWwindow* window = glfwCreateWindow(640, 480, "Compute Shader", nullptr, nullptr);
//    glfwMakeContextCurrent(window);
//    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
//
//    std::cout << "GL version: " << glGetString(GL_VERSION) << "\n";
//    std::cout << "GLSL version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << "\n";
//
//    // --- Input Data ---
//    GLdouble (*mat1)[n] = new GLdouble[n][n];
//    GLdouble (*mat2)[n] = new GLdouble[n][n];
//    /*GLdouble mat1[n][n] = { 0 };
//    GLdouble mat2[n][n] = { 0 };*/
//    fillMatrix(mat1);
//    fillMatrix(mat2);
//
//    if (n <= 64) {
//        printMatrix(mat1);
//        std::cout << std::endl;
//        printMatrix(mat2);
//        std::cout << std::endl;
//    }
//
//    // --- Buffers ---
//    GLuint ssboMat1, ssboMat2, ssboSize, ssboResult;
//    glGenBuffers(1, &ssboMat1);
//    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboMat1);
//    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(GLdouble) * n * n, mat1, GL_STATIC_DRAW);
//    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssboMat1);
//
//    glGenBuffers(1, &ssboMat2);
//    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboMat2);
//    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(GLdouble) * n * n, mat2, GL_STATIC_DRAW);
//    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssboMat2);
//
//    /*glGenBuffers(1, &ssboSize);
//    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboSize);
//    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(GLuint), &n, GL_STATIC_DRAW);
//    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ssboSize);*/
//
//    glGenBuffers(1, &ssboResult);
//    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboResult);
//    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(GLdouble) * n * n, nullptr, GL_DYNAMIC_READ);
//    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ssboResult);
//
//    // --- Shader ---
//    auto src = loadShaderSource("shaders/matrix_mult.comp");
//    GLuint program = compileComputeShader(src);
//    glUseProgram(program);
//
//    // --- Dispatch ---
//    auto start = std::chrono::steady_clock::now();
//    glDispatchCompute(n, n, 1);
//
//    // --- Synchronize ---
//    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
//    auto end = std::chrono::steady_clock::now();
//
//    // --- Read results ---
//    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboResult);
//    GLdouble (*out)[n] = (GLdouble(*)[n]) glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
//    if (out) {
//        if (n <= 64) {
//            printMatrix(out);
//        }
//        else {
//            std::cout << "Trace: ";
//			GLdouble trace = 0.;
//			for (int i = 0; i < n; ++i) trace += out[i][i];
//			std::cout << std::fixed << std::setprecision(2) << trace << std::endl;
//            std::cout << "Trace / n^2: ";
//            std::cout << std::fixed << std::setprecision(2) << trace / (n*n) << std::endl;
//
//        }
//
//        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
//        std::cout << "Time taken: " << duration << " ns.\n";
//        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
//    }
//    else {
//        std::cerr << "Failed to map buffer for reading.\n";
//    }
//
//    // Cleanup
//    glDeleteBuffers(1, &ssboMat1);
//    glDeleteBuffers(1, &ssboMat2);
//    glDeleteBuffers(1, &ssboResult);
//    glDeleteProgram(program);
//    glfwDestroyWindow(window);
//    glfwTerminate();
//}
