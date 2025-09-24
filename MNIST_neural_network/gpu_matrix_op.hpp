#pragma once

/*
Provides functions to perform matrix operations using OpenGL compute shaders.
Matrices are represented as 1D arrays in row-major order.
*/

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <array>
#include <algorithm>

using namespace std;


std::string loadShaderSource(const char* path);
GLuint compileComputeShader(const std::string& src);
int init_gl();

template <typename T>
void matrix_mult(T * mat1, T * mat2, GLuint ssboResult, GLuint height_left, GLuint common_length, GLuint width_right) {
	// multiply input square matrices mat1 and mat2 of given size using OpenGL compute shader
	// result is stored in the buffer object ssboResult

    // --- Buffers ---
    GLuint ssboMat1, ssboMat2;
    glGenBuffers(1, &ssboMat1);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboMat1);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(T) * height_left * common_length, mat1, GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssboMat1);

    glGenBuffers(1, &ssboMat2);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboMat2);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(T) * common_length * width_right, mat2, GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssboMat2);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboResult);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(T) * height_left * width_right, nullptr, GL_DYNAMIC_READ);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ssboResult);

    // --- Shader ---
    string src;
    if (is_same<T, GLdouble>::value) {
        src = loadShaderSource("shaders/matrix_mult_double.comp");
    }
    else {
        cerr << "matrix_add: unsupported type" << endl;
        exit(-1);
    }
    src.replace(src.find("%HL%"), 4, to_string(height_left));
    src.replace(src.find("%CL%"), 4, to_string(common_length));
    src.replace(src.find("%WR%"), 4, to_string(width_right));
    GLuint program = compileComputeShader(src);
    glUseProgram(program);

    // --- Dispatch ---
    auto start = std::chrono::steady_clock::now();
    glDispatchCompute(height_left, width_right, 1);

    // --- Synchronize ---
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    auto end = std::chrono::steady_clock::now();

    // --- Read results ---
    // glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboResult);
    // GLdouble(*out)[size] = (GLdouble(*)[size]) glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);

    glDeleteBuffers(1, &ssboMat1);
    glDeleteBuffers(1, &ssboMat2);
    glDeleteProgram(program);

}


template <typename T>
void matrix_add(T* mat1, T* mat2, GLuint ssboResult, GLuint height_left, GLuint common_length, GLuint width_right) {
    // add input square matrices mat1 and mat2 of given size using OpenGL compute shader
    // result is stored in the buffer object ssboResult

    // --- Buffers ---
    GLuint ssboMat1, ssboMat2;
    glGenBuffers(1, &ssboMat1);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboMat1);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(T) * height_left * common_length, mat1, GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssboMat1);

    glGenBuffers(1, &ssboMat2);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboMat2);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(T) * common_length * width_right, mat2, GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssboMat2);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboResult);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(T) * height_left * width_right, nullptr, GL_DYNAMIC_READ);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ssboResult);

    // --- Shader ---
    string src;
    if (is_same<T, GLdouble>::value) { 
        src = loadShaderSource("shaders/matrix_add_double.comp");
    }
    else {
		cerr << "matrix_add: unsupported type" << endl;
        exit(-1);
    }
    
    src.replace(src.find("%HL%"), 4, to_string(height_left));
    src.replace(src.find("%CL%"), 4, to_string(common_length));
    src.replace(src.find("%WR%"), 4, to_string(width_right));
    GLuint program = compileComputeShader(src);
    glUseProgram(program);

    // --- Dispatch ---
    auto start = std::chrono::steady_clock::now();
    glDispatchCompute(height_left, width_right, 1);

    // --- Synchronize ---
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    auto end = std::chrono::steady_clock::now();

    // --- Read results ---
    // glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboResult);
    // GLdouble(*out)[size] = (GLdouble(*)[size]) glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);

    glDeleteBuffers(1, &ssboMat1);
    glDeleteBuffers(1, &ssboMat2);
    glDeleteProgram(program);

}


template <typename T>
void printMatrix(T* mat, int width, int height) {
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < height; ++j) {
            std::cout << std::fixed << std::setprecision(2) << mat[i * width + j] << " ";
        }
        std::cout << std::endl;
    }
}

template <typename T>
void fillMatrix(T* mat, int width, int height) {
    static bool seeded = false;
    if (!seeded) {
        srand(1729);
        seeded = true;
    }
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < height; ++j) {
            mat[i * width + j] = 2. * T(rand()) / RAND_MAX;
        }
    }
}


// outdated
template <typename T>
void printMatrix(T* mat, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            std::cout << std::fixed << std::setprecision(2) << mat[i * size + j] << " ";
        }
        std::cout << std::endl;
    }
}

template <typename T>
void fillMatrix(T* mat, int size) {
    static bool seeded = false;
    if (!seeded) {
        srand(1729);
        seeded = true;
    }
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            mat[i * size + j] = 2. * T(rand()) / RAND_MAX;
        }
    }
}