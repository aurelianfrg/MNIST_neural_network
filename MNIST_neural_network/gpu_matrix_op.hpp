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
void assign_variable(string & src, const char* var_symbol, T value) {
    // sets the variable in the shader source code to the given value
    // raises an exception if the variable is not found
    try {
        src.replace(src.find(var_symbol), strlen(var_symbol), to_string(value));
    }
    catch (const out_of_range& e) {
        cerr << "Error: variable " << var_symbol << " not found in shader source code" << endl;
        throw out_of_range("Variable not found in shader source code");
    }
}


template <typename T>
void matrix_mult(T * mat1, T * mat2, GLuint ssboResult, GLuint height_left, unsigned int common_length, unsigned int width_right) {
	// multiply input square matrices mat1 and mat2 of given size using OpenGL compute shader
	// result is stored in the buffer object ssboResult

    // --- Buffers ---
    GLuint ssboMat1, ssboMat2;
    glGenBuffers(1, &ssboMat1);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboMat1);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(T) * height_left * common_length, mat1, GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssboMat1);
    /*cout << "Matrix 1:" << endl;
    printMatrix<T>(mat1, common_length, height_left);*/

    glGenBuffers(1, &ssboMat2);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboMat2);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(T) * common_length * width_right, mat2, GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssboMat2);
	/*cout << "Matrix 2:" << endl;
	printMatrix<T>(mat2, width_right, common_length); */
    
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboResult);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(T) * height_left * width_right, nullptr, GL_DYNAMIC_READ);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ssboResult);

    // --- Shader ---
    string src;
    if (is_same<T, GLdouble>::value) {
        src = loadShaderSource("shaders/matrix_mult_double.comp");
    }
    else if (is_same<T, GLfloat>::value) {
        src = loadShaderSource("shaders/matrix_mult_float.comp");
	}
    else {
        cerr << "matrix_add: unsupported type" << endl;
        exit(-1);
    }
    assign_variable<GLuint>(src, "%HL%", height_left);
    assign_variable<GLuint>(src, "%CL%", common_length);
    assign_variable<GLuint>(src, "%WR%", width_right);
    GLuint program = compileComputeShader(src);
    glUseProgram(program);

    // --- Dispatch ---
    auto start = std::chrono::steady_clock::now();
    glDispatchCompute(height_left, width_right, 1);

    // --- Synchronize ---
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    auto end = std::chrono::steady_clock::now();

    glDeleteBuffers(1, &ssboMat1);
    glDeleteBuffers(1, &ssboMat2);
    glDeleteProgram(program);

}


template <typename T>
void matrix_add(T* mat1, T* mat2, GLuint ssboResult, unsigned int width, unsigned int height) {
    // add input square matrices mat1 and mat2 of given size using OpenGL compute shader
    // result is stored in the buffer object ssboResult

    // --- Buffers ---
    GLuint ssboMat1, ssboMat2;
    glGenBuffers(1, &ssboMat1);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboMat1);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(T) * width * height, mat1, GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssboMat1);

    glGenBuffers(1, &ssboMat2);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboMat2);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(T) * width * height, mat2, GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssboMat2);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboResult);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(T) * width * height, nullptr, GL_DYNAMIC_READ);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ssboResult);

    // --- Shader ---
    string src;
    if (is_same<T, GLdouble>::value) { 
        src = loadShaderSource("shaders/matrix_add_double.comp");
    }
    else if (is_same<T, GLfloat>::value) {
        src = loadShaderSource("shaders/matrix_add_float.comp");
    }
    else {
		cerr << "matrix_add: unsupported type" << endl;
        exit(-1);
    }
    assign_variable<GLuint>(src, "%H%", height);
    assign_variable<GLuint>(src, "%W%", width);
    GLuint program = compileComputeShader(src);
    glUseProgram(program);

    // --- Dispatch ---
    auto start = std::chrono::steady_clock::now();
    glDispatchCompute(height, width, 1);

    // --- Synchronize ---
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    auto end = std::chrono::steady_clock::now();

    glDeleteBuffers(1, &ssboMat1);
    glDeleteBuffers(1, &ssboMat2);
    glDeleteProgram(program);

}


template <typename T>
void matrix_add_constant_vec(T* mat1, T* vec, GLuint ssboResult, unsigned int width, unsigned int height) {
    // add input square matrices mat1 and mat2 of given size using OpenGL compute shader
    // result is stored in the buffer object ssboResult

    // --- Buffers ---
    GLuint ssboMat1, ssboMat2;
    glGenBuffers(1, &ssboMat1);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboMat1);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(T) * width * height, mat1, GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssboMat1);

    glGenBuffers(1, &ssboMat2);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboMat2);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(T) * height, vec, GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssboMat2);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboResult);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(T) * width * height, nullptr, GL_DYNAMIC_READ);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ssboResult);

    // --- Shader ---
    string src;
    if (is_same<T, GLdouble>::value) {
        src = loadShaderSource("shaders/matrix_add_constant_vec_double.comp");
    }
    else if (is_same<T, GLfloat>::value) {
        src = loadShaderSource("shaders/matrix_add_constant_vec_float.comp");
    }
    else {
        cerr << "matrix_add_constant_vec: unsupported type" << endl;
        exit(-1);
    }
    assign_variable<GLuint>(src, "%H%", height);
    assign_variable<GLuint>(src, "%W%", width);

    GLuint program = compileComputeShader(src);
    glUseProgram(program);

    // --- Dispatch ---
    auto start = std::chrono::steady_clock::now();
    glDispatchCompute(height, width, 1);

    // --- Synchronize ---
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    auto end = std::chrono::steady_clock::now();

    glDeleteBuffers(1, &ssboMat1);
    glDeleteBuffers(1, &ssboMat2);
    glDeleteProgram(program);

}


template <typename T>
void sigmoid_activation(T* input, GLuint ssboResult, unsigned int vectorSize, unsigned int sampleSize) {

    // --- Buffers ---
    GLuint ssboInput;
    glGenBuffers(1, &ssboInput);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboInput);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(T) * vectorSize * sampleSize, input, GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssboInput);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboResult);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(T) * vectorSize * sampleSize, nullptr, GL_DYNAMIC_READ);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssboResult);

    // --- Shader ---
    string src;
    if (is_same<T, GLdouble>::value) {
        src = loadShaderSource("shaders/sigmoid_activation_double.comp");
    }
    else if (is_same<T, GLfloat>::value) {
        src = loadShaderSource("shaders/sigmoid_activation_float.comp");
    }
    else {
        cerr << "sigmoid_activation: unsupported type" << endl;
        exit(-1);
    }
    assign_variable<GLuint>(src, "%VS%", vectorSize);
    assign_variable<GLuint>(src, "%SS%", sampleSize);

    GLuint program = compileComputeShader(src);
    glUseProgram(program);

    // --- Dispatch ---
    auto start = std::chrono::steady_clock::now();
    glDispatchCompute(sampleSize, vectorSize, 1);

    // --- Synchronize ---
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    auto end = std::chrono::steady_clock::now();

    glDeleteBuffers(1, &ssboInput);
    glDeleteProgram(program);
}

template <typename T>
void calculate_dC_dZL_BCE_sigmoid(float* A_L, float* Y, GLuint ssboResult, unsigned int vectorSize, unsigned int sampleSize) {
	// dC_dZ for binary cross-entropy loss with sigmoid activation
    
    // --- Buffers ---
    GLuint ssboA, ssboY;
    glGenBuffers(1, &ssboA);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboA);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * vectorSize * sampleSize, A_L, GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssboA);

    glGenBuffers(1, &ssboY);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboY);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * vectorSize * sampleSize, Y, GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssboY);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboResult);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * vectorSize * sampleSize, nullptr, GL_DYNAMIC_READ);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ssboResult);

    // --- Shader ---
    string src;
    if (is_same<T, GLdouble>::value) {
        cerr << "calculate_dC_dZL_BCE_sigmoid: unsupported type" << endl;
        exit(-1);
        //src = loadShaderSource("shaders/calculate_dC_dZ_BCE_sigmoid_double.comp");
    }
    else if (is_same<T, GLfloat>::value) {
        src = loadShaderSource("shaders/calculate_dC_dZ_BCE_sigmoid_float.comp");
    }
    else {
        cerr << "sigmoid_activation: unsupported type" << endl;
        exit(-1);
    }
    assign_variable<GLuint>(src, "%VS%", vectorSize);
    assign_variable<GLuint>(src, "%SS%", sampleSize);

    GLuint program = compileComputeShader(src);
    glUseProgram(program);

    // --- Dispatch ---
    auto start = std::chrono::steady_clock::now();
    glDispatchCompute(sampleSize, vectorSize, 1);

    // --- Synchronize ---
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    auto end = std::chrono::steady_clock::now();

    glDeleteBuffers(1, &ssboA);
    glDeleteBuffers(1, &ssboY);
    glDeleteProgram(program);
}


template <typename T>
void calculate_dC_dWl(GLuint dC_dZl_ssbo, GLuint Al_previous_ssbo, GLuint ssboResult, unsigned int neurons, unsigned int previousNeurons, unsigned int sampleSize) {

    // --- Buffers ---

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, dC_dZl_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, Al_previous_ssbo);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboResult);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * neurons * previousNeurons, nullptr, GL_DYNAMIC_READ);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ssboResult);

    // --- Shader ---
    string src;
    if (is_same<T, GLdouble>::value) {
        cerr << "calculate_dC_dWl: unsupported type" << endl;
        exit(-1);
    }
    else if (is_same<T, GLfloat>::value) {
        src = loadShaderSource("shaders/calculate_dC_dWl_float.comp");
    }
    else {
        cerr << "sigmoid_activation: unsupported type" << endl;
        exit(-1);
    }
    assign_variable<GLuint>(src, "%N%", neurons);
    assign_variable<GLuint>(src, "%PN%", previousNeurons);
    assign_variable<GLuint>(src, "%SS%", sampleSize);

    GLuint program = compileComputeShader(src);
    glUseProgram(program);

    // --- Dispatch ---
    auto start = std::chrono::steady_clock::now();
    glDispatchCompute(neurons, previousNeurons, 1);

    // --- Synchronize ---
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    auto end = std::chrono::steady_clock::now();

    glDeleteProgram(program);
}

template <typename T>
void compress_matrix_columns(GLuint ssboInput, GLuint ssboResult, unsigned int height, unsigned int width) {

    // --- Buffers ---
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssboInput);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboResult);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(T) * height, nullptr, GL_DYNAMIC_READ);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssboResult);

    // --- Shader ---
    string src;
    if (is_same<T, GLdouble>::value) {
        cerr << "compress_matrix_columns: unsupported type" << endl;
        exit(-1);
    }
    else if (is_same<T, GLfloat>::value) {
        src = loadShaderSource("shaders/matrix_columns_compress_float.comp");
    }
    else {
        cerr << "sigmoid_activation: unsupported type" << endl;
        exit(-1);
    }
    assign_variable<GLuint>(src, "%H%", height);
    assign_variable<GLuint>(src, "%W%", width);

    GLuint program = compileComputeShader(src);
    glUseProgram(program);

    // --- Dispatch ---
    auto start = std::chrono::steady_clock::now();
    glDispatchCompute(height, 1, 1);

    // --- Synchronize ---
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    auto end = std::chrono::steady_clock::now();

    glDeleteProgram(program);
}




template <typename T>
void printMatrix(T* mat, int width, int height) {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            // printing one less digit if negative
            if (mat[i * width + j] < 0) {
                std::cout << std::fixed << std::setprecision(1) << mat[i * width + j] << " ";
            }
            else {
                std::cout << std::fixed << std::setprecision(2) << mat[i * width + j] << " ";
            }
            
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
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            mat[i * width + j] = 2. * T(rand()) / RAND_MAX;
        }
    }
}