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
#include <cstring>

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
void matrix_mult(T * mat1, T * mat2, GLuint ssboResult, unsigned int height_left, unsigned int common_length, unsigned int width_right) {
	// multiply input square matrices mat1 and mat2 of given size using OpenGL compute shader
	// result is stored in the buffer object ssboResult

    // --- Buffers ---
    GLuint ssboMat1, ssboMat2;
    glGenBuffers(1, &ssboMat1);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboMat1);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(T) * height_left * common_length, mat1, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssboMat1);
    /*cout << "Matrix 1:" << endl;
    printMatrix<T>(mat1, common_length, height_left);*/

    glGenBuffers(1, &ssboMat2);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboMat2);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(T) * common_length * width_right, mat2, GL_DYNAMIC_DRAW);
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
    glDispatchCompute(height_left, width_right, 1);

    // --- Synchronize ---
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    glDeleteBuffers(1, &ssboMat1);
    glDeleteBuffers(1, &ssboMat2);
    glDeleteProgram(program);

}

template <typename T>
void matrix_mult(T* mat1, GLuint mat2Ssbo, GLuint ssboResult, unsigned int height_left, unsigned int common_length, unsigned int width_right) {
    // multiply input square matrices mat1 and mat2 of given size using OpenGL compute shader
    // result is stored in the buffer object ssboResult

    // --- Buffers ---
    GLuint ssboMat1;
    glGenBuffers(1, &ssboMat1);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboMat1);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(T) * height_left * common_length, mat1, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssboMat1);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, mat2Ssbo);

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
    glDispatchCompute(height_left, width_right, 1);

    // --- Synchronize ---
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    glDeleteBuffers(1, &ssboMat1);
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
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(T) * width * height, mat1, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssboMat1);

    glGenBuffers(1, &ssboMat2);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboMat2);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(T) * width * height, mat2, GL_DYNAMIC_DRAW);
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
    glDispatchCompute(height, width, 1);

    // --- Synchronize ---
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    glDeleteBuffers(1, &ssboMat1);
    glDeleteBuffers(1, &ssboMat2);
    glDeleteProgram(program);

}


template <typename T>
void matrix_add_constant_vec(T* mat, T* vec, GLuint ssboResult, unsigned int width, unsigned int height) {
    // add input square matrices mat1 and mat2 of given size using OpenGL compute shader
    // result is stored in the buffer object ssboResult

    // --- Buffers ---
    GLuint matSsbo, vecSsbo;
    glGenBuffers(1, &matSsbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, matSsbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(T) * width * height, mat, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, matSsbo);

    glGenBuffers(1, &vecSsbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, vecSsbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(T) * height, vec, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, vecSsbo);

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
    glDispatchCompute(height, width, 1);

    // --- Synchronize ---
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    glDeleteBuffers(1, &matSsbo);
    glDeleteBuffers(1, &vecSsbo);
    glDeleteProgram(program);

}



template <typename T>
void matrix_add_constant_vec(GLuint matSsbo, T* vec, GLuint ssboResult, unsigned int width, unsigned int height) {
    // add input square matrices mat1 and mat2 of given size using OpenGL compute shader
    // result is stored in the buffer object ssboResult

    // --- Buffers ---
    GLuint vecSsbo;

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, matSsbo);

    glGenBuffers(1, &vecSsbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, vecSsbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(T) * height, vec, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, vecSsbo);

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
    glDispatchCompute(height, width, 1);

    // --- Synchronize ---
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    glDeleteBuffers(1, &vecSsbo);
    glDeleteProgram(program);

}



template <typename T>
void sigmoid_activation(T* input, GLuint ssboResult, unsigned int vectorSize, unsigned int sampleSize) {

    // --- Buffers ---
    GLuint ssboInput;
    glGenBuffers(1, &ssboInput);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboInput);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(T) * vectorSize * sampleSize, input, GL_DYNAMIC_DRAW);
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
    glDispatchCompute(sampleSize, vectorSize, 1);

    // --- Synchronize ---
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    glDeleteBuffers(1, &ssboInput);
    glDeleteProgram(program);
}

template <typename T>
void sigmoid_activation(GLuint inputSsbo, GLuint ssboResult, unsigned int vectorSize, unsigned int sampleSize) {

    // --- Buffers ---
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, inputSsbo);

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
    glDispatchCompute(sampleSize, vectorSize, 1);

    // --- Synchronize ---
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    glDeleteProgram(program);
}

template <typename T>
void ReLu_activation(GLuint inputSsbo, GLuint ssboResult, unsigned int vectorSize, unsigned int sampleSize) {

    // --- Buffers ---
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, inputSsbo);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboResult);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(T) * vectorSize * sampleSize, nullptr, GL_DYNAMIC_READ);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssboResult);

    // --- Shader ---
    string src;
    if (is_same<T, GLfloat>::value) {
        src = loadShaderSource("shaders/ReLu_activation_float.comp");
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
    glDispatchCompute(sampleSize, vectorSize, 1);

    // --- Synchronize ---
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    glDeleteProgram(program);
}

template <typename T>
void softmax_activation(GLuint inputSsbo, GLuint ssboResult, unsigned int vectorSize, unsigned int sampleSize) {

    // --- Buffers ---
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, inputSsbo);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboResult);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(T) * vectorSize * sampleSize, nullptr, GL_DYNAMIC_READ);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssboResult);

    // --- Shader ---
    string src;
    if (is_same<T, GLfloat>::value) {
        src = loadShaderSource("shaders/softmax_activation_float.comp");
    }
    else {
        cerr << "softmax_activation: unsupported type" << endl;
        exit(-1);
    }
    assign_variable<GLuint>(src, "%VS%", vectorSize);
    assign_variable<GLuint>(src, "%SS%", sampleSize);

    GLuint program = compileComputeShader(src);
    glUseProgram(program);

    // --- Dispatch ---
    glDispatchCompute(sampleSize, vectorSize, 1);

    // --- Synchronize ---
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    glDeleteProgram(program);
}

template <typename T>
void calculate_dC_dZL_BCE_sigmoid(float* A_L, float* Y, GLuint ssboResult, unsigned int vectorSize, unsigned int sampleSize) {
	// dC_dZ for binary cross-entropy loss with sigmoid activation
    
    // --- Buffers ---
    GLuint ssboA, ssboY;
    glGenBuffers(1, &ssboA);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboA);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * vectorSize * sampleSize, A_L, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssboA);

    glGenBuffers(1, &ssboY);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboY);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * vectorSize * sampleSize, Y, GL_DYNAMIC_DRAW);
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
        cerr << "calculate_dC_dZ_BCE_sigmoid: unsupported type" << endl;
        exit(-1);
    }
    assign_variable<GLuint>(src, "%VS%", vectorSize);
    assign_variable<GLuint>(src, "%SS%", sampleSize);

    GLuint program = compileComputeShader(src);
    glUseProgram(program);

    // --- Dispatch ---
    glDispatchCompute(sampleSize, vectorSize, 1);

    // --- Synchronize ---
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    glDeleteBuffers(1, &ssboA);
    glDeleteBuffers(1, &ssboY);
    glDeleteProgram(program);
}

template <typename T>
void calculate_dC_dZL_CCE_softmax(float* A_L, float* Y, GLuint ssboResult, unsigned int vectorSize, unsigned int sampleSize) {
    // dC_dZ for binary cross-entropy loss with sigmoid activation

    // --- Buffers ---
    GLuint ssboA, ssboY;
    glGenBuffers(1, &ssboA);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboA);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * vectorSize * sampleSize, A_L, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssboA);

    glGenBuffers(1, &ssboY);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboY);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * vectorSize * sampleSize, Y, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssboY);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboResult);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * vectorSize * sampleSize, nullptr, GL_DYNAMIC_READ);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ssboResult);

    // --- Shader ---
    string src;
    if (is_same<T, GLdouble>::value) {
        cerr << "calculate_dC_dZ_CCE_softmax: unsupported type" << endl;
        exit(-1);
        //src = loadShaderSource("shaders/calculate_dC_dZ_BCE_sigmoid_double.comp");
    }
    else if (is_same<T, GLfloat>::value) {
        src = loadShaderSource("shaders/calculate_dC_dZ_CCE_softmax_float.comp");
    }
    else {
        cerr << "calculate_dC_dZ_CCE_softmax: unsupported type" << endl;
        exit(-1);
    }
    assign_variable<GLuint>(src, "%VS%", vectorSize);
    assign_variable<GLuint>(src, "%SS%", sampleSize);

    GLuint program = compileComputeShader(src);
    glUseProgram(program);

    // --- Dispatch ---
    glDispatchCompute(sampleSize, vectorSize, 1);

    // --- Synchronize ---
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

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
    glDispatchCompute(neurons, previousNeurons, 1);

    // --- Synchronize ---
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    glDeleteProgram(program);
}

template <typename T>
void calculate_dC_dbl(GLuint ssboInput, GLuint ssboResult, unsigned int vectorSize, unsigned int sampleSize) {

    // --- Buffers ---
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssboInput);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboResult);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(T) * vectorSize * sampleSize, nullptr, GL_DYNAMIC_READ);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssboResult);

    // --- Shader ---
    string src;
    if (is_same<T, GLdouble>::value) {
        cerr << "compress_matrix_columns: unsupported type" << endl;
        exit(-1);
    }
    else if (is_same<T, GLfloat>::value) {
        src = loadShaderSource("shaders/calculate_dC_dbl_float.comp");
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
    glDispatchCompute(vectorSize, 1, 1);

    // --- Synchronize ---
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    glDeleteProgram(program);
}

template <typename T>
void update_parameters(T* weights, GLuint dC_dWl_ssbo, GLuint ssboResult, unsigned int previous_neurons, unsigned int neurons, T learning_rate) {

    GLuint ssboW;
    glGenBuffers(1, &ssboW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboW);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * previous_neurons * neurons, weights, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssboW);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, dC_dWl_ssbo);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboResult);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * previous_neurons * neurons, nullptr, GL_DYNAMIC_READ);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ssboResult);

    // --- Shader ---
    string src;
    if (is_same<T, GLdouble>::value) {
        cerr << "update_parameters_float: unsupported type" << endl;
        exit(-1);
        //src = loadShaderSource("shaders/calculate_dC_dZ_BCE_sigmoid_double.comp");
    }
    else if (is_same<T, GLfloat>::value) {
        src = loadShaderSource("shaders/update_parameters_float.comp");
    }
    else {
        cerr << "update_parameters: unsupported type" << endl;
        exit(-1);
    }
    assign_variable<GLuint>(src, "%N%", neurons);
    assign_variable<GLuint>(src, "%PN%", previous_neurons);
    assign_variable<T>(src, "%LR%", learning_rate);

    GLuint program = compileComputeShader(src);
    glUseProgram(program);

    // --- Dispatch ---
    glDispatchCompute(neurons, previous_neurons, 1);

    // --- Synchronize ---
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    glDeleteBuffers(1, &ssboW);
    glDeleteProgram(program);
}


template <typename T>
void calculate_dC_dZl_previous_sigmoid(GLuint dC_dZl_ssbo, GLuint A_previous_ssbo, GLuint Wl_ssbo, GLuint ssboResult, unsigned int neurons, unsigned int previousNeurons, unsigned int sampleSize) {

    // --- Buffers ---

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, dC_dZl_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, A_previous_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, Wl_ssbo);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboResult);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * sampleSize * previousNeurons, nullptr, GL_DYNAMIC_READ);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, ssboResult);

    // --- Shader ---
    string src;
    if (is_same<T, GLdouble>::value) {
        cerr << "calculate_dC_dZl_previous: unsupported type" << endl;
        exit(-1);
    }
    else if (is_same<T, GLfloat>::value) {
        src = loadShaderSource("shaders/calculate_dC_dZl_previous_sigmoid_float.comp");
    }
    else {
        cerr << "calculate_dC_dZl_previous: unsupported type" << endl;
        exit(-1);
    }
    assign_variable<GLuint>(src, "%N%", neurons);
    assign_variable<GLuint>(src, "%PN%", previousNeurons);
    assign_variable<GLuint>(src, "%SS%", sampleSize);

    GLuint program = compileComputeShader(src);
    glUseProgram(program);

    // --- Dispatch ---
    glDispatchCompute(sampleSize, previousNeurons, 1);

    // --- Synchronize ---
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    glDeleteProgram(program);
}

template <typename T>
void calculate_dC_dZl_previous_ReLu(GLuint dC_dZl_ssbo, GLuint A_previous_ssbo, GLuint Wl_ssbo, GLuint ssboResult, unsigned int neurons, unsigned int previousNeurons, unsigned int sampleSize) {

    // --- Buffers ---

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, dC_dZl_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, A_previous_ssbo);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, Wl_ssbo);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboResult);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * sampleSize * previousNeurons, nullptr, GL_DYNAMIC_READ);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, ssboResult);

    // --- Shader ---
    string src;
    if (is_same<T, GLdouble>::value) {
        cerr << "calculate_dC_dZl_previous: unsupported type" << endl;
        exit(-1);
    }
    else if (is_same<T, GLfloat>::value) {
        src = loadShaderSource("shaders/calculate_dC_dZl_previous_ReLu_float.comp");
    }
    else {
        cerr << "calculate_dC_dZl_previous: unsupported type" << endl;
        exit(-1);
    }
    assign_variable<GLuint>(src, "%N%", neurons);
    assign_variable<GLuint>(src, "%PN%", previousNeurons);
    assign_variable<GLuint>(src, "%SS%", sampleSize);

    GLuint program = compileComputeShader(src);
    glUseProgram(program);

    // --- Dispatch ---
    glDispatchCompute(sampleSize, previousNeurons, 1);

    // --- Synchronize ---
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    glDeleteProgram(program);
}


template <typename T>
void printMatrix(T* mat, int width, int height) {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            // printing one less digit if negative
            if (mat[i * width + j] < 0) {
                std::cout << std::fixed << std::setprecision(2) << mat[i * width + j] << " ";
            }
            else {
                std::cout << std::fixed << std::setprecision(3) << mat[i * width + j] << " ";
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