#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <array>

using namespace std;


bool GL_INIT = 0;
const GLuint n = 1024;
const GLuint size = 1024;



std::string loadShaderSource(const char* path);

GLuint compileComputeShader(const std::string& src);

int init_gl();

template<int size>
array<array<GLdouble, size>, size> matrix_mult(GLdouble mat1[size][size], GLdouble mat2[size][size], GLuint ssboResult) {
    if (!GL_INIT) {
        if (init_gl() != 0) {
            std::cerr << "Failed to initialize OpenGL context.\n";
            exit(-1);
        }
        GL_INIT = 1;  
    }
    // --- Buffers ---
    GLuint ssboMat1, ssboMat2, ssboSize, ssboResult;
    glGenBuffers(1, &ssboMat1);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboMat1);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(GLdouble) * size * size, mat1, GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssboMat1);

    glGenBuffers(1, &ssboMat2);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboMat2);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(GLdouble) * size * size, mat2, GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssboMat2);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboResult);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(GLdouble) * size * size, nullptr, GL_DYNAMIC_READ);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ssboResult);

    // --- Shader ---
    auto src = loadShaderSource("shaders/matrix_mult.comp");
    GLuint program = compileComputeShader(src);
    glUseProgram(program);

    // --- Dispatch ---
    auto start = std::chrono::steady_clock::now();
    glDispatchCompute(size, size, 1);

    // --- Synchronize ---
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    auto end = std::chrono::steady_clock::now();

    // --- Read results ---
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboResult);
    GLdouble(*out)[size] = (GLdouble(*)[size]) glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);

    glDeleteBuffers(1, &ssboMat1);
    glDeleteBuffers(1, &ssboMat2);
    glDeleteProgram(program);
    return out;
}


void fillMatrix(GLdouble mat[n][n]);

void fillMatrix(array<array<GLdouble, n>, n> mat);

void printMatrix(GLdouble mat[n][n]);
void printMatrix(GLdouble mat[n][n]);
