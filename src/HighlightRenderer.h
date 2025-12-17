#ifndef HIGHLIGHT_RENDERER_H
#define HIGHLIGHT_RENDERER_H

#include <glad/glad.h>
#include <vector>
#include <glm/glm.hpp>
#include <algorithm>

#include "Shader.h"
#include "Model.h"
#include "SelectionSystem.h" 

struct OverlayVertex {
    glm::vec3 position;
    glm::vec2 uv;
};

class HighlightRenderer {
private:
    unsigned int VAO, VBO;
    Shader shader;
    int indexCount = 0;

public:
    HighlightRenderer() 
        : shader("../src/shaders/textured_highlight.vert", "../src/shaders/textured_highlight.frag") 
    {
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, 50000 * sizeof(OverlayVertex), NULL, GL_DYNAMIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(OverlayVertex), (void*)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(OverlayVertex), (void*)offsetof(OverlayVertex, uv));
        glBindVertexArray(0);
    }

    ~HighlightRenderer() {
        glDeleteVertexArrays(1, &VAO);
        glDeleteBuffers(1, &VBO);
    }

    void UpdateTextured(const Model* model, const std::vector<FlatVertex>& flatResults, float uvScale, float normalizeRadius, float rotationDeg) {
        if (!model || flatResults.empty()) {
            indexCount = 0;
            return;
        }

        if (normalizeRadius < 0.0001f) normalizeRadius = 1.0f;

        // 預先計算旋轉矩陣參數 (角度轉弧度)
        float rad = glm::radians(rotationDeg);
        float cosR = cos(rad);
        float sinR = sin(rad);

        std::vector<OverlayVertex> overlayVertices;
        const auto& meshVertices = model->meshes[0].vertices; 

        for (const auto& flatV : flatResults) {
            OverlayVertex ov;
            ov.position = meshVertices[flatV.originalIndex].Position;
            
            // 1. 歸一化 + 縮放 (此時 UV 中心仍在 0,0)
            glm::vec2 centeredUV = (flatV.uv / normalizeRadius) * uvScale;

            // 2. [新增] 旋轉 (繞著 0,0 旋轉)
            // 公式: x' = x*cos - y*sin, y' = x*sin + y*cos
            glm::vec2 rotatedUV;
            rotatedUV.x = centeredUV.x * cosR - centeredUV.y * sinR;
            rotatedUV.y = centeredUV.x * sinR + centeredUV.y * cosR;

            // 3. 位移至貼圖中心 (0.5, 0.5)
            ov.uv = rotatedUV + glm::vec2(0.5f, 0.5f);

            overlayVertices.push_back(ov);
        }

        indexCount = (int)overlayVertices.size();

        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, overlayVertices.size() * sizeof(OverlayVertex), overlayVertices.data(), GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    void Draw(const glm::mat4& view, const glm::mat4& projection, const glm::mat4& modelMatrix, unsigned int textureID) {
        if (indexCount == 0) return;
        shader.use();
        shader.setMat4("view", view);
        shader.setMat4("projection", projection);
        shader.setMat4("model", modelMatrix);
        shader.setInt("texture1", 0);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, textureID);
        glBindVertexArray(VAO);
        glDrawArrays(GL_TRIANGLES, 0, indexCount);
        glBindVertexArray(0);
        glBindTexture(GL_TEXTURE_2D, 0);
    }
};

#endif