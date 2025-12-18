#ifndef HIGHLIGHT_RENDERER_H
#define HIGHLIGHT_RENDERER_H

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <vector>
#include <string>

// 引用專案的其他標頭檔
#include "Shader.h"
#include "Model.h"
#include "SelectionSystem.h" // 為了使用 FlatVertex 結構

// -------------------------------------------------------
// HighlightVertex
// 定義傳給 Shader 的資料結構，必須與 textured_highlight.vert 對應
// layout (location = 0) in vec3 aPos;
// layout (location = 1) in vec3 aNormal;
// layout (location = 2) in vec2 aTexCoord;
// -------------------------------------------------------
struct HighlightVertex {
    glm::vec3 Position;
    glm::vec3 Normal;    // [關鍵] 新增法線資料，用於 Shader 計算光照
    glm::vec2 TexCoords;
};

class HighlightRenderer {
private:
    unsigned int VAO, VBO;
    int vertexCount;
    Shader shader;

public:
    // 建構子：初始化 VAO/VBO 與載入 Shader
    HighlightRenderer() 
        : shader("../src/shaders/textured_highlight.vert", "../src/shaders/textured_highlight.frag"), 
          vertexCount(0), VAO(0), VBO(0)
    {
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);

        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);

        // [重要] 設定頂點屬性指標 (Attribute Pointers)
        
        // 1. Position (Location 0)
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(HighlightVertex), (void*)0);

        // 2. Normal (Location 1) - [新增] 用於凹凸感光照計算
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(HighlightVertex), (void*)offsetof(HighlightVertex, Normal));

        // 3. TexCoords (Location 2)
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(HighlightVertex), (void*)offsetof(HighlightVertex, TexCoords));

        glBindVertexArray(0);
    }

    ~HighlightRenderer() {
        glDeleteVertexArrays(1, &VAO);
        glDeleteBuffers(1, &VBO);
    }

    // 取得 Shader 指標 (如果外部需要設定額外參數)
    Shader* GetShader() {
        return &shader;
    }

    // 更新預覽網格資料 (當滑鼠移動或參數改變時呼叫)
    void UpdateTextured(const Model* model, const std::vector<FlatVertex>& flatRes, float uvScale, float normalizeRadius, float rotationDeg) {
        if (!model || flatRes.empty()) {
            vertexCount = 0;
            return;
        }

        std::vector<HighlightVertex> vertices;
        const auto& meshVerts = model->meshes[0].vertices;

        // 防呆
        if (normalizeRadius < 0.0001f) normalizeRadius = 1.0f;

        // 預計算旋轉矩陣
        float rad = glm::radians(rotationDeg);
        float cosR = cos(rad);
        float sinR = sin(rad);

        for (const auto& v : flatRes) {
            // 安全檢查
            if(v.originalIndex >= meshVerts.size()) continue;

            HighlightVertex hv;
            // [關鍵] 從原始模型複製 位置 與 法線
            hv.Position = meshVerts[v.originalIndex].Position;
            hv.Normal   = meshVerts[v.originalIndex].Normal; 

            // 計算 UV (類似 CreateOverlayData 的邏輯)
            glm::vec2 rawUV = (v.uv / normalizeRadius) * uvScale;

            // 旋轉
            glm::vec2 rotatedUV;
            rotatedUV.x = rawUV.x * cosR - rawUV.y * sinR;
            rotatedUV.y = rawUV.x * sinR + rawUV.y * cosR;

            // 置中
            hv.TexCoords = rotatedUV + glm::vec2(0.5f, 0.5f);

            vertices.push_back(hv);
        }

        vertexCount = (int)vertices.size();

        // 更新 GPU 上的資料
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(HighlightVertex), vertices.data(), GL_STATIC_DRAW);
    }

    // 繪製函式
    // [修改] 增加了 viewPos 參數，用於傳遞相機位置給 Shader
    void Draw(const glm::mat4& view, const glm::mat4& proj, const glm::mat4& model, unsigned int textureID, const glm::vec3& viewPos) {
        if (vertexCount == 0) return;

        shader.use();
        shader.setMat4("view", view);
        shader.setMat4("projection", proj);
        shader.setMat4("model", model);
        
        // [關鍵] 傳入相機位置，這樣 Fragment Shader 才能算出正確的光線角度
        shader.setVec3("viewPos", viewPos);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, textureID);
        shader.setInt("texture1", 0);

        glBindVertexArray(VAO);
        glDrawArrays(GL_TRIANGLES, 0, vertexCount);
        glBindVertexArray(0);
    }
};

#endif