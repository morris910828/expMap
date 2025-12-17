#ifndef DECAL_SYSTEM_H
#define DECAL_SYSTEM_H

#include <glad/glad.h>
#include <vector>
#include <string>
#include <glm/glm.hpp>
#include "Shader.h"
#include "HighlightRenderer.h" 

struct Decal {
    std::string name;
    unsigned int textureID;
    unsigned int VAO, VBO;
    int vertexCount;
    std::vector<int> sourceTriangles; 
    float radius; 
    int centerTriangle;
    float textureTiling; 
    float rotation; // [新增] 儲存旋轉角度

    // [修改] 建構子加入 _rotation
    Decal(std::string _name, unsigned int _texID, const std::vector<OverlayVertex>& vertices, 
          const std::vector<int>& _srcTris, int _centerTri, float _rad, float _tiling, float _rotation) 
        : name(_name), textureID(_texID), sourceTriangles(_srcTris), centerTriangle(_centerTri), 
          radius(_rad), textureTiling(_tiling), rotation(_rotation)
    {
        // ... (中間的 VAO/VBO 建立程式碼保持不變) ...
        vertexCount = (int)vertices.size();
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(OverlayVertex), vertices.data(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(OverlayVertex), (void*)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(OverlayVertex), (void*)offsetof(OverlayVertex, uv));
        glBindVertexArray(0);
    }

    void Destroy() {
        glDeleteVertexArrays(1, &VAO);
        glDeleteBuffers(1, &VBO);
    }
};

class DecalSystem {
public:
    std::vector<Decal> decals;
    Shader shader;

    DecalSystem() : shader("../src/shaders/textured_highlight.vert", "../src/shaders/textured_highlight.frag") {}

    void AddDecal(const Decal& d) {
        decals.push_back(d);
    }

    // [修改] Draw 函式增加 ignoreIndex 參數
    // 當我們正在編輯第 i 個貼圖時，這裡就不畫它，改由 HighlightRenderer 畫預覽
    void Draw(const glm::mat4& view, const glm::mat4& proj, const glm::mat4& model, int ignoreIndex = -1) {
        shader.use();
        shader.setMat4("view", view);
        shader.setMat4("projection", proj);
        shader.setMat4("model", model);
        shader.setInt("texture1", 0);

        for (int i = 0; i < decals.size(); i++) {
            if (i == ignoreIndex) continue; // 跳過正在編輯的

            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, decals[i].textureID);

            glBindVertexArray(decals[i].VAO);
            glDrawArrays(GL_TRIANGLES, 0, decals[i].vertexCount);
        }
        
        glBindVertexArray(0);
        glBindTexture(GL_TEXTURE_2D, 0);
    }
};

#endif