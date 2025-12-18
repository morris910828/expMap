#ifndef DECAL_SYSTEM_H
#define DECAL_SYSTEM_H

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <vector>
#include <string>
#include <iostream>

// ==========================================================
// [結構定義] OverlayVertex
// 用於儲存 Decal 的頂點資料。
// 必須包含 Position, Normal, UV 以配合 textured_highlight.vert
// ==========================================================
struct OverlayVertex {
    glm::vec3 position;
    glm::vec3 normal;   // [新增] 法線資料，用於光照計算
    glm::vec2 uv;
};

// ==========================================================
// [類別] Decal
// 代表單一張貼圖實體
// ==========================================================
class Decal {
public:
    std::string name;
    unsigned int textureID;
    
    // 幾何資料
    std::vector<OverlayVertex> vertices; 
    std::vector<int> selectionIndices; // 記錄它覆蓋了哪些原始三角形 (編輯用)
    
    // 參數 (用於重新編輯)
    int centerTriangle;
    float radius;
    float textureTiling;
    float rotation;
    
    // OpenGL 物件
    unsigned int VAO = 0, VBO = 0;

    // 預設建構子
    Decal() {}

    // 建構子：初始化並建立 Mesh
    Decal(std::string n, unsigned int texID, std::vector<OverlayVertex> verts, std::vector<int> indices, int center, float rad, float tiling, float rot)
        : name(n), textureID(texID), vertices(verts), selectionIndices(indices), centerTriangle(center), radius(rad), textureTiling(tiling), rotation(rot)
    {
        SetupMesh();
    }

    // 建立 OpenGL 緩衝區
    void SetupMesh() {
        if (vertices.empty()) return;

        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);

        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        // 填入資料
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(OverlayVertex), vertices.data(), GL_STATIC_DRAW);

        // [重要] 設定 Vertex Attributes (必須跟 Shader 的 layout 對齊)
        
        // 1. Position (Location 0)
        // layout (location = 0) in vec3 aPos;
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(OverlayVertex), (void*)0);
        
        // 2. Normal (Location 1) 
        // layout (location = 1) in vec3 aNormal;
        glEnableVertexAttribArray(1); 
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(OverlayVertex), (void*)offsetof(OverlayVertex, normal));

        // 3. TexCoords (Location 2)
        // layout (location = 2) in vec2 aTexCoord;
        glEnableVertexAttribArray(2); 
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(OverlayVertex), (void*)offsetof(OverlayVertex, uv));

        glBindVertexArray(0);
    }

    // 釋放資源
    void Destroy() {
        if (VAO) glDeleteVertexArrays(1, &VAO);
        if (VBO) glDeleteBuffers(1, &VBO);
        VAO = 0; VBO = 0;
    }
};

// ==========================================================
// [系統] DecalSystem
// 管理所有貼圖的繪製與儲存
// ==========================================================
class DecalSystem {
public:
    std::vector<Decal> decals;

    void AddDecal(const Decal& decal) {
        decals.push_back(decal);
    }

    // 繪製單一 Decal 或全部
    // 注意：外部必須先設定好 Shader (textured_highlight) 和 OpenGL 狀態 (Blend/Offset)
    void Draw(const glm::mat4& view, const glm::mat4& proj, const glm::mat4& model, int editingIndex = -1) {
        for (int i = 0; i < decals.size(); i++) {
            // 如果正在編輯這個 Decal，就不畫它 (改由 HighlightRenderer 畫預覽)
            // 這樣才不會出現重疊閃爍
            if (i == editingIndex) continue;

            Decal& d = decals[i];
            if (d.VAO == 0) continue;

            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, d.textureID);

            glBindVertexArray(d.VAO);
            glDrawArrays(GL_TRIANGLES, 0, (int)d.vertices.size());
            glBindVertexArray(0);
        }
    }
};

#endif