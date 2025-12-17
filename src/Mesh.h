#ifndef MESH_H
#define MESH_H

#include <glad/glad.h>
#include <glm/glm.hpp>

#include <vector>
#include <string>
#include <algorithm> // for sort
#include <map>
#include <cmath>
#include "Shader.h"

// 頂點結構
struct Vertex {
    glm::vec3 Position;
    glm::vec3 Normal;
    glm::vec2 TexCoords;
};

// 紋理結構
struct Texture {
    unsigned int id;
    std::string type;
    std::string path;
};

// 面鄰接資訊 (ExpMap 核心)
struct FaceAdj {
    int v[3];       // 3 個頂點 index
    int neigh[3];   // 每個邊對應的鄰居三角形 index；-1 表示無鄰居
};

class Mesh {
public:
    // Mesh 資料
    std::vector<Vertex>       vertices;
    std::vector<unsigned int> indices;
    std::vector<Texture>      textures; // 預留給材質
    std::vector<FaceAdj>      faceAdj;

    unsigned int VAO;

    // ---------------------------
    // 1. 建構子 (Constructor)
    // ---------------------------
    Mesh(std::vector<Vertex> vertices, std::vector<unsigned int> indices, std::vector<Texture> textures = {})
    {
        // 使用 std::move 避免大量資料複製，提升效能
        this->vertices = std::move(vertices);
        this->indices  = std::move(indices);
        this->textures = std::move(textures);

        setupMesh();      // 建立 VAO/VBO
        BuildAdjacency(); // 建立鄰接關係 (修復三角形斷開問題)
    }

    // ---------------------------
    // 2. 移動建構子 (Move Constructor) [重要安全機制]
    // ---------------------------
    // 當 vector 擴充或 push_back 時，所有權會轉移，而不是複製。
    // 這確保舊物件銷毀時，不會誤刪 GPU 上的 VAO/VBO。
    Mesh(Mesh&& other) noexcept
        : vertices(std::move(other.vertices)),
          indices(std::move(other.indices)),
          textures(std::move(other.textures)),
          faceAdj(std::move(other.faceAdj)),
          VAO(other.VAO), VBO(other.VBO), EBO(other.EBO)
    {
        // 將來源物件的 ID 歸零，防止它的解構函式刪除我們的緩衝區
        other.VAO = 0;
        other.VBO = 0;
        other.EBO = 0;
    }

    // 禁止複製 (因為我們管理 GPU 資源，複製會導致 Double Free 錯誤)
    Mesh(const Mesh&) = delete;
    Mesh& operator=(const Mesh&) = delete;

    // ---------------------------
    // 3. 解構函式 (Destructor)
    // ---------------------------
    // 當 Mesh 物件消失時，自動釋放 GPU 記憶體
    ~Mesh()
    {
        if (VAO != 0) glDeleteVertexArrays(1, &VAO);
        if (VBO != 0) glDeleteBuffers(1, &VBO);
        if (EBO != 0) glDeleteBuffers(1, &EBO);
    }

    // ---------------------------
    // 繪製 Mesh
    // ---------------------------
    void Draw(Shader& shader)
    {
        if (VAO == 0) return;

        // 綁定貼圖 (若有)
        unsigned int diffuseNr  = 1;
        unsigned int specularNr = 1;
        for (unsigned int i = 0; i < textures.size(); i++)
        {
            glActiveTexture(GL_TEXTURE0 + i);
            std::string number;
            std::string name = textures[i].type;
            if (name == "texture_diffuse")
                number = std::to_string(diffuseNr++);
            else if (name == "texture_specular")
                number = std::to_string(specularNr++);

            shader.setInt((name + number).c_str(), i);
            glBindTexture(GL_TEXTURE_2D, textures[i].id);
        }

        // 繪製三角形
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, (GLsizei)indices.size(), GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);

        glActiveTexture(GL_TEXTURE0);
    }

    // ---------------------------
    // 建立 Adjacency (Vertex Welding 版)
    // ---------------------------
    void BuildAdjacency()
    {
        // 1. 初始化
        int F = (int)indices.size() / 3;
        faceAdj.resize(F);
        for (int i = 0; i < F; i++) {
            faceAdj[i].v[0] = indices[i * 3];
            faceAdj[i].v[1] = indices[i * 3 + 1];
            faceAdj[i].v[2] = indices[i * 3 + 2];
            faceAdj[i].neigh[0] = faceAdj[i].neigh[1] = faceAdj[i].neigh[2] = -1;
        }

        // 2. 位置焊接 (Position Welding)
        // 這是解決 Model Loading 造成頂點斷開的關鍵
        struct Vec3Key {
            glm::vec3 v;
            bool operator<(const Vec3Key& o) const {
                const float EPS = 1e-6f;
                if (std::abs(v.x - o.v.x) > EPS) return v.x < o.v.x;
                if (std::abs(v.y - o.v.y) > EPS) return v.y < o.v.y;
                if (std::abs(v.z - o.v.z) > EPS) return v.z < o.v.z;
                return false;
            }
        };

        std::map<Vec3Key, int> posMap;
        std::vector<int> uniqueIndexMap(vertices.size());
        int uniqueCount = 0;

        for (size_t i = 0; i < vertices.size(); i++) {
            Vec3Key key = { vertices[i].Position };
            if (posMap.find(key) == posMap.end()) {
                posMap[key] = uniqueCount++;
            }
            uniqueIndexMap[i] = posMap[key];
        }

        // 3. 建立 Edge 對照表
        struct EdgeKey {
            int u, v; // Unique ID
            int faceIdx;
            int edgeIdx;
        };
        std::vector<EdgeKey> edges;
        edges.reserve(F * 3);

        for (int f = 0; f < F; f++) {
            for (int e = 0; e < 3; e++) {
                int idx0 = indices[f * 3 + e];
                int idx1 = indices[f * 3 + (e + 1) % 3];
                
                // 轉成 Unique ID
                int u = uniqueIndexMap[idx0];
                int v = uniqueIndexMap[idx1];

                edges.push_back({ std::min(u, v), std::max(u, v), f, e });
            }
        }

        // 4. 排序 Edge 以尋找鄰居
        std::sort(edges.begin(), edges.end(), [](const EdgeKey& a, const EdgeKey& b) {
            if (a.u != b.u) return a.u < b.u;
            return a.v < b.v;
        });

        // 5. 填入鄰接表
        for (size_t i = 0; i + 1 < edges.size(); i++) {
            if (edges[i].u == edges[i + 1].u && edges[i].v == edges[i + 1].v) {
                int f0 = edges[i].faceIdx;
                int e0 = edges[i].edgeIdx;
                int f1 = edges[i + 1].faceIdx;
                int e1 = edges[i + 1].edgeIdx;

                faceAdj[f0].neigh[e0] = f1;
                faceAdj[f1].neigh[e1] = f0;
            }
        }
    }

private:
    unsigned int VBO, EBO;

    // ---------------------------
    // 初始化 OpenGL Buffer
    // ---------------------------
    void setupMesh()
    {
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        glGenBuffers(1, &EBO);

        glBindVertexArray(VAO);

        // VBO
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), &vertices[0], GL_STATIC_DRAW);

        // EBO
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0], GL_STATIC_DRAW);

        // 1. Position
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);

        // 2. Normal
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Normal));

        // 3. TexCoords
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, TexCoords));

        glBindVertexArray(0);
    }
};

#endif