#ifndef MESH_H
#define MESH_H

#include <glad/glad.h>
#include <glm/glm.hpp>

#include <vector>
#include <string>
#include <algorithm>        // <-- sort 需要這個
#include "Shader.h"

// 頂點結構
struct Vertex {
    glm::vec3 Position;
    glm::vec3 Normal;
    glm::vec2 TexCoords;
};

// 紋理（你暫時沒用到）
struct Texture {
    unsigned int id;
    std::string type;
    std::string path;
};

// 面鄰接資訊（ExpMap 必須）
struct FaceAdj {
    int v[3];       // 3 個頂點 index
    int neigh[3];   // 每個 edge 對應鄰接面 index；沒有鄰居則為 -1
};

class Mesh {
public:
    // Mesh 資料
    std::vector<Vertex>       vertices;
    std::vector<unsigned int> indices;
    std::vector<FaceAdj>      faceAdj;

    unsigned int VAO;

    // ---------------------------
    // Mesh Constructor
    // ---------------------------
    Mesh(std::vector<Vertex> vertices,
         std::vector<unsigned int> indices)
    {
        this->vertices = vertices;
        this->indices  = indices;

        setupMesh();          // 建 VAO / VBO / EBO
        BuildAdjacency();     // 建立 adjacency（你之前少了這一步）
    }

    // ---------------------------
    // 繪製 Mesh
    // ---------------------------
    void Draw(Shader& shader)
    {
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES,
                       (GLsizei)indices.size(),
                       GL_UNSIGNED_INT,
                       0);
        glBindVertexArray(0);
    }

    // ---------------------------
    // 更新 Mesh（修改頂點時用）
    // ---------------------------
    void UpdateMesh()
    {
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferSubData(GL_ARRAY_BUFFER,
                        0,
                        vertices.size() * sizeof(Vertex),
                        vertices.data());
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    // =====================================================
    // 建立 Adjacency（ExpMap 重要核心）
    // =====================================================
    void BuildAdjacency()
    {
        int F = indices.size() / 3;
        faceAdj.resize(F);

        // 初始化每個面（三角形）
        for (int i = 0; i < F; i++)
        {
            faceAdj[i].v[0] = indices[i * 3];
            faceAdj[i].v[1] = indices[i * 3 + 1];
            faceAdj[i].v[2] = indices[i * 3 + 2];

            faceAdj[i].neigh[0] = -1;
            faceAdj[i].neigh[1] = -1;
            faceAdj[i].neigh[2] = -1;
        }

        struct EdgeKey {
            int a, b;    // edge: (a -> b)
            int face;    // 此 edge 屬於哪個面
            int edge;    // 面中第幾條 edge
        };

        std::vector<EdgeKey> edges;
        edges.reserve(F * 3);

        // 建 edge 列表
        for (int f = 0; f < F; f++)
        {
            for (int e = 0; e < 3; e++)
            {
                int v0 = faceAdj[f].v[e];
                int v1 = faceAdj[f].v[(e + 1) % 3];
                edges.push_back({v0, v1, f, e});
            }
        }

        // 排序找 matching edges（無向 edge）
        std::sort(edges.begin(), edges.end(),
            [](auto& a, auto& b)
            {
                if (a.a != b.a) return a.a < b.a;
                return a.b < b.b;
            }
        );

        // 建立 adjacency relation
        for (int i = 0; i + 1 < edges.size(); i++)
        {
            if (edges[i].a == edges[i + 1].a &&
                edges[i].b == edges[i + 1].b)
            {
                int f0 = edges[i].face;
                int e0 = edges[i].edge;
                int f1 = edges[i + 1].face;
                int e1 = edges[i + 1].edge;

                faceAdj[f0].neigh[e0] = f1;
                faceAdj[f1].neigh[e1] = f0;
            }
        }
    }

private:
    unsigned int VBO, EBO;

    // =====================================================
    // 初始化 Mesh（VAO / VBO / EBO）
    // =====================================================
    void setupMesh()
    {
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        glGenBuffers(1, &EBO);

        glBindVertexArray(VAO);

        // VBO
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER,
                     vertices.size() * sizeof(Vertex),
                     vertices.data(),
                     GL_STATIC_DRAW);

        // EBO
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                     indices.size() * sizeof(unsigned int),
                     indices.data(),
                     GL_STATIC_DRAW);

        // 頂點屬性
        // layout = 0 : Position
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE,
                              sizeof(Vertex), (void*)0);

        // layout = 1 : Normal
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE,
                              sizeof(Vertex),
                              (void*)offsetof(Vertex, Normal));

        // layout = 2 : TexCoords
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE,
                              sizeof(Vertex),
                              (void*)offsetof(Vertex, TexCoords));

        glBindVertexArray(0);
    }
};

#endif