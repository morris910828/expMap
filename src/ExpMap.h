#ifndef EXPMAP_H
#define EXPMAP_H

#include <vector>
#include <glm/glm.hpp>
#include "Model.h"

// 展平後的 2D 三角形
struct FlattenTri {
    glm::vec2 a, b, c;
};

// ------------------------------------------------------
// Minimal Working ExpMap
// 把 3D 三角形投影到第一個三角形的 tangent plane
// ------------------------------------------------------
inline void ComputeExpMap(Model* model, int meshIndex, const std::vector<SelectedTri>& selected, std::vector<FlattenTri>& outFlatten)
{
    outFlatten.clear();
    if (!model || selected.empty()) return;

    Mesh& mesh = model->meshes[meshIndex];

    // ---------------------------
    // Step 1: 用第一個三角形建立本地 2D 平面
    // ---------------------------
    const SelectedTri& root = selected[0];

    glm::vec3 p0 = mesh.vertices[root.i0].Position;
    glm::vec3 p1 = mesh.vertices[root.i1].Position;
    glm::vec3 p2 = mesh.vertices[root.i2].Position;

    glm::vec3 e1 = glm::normalize(p1 - p0);
    glm::vec3 normal = glm::normalize(glm::cross(p1 - p0, p2 - p0));
    glm::vec3 e2 = glm::cross(normal, e1);

    // 投影函式：3D → 2D
    auto ProjectTo2D = [&](glm::vec3 v)
    {
        glm::vec3 d = v - p0;
        return glm::vec2(glm::dot(d, e1), glm::dot(d, e2));
    };

    // ---------------------------
    // Step 2: 對所有選取三角形逐一展平
    // ---------------------------
    for (auto& tri : selected)
    {
        glm::vec3 v0 = mesh.vertices[tri.i0].Position;
        glm::vec3 v1 = mesh.vertices[tri.i1].Position;
        glm::vec3 v2 = mesh.vertices[tri.i2].Position;

        FlattenTri ft;
        ft.a = ProjectTo2D(v0);
        ft.b = ProjectTo2D(v1);
        ft.c = ProjectTo2D(v2);

        outFlatten.push_back(ft);
    }
    // ------------------------------------------------------------
    // Step 3: Normalize UV to [0,1]
    // ------------------------------------------------------------
    glm::vec2 minUV(1e9f), maxUV(-1e9f);

    for (auto& f : outFlatten) {
        minUV.x = std::min({minUV.x, f.a.x, f.b.x, f.c.x});
        minUV.y = std::min({minUV.y, f.a.y, f.b.y, f.c.y});

        maxUV.x = std::max({maxUV.x, f.a.x, f.b.x, f.c.x});
        maxUV.y = std::max({maxUV.y, f.a.y, f.b.y, f.c.y});
    }

    glm::vec2 size = maxUV - minUV;

    for (auto& f : outFlatten) {
        f.a = (f.a - minUV) / size;
        f.b = (f.b - minUV) / size;
        f.c = (f.c - minUV) / size;

        // flip Y (OpenGL texture coordinates)
        f.a.y = 1.0f - f.a.y;
        f.b.y = 1.0f - f.b.y;
        f.c.y = 1.0f - f.c.y;
    }

}


#endif
