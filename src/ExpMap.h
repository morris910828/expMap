#ifndef EXPMAP_H
#define EXPMAP_H

#include <vector>
#include <queue>
#include <unordered_set>
#include <unordered_map>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "Model.h"

// 展平後的 2D 三角形
struct FlattenTri {
    glm::vec2 a, b, c;
    int triIndex; // 對應的三角形索引
};

// ExpMap 變換參數
struct ExpMapTransform {
    glm::vec2 surfaceOffset = glm::vec2(0.0f);  // 表面測地線偏移
    float rotation = 0.0f;                       // 旋轉角度(弧度)
    float scale = 1.0f;                          // 統一縮放
};

// ExpMap 座標系統資訊
struct ExpMapSystem {
    glm::vec3 origin;   // 原點(p0)
    glm::vec3 axisU;    // U 軸(e1)
    glm::vec3 axisV;    // V 軸(e2)
    glm::vec3 normal;   // 法向量
    
    glm::vec2 uvMin;    // UV 邊界
    glm::vec2 uvMax;
    glm::vec2 uvCenter; // UV 中心點
    
    int meshIndex;      // 所屬 mesh
};

// 三角形鄰接資訊的輔助結構
struct TriInfo {
    int meshIdx;
    int triIdx;
    glm::vec3 v0, v1, v2;
    glm::vec2 uv0, uv1, uv2; // ExpMap 2D 座標
};

// 3D → 2D 投影到 ExpMap 平面
inline glm::vec2 ProjectTo2D(const glm::vec3& point, const ExpMapSystem& sys)
{
    glm::vec3 d = point - sys.origin;
    return glm::vec2(glm::dot(d, sys.axisU), glm::dot(d, sys.axisV));
}

// 2D → 3D 從 ExpMap 平面反推
inline glm::vec3 UnprojectTo3D(const glm::vec2& uv, const ExpMapSystem& sys)
{
    return sys.origin + sys.axisU * uv.x + sys.axisV * uv.y;
}

// 在 UV 中心點應用變換
inline glm::vec2 ApplyTransformAroundCenter(const glm::vec2& uv, 
                                             const ExpMapSystem& sys,
                                             const ExpMapTransform& transform)
{
    // 1. 移到中心點
    glm::vec2 p = uv - sys.uvCenter;
    
    // 2. 縮放
    p = p * transform.scale;
    
    // 3. 旋轉
    float c = cos(transform.rotation);
    float s = sin(transform.rotation);
    glm::vec2 rotated = glm::vec2(
        p.x * c - p.y * s,
        p.x * s + p.y * c
    );
    
    // 4. 移回中心 + 表面偏移
    return rotated + sys.uvCenter + transform.surfaceOffset;
}

// ------------------------------------------------------
// 廣度優先搜尋展開鄰近三角形 (ExpMap Geodesic Unfolding)
// ------------------------------------------------------
inline void ExpandExpMap(Model* model, 
                        int meshIndex,
                        const std::vector<SelectedTri>& initialSelection,
                        const ExpMapSystem& sys,
                        float radius,
                        std::vector<TriInfo>& expandedTris)
{
    expandedTris.clear();
    
    if (!model || initialSelection.empty()) return;
    
    Mesh& mesh = model->meshes[meshIndex];
    
    std::unordered_set<int> visited;
    std::queue<int> toProcess;
    
    // 初始化：展開初始選取的三角形
    for (auto& st : initialSelection)
    {
        int triIdx = -1;
        // 找到對應的三角形索引
        for (int i = 0; i < mesh.indices.size(); i += 3)
        {
            if (mesh.indices[i] == st.i0 && 
                mesh.indices[i+1] == st.i1 && 
                mesh.indices[i+2] == st.i2)
            {
                triIdx = i / 3;
                break;
            }
        }
        
        if (triIdx >= 0)
        {
            visited.insert(triIdx);
            toProcess.push(triIdx);
            
            glm::vec3 p0 = mesh.vertices[st.i0].Position;
            glm::vec3 p1 = mesh.vertices[st.i1].Position;
            glm::vec3 p2 = mesh.vertices[st.i2].Position;
            
            TriInfo info;
            info.meshIdx = meshIndex;
            info.triIdx = triIdx;
            info.v0 = p0; info.v1 = p1; info.v2 = p2;
            info.uv0 = ProjectTo2D(p0, sys);
            info.uv1 = ProjectTo2D(p1, sys);
            info.uv2 = ProjectTo2D(p2, sys);
            
            expandedTris.push_back(info);
        }
    }
    
    // BFS 展開鄰近三角形
    while (!toProcess.empty())
    {
        int currentTri = toProcess.front();
        toProcess.pop();
        
        // 檢查三個鄰居
        if (currentTri >= 0 && currentTri < mesh.faceAdj.size())
        {
            for (int e = 0; e < 3; e++)
            {
                int neighTri = mesh.faceAdj[currentTri].neigh[e];
                
                if (neighTri < 0 || visited.count(neighTri) > 0)
                    continue;
                
                // 獲取鄰居三角形的頂點
                int idx0 = mesh.faceAdj[neighTri].v[0];
                int idx1 = mesh.faceAdj[neighTri].v[1];
                int idx2 = mesh.faceAdj[neighTri].v[2];
                
                glm::vec3 p0 = mesh.vertices[idx0].Position;
                glm::vec3 p1 = mesh.vertices[idx1].Position;
                glm::vec3 p2 = mesh.vertices[idx2].Position;
                
                // 計算中心點距離
                glm::vec3 center = (p0 + p1 + p2) / 3.0f;
                glm::vec2 centerUV = ProjectTo2D(center, sys);
                float dist = glm::length(centerUV - sys.uvCenter);
                
                // 只展開在半徑內的三角形
                if (dist > radius)
                    continue;
                
                visited.insert(neighTri);
                toProcess.push(neighTri);
                
                TriInfo info;
                info.meshIdx = meshIndex;
                info.triIdx = neighTri;
                info.v0 = p0; info.v1 = p1; info.v2 = p2;
                info.uv0 = ProjectTo2D(p0, sys);
                info.uv1 = ProjectTo2D(p1, sys);
                info.uv2 = ProjectTo2D(p2, sys);
                
                expandedTris.push_back(info);
            }
        }
    }
}

// ------------------------------------------------------
// 計算 ExpMap 並建立座標系統
// ------------------------------------------------------
inline ExpMapSystem ComputeExpMap(Model* model, 
                                  int meshIndex, 
                                  const std::vector<SelectedTri>& selected, 
                                  std::vector<FlattenTri>& outFlatten)
{
    outFlatten.clear();
    ExpMapSystem sys;
    sys.meshIndex = meshIndex;
    
    if (!model || selected.empty()) return sys;

    Mesh& mesh = model->meshes[meshIndex];

    // 建立 ExpMap 座標系
    const SelectedTri& root = selected[0];

    glm::vec3 p0 = mesh.vertices[root.i0].Position;
    glm::vec3 p1 = mesh.vertices[root.i1].Position;
    glm::vec3 p2 = mesh.vertices[root.i2].Position;

    sys.origin = p0;
    sys.axisU = glm::normalize(p1 - p0);
    sys.normal = glm::normalize(glm::cross(p1 - p0, p2 - p0));
    sys.axisV = glm::cross(sys.normal, sys.axisU);

    // 投影所有三角形
    for (auto& tri : selected)
    {
        glm::vec3 v0 = mesh.vertices[tri.i0].Position;
        glm::vec3 v1 = mesh.vertices[tri.i1].Position;
        glm::vec3 v2 = mesh.vertices[tri.i2].Position;

        FlattenTri ft;
        ft.a = ProjectTo2D(v0, sys);
        ft.b = ProjectTo2D(v1, sys);
        ft.c = ProjectTo2D(v2, sys);

        outFlatten.push_back(ft);
    }
    
    // 計算 UV 邊界和中心
    sys.uvMin = glm::vec2(1e9f);
    sys.uvMax = glm::vec2(-1e9f);

    for (auto& f : outFlatten) {
        sys.uvMin.x = std::min({sys.uvMin.x, f.a.x, f.b.x, f.c.x});
        sys.uvMin.y = std::min({sys.uvMin.y, f.a.y, f.b.y, f.c.y});
        sys.uvMax.x = std::max({sys.uvMax.x, f.a.x, f.b.x, f.c.x});
        sys.uvMax.y = std::max({sys.uvMax.y, f.a.y, f.b.y, f.c.y});
    }

    sys.uvCenter = (sys.uvMin + sys.uvMax) * 0.5f;

    return sys;
}

// ------------------------------------------------------
// 應用變換 - 最終修正版：幾何遮罩 + 頂點鎖定
// ------------------------------------------------------
inline void ApplyExpMapTransform(Model* model,
                                const std::vector<SelectedTri>& selected,
                                const std::vector<FlattenTri>& flattened,
                                const ExpMapSystem& sys,
                                const ExpMapTransform& transform,
                                std::vector<glm::vec3>& outPositions,
                                std::vector<glm::vec2>& outTexCoords)
{
    outPositions.clear();
    outTexCoords.clear();
    
    if (!model || selected.empty() || flattened.empty()) return;
    
    Mesh& mesh = model->meshes[sys.meshIndex];
    glm::vec2 size = sys.uvMax - sys.uvMin;
    float maxDim = std::max(size.x, size.y);
    
    // 預先計算旋轉
    float c = cos(transform.rotation);
    float s = sin(transform.rotation);

    // 設定基礎顯示半徑 (這決定了 scale = 1.0 時 Patch 有多大)
    // 這裡假設模型單位的 0.3 為基礎半徑，你可以根據需要調整
    float baseRadius = 0.3f; 
    float currentRadius = baseRadius * transform.scale;

    for (size_t i = 0; i < flattened.size(); i++)
    {
        const SelectedTri& st = selected[i];
        const FlattenTri& ft = flattened[i];
        
        // --- 1. 範圍檢查 (幾何縮放) ---
        // 計算三角形中心到 ExpMap 原點的距離
        glm::vec2 triCenter = (ft.a + ft.b + ft.c) / 3.0f;
        float dist = glm::length(triCenter - sys.uvCenter);

        // 如果距離超過當前的縮放半徑，就不產生這個三角形 (隱藏)
        // 這樣視覺上 Patch 就變小了，但不需要改變頂點位置
        if (dist > currentRadius) continue;

        // --- 2. 處理 3D 位置 (服貼) ---
        // 絕對鎖定使用原始模型的頂點，確保 100% 服貼
        outPositions.push_back(mesh.vertices[st.i0].Position);
        outPositions.push_back(mesh.vertices[st.i1].Position);
        outPositions.push_back(mesh.vertices[st.i2].Position);
        
        // --- 3. 處理 UV 座標 (紋理縮放) ---
        glm::vec2 uvs[3] = { ft.a, ft.b, ft.c };
        glm::vec2 finalUVs[3];

        for(int k=0; k<3; k++)
        {
            glm::vec2 p = uvs[k] - sys.uvCenter;

            // 這裡將 UV 除以 scale。
            // 效果：當 Patch 變大 (scale 變大) 時，紋理圖案也會跟著等比例放大。
            // 如果你不希望圖案變大 (想要更多格子)，可以把這行拿掉。
            if (transform.scale > 0.001f) {
                p /= transform.scale;
            }

            // 旋轉
            float px = p.x;
            float py = p.y;
            p.x = px * c - py * s;
            p.y = px * s + py * c;

            // 位移
            p -= transform.surfaceOffset;

            // 正規化
            finalUVs[k] = (p + sys.uvCenter - sys.uvMin) / maxDim;
            finalUVs[k].y = 1.0f - finalUVs[k].y;
        }

        outTexCoords.push_back(finalUVs[0]);
        outTexCoords.push_back(finalUVs[1]);
        outTexCoords.push_back(finalUVs[2]);
    }
}

#endif