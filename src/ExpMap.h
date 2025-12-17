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
    glm::vec3 origin;
    glm::vec3 axisU;
    glm::vec3 axisV;
    glm::vec3 normal;
    
    glm::vec2 uvMin;
    glm::vec2 uvMax;
    glm::vec2 uvCenter;
    
    int meshIndex;
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
    
    // 用於 BFS 的節點結構
    struct Node {
        int triIdx;
        float accumulatedDist; // 從起點累積的表面距離
    };

    std::queue<Node> toProcess;
    std::unordered_map<int, float> visited; // 記錄已訪問的三角形及其距離

    // 1. 初始化種子三角形
    for (auto& st : initialSelection)
    {
        // 找回三角形索引
        int triIdx = -1;
        // 這裡做一個簡單的搜尋 (優化版應該直接存 triIdx，但這裡沿用舊結構)
        // 假設使用者傳進來的 initialSelection 正確對應 mesh indices
        // 為了安全，我們重新遍歷一次或假設 triIndex 已知。
        // 由於 main.cpp 傳入時是將 raycast 的結果轉成 SelectedTri，
        // 我們其實可以直接用 raycast 的 hit.triIndex。
        // 但這裡為了通用，我們用頂點比對 (或如果您確定 SelectedTri 結構不變，可沿用)
        
        // *快速修正*：為了避免複雜的 index 反查，我們假設 initialSelection
        // 裡面的 i0 就是 indices 裡的索引。
        // 實際上，最快的方法是傳入 seedTriIndex 而不是 vector。
        // 但為了不改動太多介面，我們用下面的方式找 index:
        
        for (int i = 0; i < mesh.indices.size(); i += 3) {
            if (mesh.indices[i] == st.i0 && 
                mesh.indices[i+1] == st.i1) { // 比對前兩個點通常就夠了
                triIdx = i / 3;
                break;
            }
        }
        
        if (triIdx >= 0)
        {
            visited[triIdx] = 0.0f;
            toProcess.push({ triIdx, 0.0f });

            // 儲存輸出資訊
            glm::vec3 p0 = mesh.vertices[mesh.indices[triIdx*3+0]].Position;
            glm::vec3 p1 = mesh.vertices[mesh.indices[triIdx*3+1]].Position;
            glm::vec3 p2 = mesh.vertices[mesh.indices[triIdx*3+2]].Position;

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
    
    // 2. BFS 擴散 (累積距離)
    while (!toProcess.empty())
    {
        Node current = toProcess.front();
        toProcess.pop();

        // 取得當前三角形中心點
        glm::vec3 cP0 = mesh.vertices[mesh.indices[current.triIdx*3+0]].Position;
        glm::vec3 cP1 = mesh.vertices[mesh.indices[current.triIdx*3+1]].Position;
        glm::vec3 cP2 = mesh.vertices[mesh.indices[current.triIdx*3+2]].Position;
        glm::vec3 currentCenter = (cP0 + cP1 + cP2) / 3.0f;

        // 檢查鄰居
        if (current.triIdx < mesh.faceAdj.size())
        {
            for (int e = 0; e < 3; e++)
            {
                int neighTri = mesh.faceAdj[current.triIdx].neigh[e];
                if (neighTri < 0) continue;

                // 取得鄰居中心點
                glm::vec3 nP0 = mesh.vertices[mesh.indices[neighTri*3+0]].Position;
                glm::vec3 nP1 = mesh.vertices[mesh.indices[neighTri*3+1]].Position;
                glm::vec3 nP2 = mesh.vertices[mesh.indices[neighTri*3+2]].Position;
                glm::vec3 neighCenter = (nP0 + nP1 + nP2) / 3.0f;

                // [核心修改] 計算兩中心點的 3D 距離並累加
                float stepDist = glm::distance(currentCenter, neighCenter);
                float newDist = current.accumulatedDist + stepDist;

                // 只有在半徑內且未訪問(或發現更短路徑)時才處理
                if (newDist <= radius)
                {
                    if (visited.find(neighTri) == visited.end() || visited[neighTri] > newDist)
                    {
                        visited[neighTri] = newDist;
                        toProcess.push({ neighTri, newDist });

                        // 避免重複加入輸出列表 (這裡簡單處理：只在第一次發現時加入)
                        // 若要嚴謹 Dijkstra 需更新列表，但做選取特效這樣足夠了
                        bool alreadyInList = false; 
                        // 為了效能，這裡暫時假設 BFS 第一次碰到就是最短路徑 (對於 unweighted graph 是真，對於幾何圖形是大約)
                        
                        TriInfo info;
                        info.meshIdx = meshIndex;
                        info.triIdx = neighTri;
                        info.v0 = nP0; info.v1 = nP1; info.v2 = nP2;
                        // 依然使用投影算 UV，但選取範圍由 newDist 控制
                        info.uv0 = ProjectTo2D(nP0, sys);
                        info.uv1 = ProjectTo2D(nP1, sys);
                        info.uv2 = ProjectTo2D(nP2, sys);
                        expandedTris.push_back(info);
                    }
                }
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

// ============== ExpMap.h 修改 ==============

// 修正 ApplyExpMapTransform 函數
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
    
    float c = cos(transform.rotation);
    float s = sin(transform.rotation);

    // 關鍵：讓顯示半徑根據 scale 動態變化
    float baseRadius = 0.3f; 
    float currentRadius = baseRadius * transform.scale;  // 直接乘以 scale

    for (size_t i = 0; i < flattened.size(); i++)
    {
        const SelectedTri& st = selected[i];
        const FlattenTri& ft = flattened[i];
        
        glm::vec2 triCenter = (ft.a + ft.b + ft.c) / 3.0f;
        float dist = glm::length(triCenter - sys.uvCenter);

        // 只顯示在當前半徑內的三角形
        if (dist > currentRadius) continue;

        outPositions.push_back(mesh.vertices[st.i0].Position);
        outPositions.push_back(mesh.vertices[st.i1].Position);
        outPositions.push_back(mesh.vertices[st.i2].Position);
        
        glm::vec2 uvs[3] = { ft.a, ft.b, ft.c };
        glm::vec2 finalUVs[3];

        for(int k=0; k<3; k++)
        {
            glm::vec2 p = uvs[k] - sys.uvCenter;

            // 關鍵：除以 scale，讓貼圖跟著範圍同步放大
            // scale 大 → 範圍大 → UV 變小 → 取更多貼圖 → 格子變多
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