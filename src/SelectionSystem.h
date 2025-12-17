#ifndef SELECTION_SYSTEM_H
#define SELECTION_SYSTEM_H

#include <vector>
#include <map>
#include <queue>
#include <set>
#include <algorithm>
#include <cmath>
#include <glm/glm.hpp>
#include <iostream>

struct MeshTopology {
    std::vector<std::vector<int>> neighbors; 
    std::vector<std::vector<int>> vertNeighbors; 
    std::vector<glm::vec3> centroids;
    bool isValid = false;
};

struct FlatVertex {
    glm::vec2 uv;
    int originalIndex;
};

struct Vec3Key {
    glm::vec3 v;
    bool operator<(const Vec3Key& o) const {
        const float EPS = 1e-5f;
        if (std::abs(v.x - o.v.x) > EPS) return v.x < o.v.x;
        if (std::abs(v.y - o.v.y) > EPS) return v.y < o.v.y;
        if (std::abs(v.z - o.v.z) > EPS) return v.z < o.v.z;
        return false;
    }
};

struct Edge {
    int v1, v2;
    Edge(int a, int b) {
        if (a < b) { v1 = a; v2 = b; }
        else       { v1 = b; v2 = a; }
    }
    bool operator<(const Edge& other) const {
        return std::tie(v1, v2) < std::tie(other.v1, other.v2);
    }
};

struct DijkstraNode {
    int triIndex;
    float dist;
    bool operator>(const DijkstraNode& other) const {
        return dist > other.dist;
    }
};

class SelectionSystem {
public:
    MeshTopology topology;
    std::vector<int> uniqueIndexMap; 
    std::vector<int> uniqueToRealMap; 

    glm::vec3 Barycentric(glm::vec3 p, glm::vec3 a, glm::vec3 b, glm::vec3 c) {
        glm::vec3 v0 = b - a, v1 = c - a, v2 = p - a;
        float d00 = glm::dot(v0, v0);
        float d01 = glm::dot(v0, v1);
        float d11 = glm::dot(v1, v1);
        float d20 = glm::dot(v2, v0);
        float d21 = glm::dot(v2, v1);
        float denom = d00 * d11 - d01 * d01;
        if (std::abs(denom) < 1e-6f) return glm::vec3(1.0f/3.0f); 
        float v = (d11 * d20 - d01 * d21) / denom;
        float w = (d00 * d21 - d01 * d20) / denom;
        float u = 1.0f - v - w;
        return glm::vec3(u, v, w);
    }

    glm::vec2 UnfoldVertex(glm::vec2 uvA, glm::vec2 uvB, float lenAC, float lenBC) {
        glm::vec2 vAB = uvB - uvA;
        float lenAB = glm::length(vAB);
        if (lenAB < 1e-6f) return uvA;

        float cosAlpha = (lenAB * lenAB + lenAC * lenAC - lenBC * lenBC) / (2.0f * lenAB * lenAC);
        cosAlpha = glm::clamp(cosAlpha, -1.0f, 1.0f);
        float sinAlpha = std::sqrt(1.0f - cosAlpha * cosAlpha);

        glm::vec2 uAB = glm::normalize(vAB);
        glm::vec2 uAC = glm::vec2(
            uAB.x * cosAlpha - uAB.y * sinAlpha, 
            uAB.x * sinAlpha + uAB.y * cosAlpha
        );
        return uvA + uAC * lenAC;
    }

    template <typename VertexType>
    void BuildTopology(const std::vector<VertexType>& vertices, const std::vector<unsigned int>& indices) {
        int numTriangles = (int)indices.size() / 3;
        topology.neighbors.clear();
        topology.neighbors.resize(numTriangles);
        topology.centroids.resize(numTriangles);
        
        uniqueIndexMap.clear();
        uniqueIndexMap.resize(vertices.size());
        uniqueToRealMap.clear();

        std::map<Vec3Key, int> posMap;
        int uniqueCount = 0;

        for (size_t i = 0; i < vertices.size(); i++) {
            Vec3Key key = { vertices[i].Position };
            if (posMap.find(key) == posMap.end()) {
                posMap[key] = uniqueCount++;
                uniqueToRealMap.push_back((int)i);
            }
            uniqueIndexMap[i] = posMap[key];
        }

        topology.vertNeighbors.clear();
        topology.vertNeighbors.resize(uniqueCount);

        std::map<Edge, std::vector<int>> edgeToTriangles;
        for (int i = 0; i < numTriangles; i++) {
            unsigned int idx0 = indices[i * 3 + 0];
            unsigned int idx1 = indices[i * 3 + 1];
            unsigned int idx2 = indices[i * 3 + 2];
            glm::vec3 p0 = vertices[idx0].Position;
            glm::vec3 p1 = vertices[idx1].Position;
            glm::vec3 p2 = vertices[idx2].Position;
            topology.centroids[i] = (p0 + p1 + p2) / 3.0f;

            int u0 = uniqueIndexMap[idx0];
            int u1 = uniqueIndexMap[idx1];
            int u2 = uniqueIndexMap[idx2];

            edgeToTriangles[Edge(u0, u1)].push_back(i);
            edgeToTriangles[Edge(u1, u2)].push_back(i);
            edgeToTriangles[Edge(u2, u0)].push_back(i);

            auto AddUniqueNeighbor = [&](int u, int v) {
                bool found = false;
                for(int n : topology.vertNeighbors[u]) if(n == v) found = true;
                if(!found) topology.vertNeighbors[u].push_back(v);
            };
            AddUniqueNeighbor(u0, u1); AddUniqueNeighbor(u1, u0);
            AddUniqueNeighbor(u1, u2); AddUniqueNeighbor(u2, u1);
            AddUniqueNeighbor(u2, u0); AddUniqueNeighbor(u0, u2);
        }

        for (auto const& [edge, triList] : edgeToTriangles) {
            for (size_t i = 0; i < triList.size(); i++) {
                for (size_t j = i + 1; j < triList.size(); j++) {
                    int tA = triList[i];
                    int tB = triList[j];
                    bool hasB = false; for (int n : topology.neighbors[tA]) if (n == tB) hasB = true;
                    if (!hasB) topology.neighbors[tA].push_back(tB);
                    bool hasA = false; for (int n : topology.neighbors[tB]) if (n == tA) hasA = true;
                    if (!hasA) topology.neighbors[tB].push_back(tA);
                }
            }
        }
        topology.isValid = true;
        std::cout << "[SelectionSystem] Topology Built." << std::endl;
    }

    bool RayTriangleIntersect(const glm::vec3& orig, const glm::vec3& dir, const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2, float& t, glm::vec3& hitPoint) {
        const float EPSILON = 1e-6f;
        glm::vec3 edge1 = v1 - v0;
        glm::vec3 edge2 = v2 - v0;
        glm::vec3 h = glm::cross(dir, edge2);
        float a = glm::dot(edge1, h);
        if (a > -EPSILON && a < EPSILON) return false;
        float f = 1.0f / a;
        glm::vec3 s = orig - v0;
        float u = f * glm::dot(s, h);
        if (u < 0.0f || u > 1.0f) return false;
        glm::vec3 q = glm::cross(s, edge1);
        float v = f * glm::dot(dir, q);
        if (v < 0.0f || u + v > 1.0f) return false;
        t = f * glm::dot(edge2, q);
        if (t > EPSILON) {
            hitPoint = orig + dir * t;
            return true;
        }
        return false;
    }

    template <typename VertexType>
    int PickTriangle(const glm::vec3& rayOrigin, const glm::vec3& rayDir, const std::vector<VertexType>& vertices, const std::vector<unsigned int>& indices, glm::vec3& outHitPos) {
        float minDist = std::numeric_limits<float>::max();
        int hitIndex = -1;
        for (int i = 0; i < indices.size(); i += 3) {
            glm::vec3 v0 = vertices[indices[i]].Position;
            glm::vec3 v1 = vertices[indices[i+1]].Position;
            glm::vec3 v2 = vertices[indices[i+2]].Position;
            float t; glm::vec3 p;
            if (RayTriangleIntersect(rayOrigin, rayDir, v0, v1, v2, t, p)) {
                if (t < minDist) { minDist = t; hitIndex = i / 3; outHitPos = p; }
            }
        }
        return hitIndex;
    }

    std::vector<int> GetPatchByRadius(int centerTriIndex, float radius) {
        std::vector<int> selectedIndices;
        if (centerTriIndex == -1 || !topology.isValid) return selectedIndices;
        std::queue<int> q;
        std::set<int> visited;
        q.push(centerTriIndex);
        visited.insert(centerTriIndex);
        selectedIndices.push_back(centerTriIndex);
        glm::vec3 centerPos = topology.centroids[centerTriIndex];
        while(!q.empty()) {
            int current = q.front(); q.pop();
            for (int neighbor : topology.neighbors[current]) {
                if (visited.count(neighbor)) continue;
                if (glm::distance(centerPos, topology.centroids[neighbor]) <= radius) {
                    visited.insert(neighbor);
                    selectedIndices.push_back(neighbor);
                    q.push(neighbor);
                }
            }
        }
        return selectedIndices;
    }

    // ----------------------------------------------------
    // [最終強化] 鎖定種子點的高斯-賽德爾鬆弛
    // ----------------------------------------------------
    template <typename VertexType>
    void SmoothUVs(std::map<int, glm::vec2>& uvMap, 
                   const std::vector<VertexType>& vertices, 
                   const std::set<int>& activeVertices,
                   const std::set<int>& lockedVertices, 
                   int iterations = 10) 
    {
        for (int iter = 0; iter < iterations; iter++) {
            for (int uIdx : activeVertices) {
                
                // 鎖定的中心點不動
                if (lockedVertices.count(uIdx)) continue;

                // 查表取得 3D 資料
                if (uIdx >= uniqueToRealMap.size()) continue;
                int realVIdx = uniqueToRealMap[uIdx];

                glm::vec3 pCenter = vertices[realVIdx].Position;
                glm::vec2 uvCenter = uvMap[uIdx];

                glm::vec2 force(0.0f);
                float weightSum = 0.0f;

                for (int nIdx : topology.vertNeighbors[uIdx]) {
                    if (uvMap.find(nIdx) == uvMap.end()) continue;

                    if (nIdx >= uniqueToRealMap.size()) continue;
                    int realNIdx = uniqueToRealMap[nIdx];

                    glm::vec3 pNeighbor = vertices[realNIdx].Position;
                    glm::vec2 uvNeighbor = uvMap[nIdx];

                    float dist3D = glm::distance(pCenter, pNeighbor);
                    
                    // [修改] 計算權重：距離越近，權重越大 (1.0 / dist)
                    // 這能讓網格形狀保持得更好
                    float w = 1.0f / (dist3D + 1e-5f); 

                    // 期望位置
                    glm::vec2 dir = glm::normalize(uvCenter - uvNeighbor);
                    // 這裡可以做一個取捨：
                    // 如果用 uvNeighbor + dir * dist3D，是強迫「長度」準確 (Isometric)
                    // 如果這導致破圖，我們可以稍微放寬，只單純做拉普拉斯平滑 (Laplacian Smoothing)
                    // 這裡維持 Isometric 目標，但加入權重
                    glm::vec2 targetPos = uvNeighbor + dir * dist3D;
                    
                    force += targetPos * w;
                    weightSum += w;
                }

                if (weightSum > 0.0f) {
                    glm::vec2 idealPos = force / weightSum;
                    // 0.6f 是更新率，數值越高收斂越快，但也越容易震盪
                    uvMap[uIdx] = glm::mix(uvCenter, idealPos, 0.6f);
                }
            }
        }
    }

    template <typename VertexType>
    std::vector<FlatVertex> FlattenPatch(const std::vector<int>& selectedTriangles, 
                                         const std::vector<VertexType>& vertices, 
                                         const std::vector<unsigned int>& indices,
                                         const glm::vec3& mouseWorldPos) 
    {
        std::vector<FlatVertex> flatResult;
        if (selectedTriangles.empty() || uniqueIndexMap.empty()) return flatResult;

        std::map<int, glm::vec2> uvMap;
        std::set<int> processedTris;
        std::set<int> activeUniqueVertices;
        
        // [新增] 鎖定頂點集合
        std::set<int> lockedVertices;

        int seedIdx = selectedTriangles[0];
        int idx0 = indices[seedIdx * 3 + 0];
        int idx1 = indices[seedIdx * 3 + 1];
        int idx2 = indices[seedIdx * 3 + 2];
        int u0 = uniqueIndexMap[idx0], u1 = uniqueIndexMap[idx1], u2 = uniqueIndexMap[idx2];
        glm::vec3 p0 = vertices[idx0].Position, p1 = vertices[idx1].Position, p2 = vertices[idx2].Position;

        uvMap[u0] = glm::vec2(0.0f, 0.0f);
        uvMap[u1] = glm::vec2(glm::distance(p0, p1), 0.0f);
        uvMap[u2] = UnfoldVertex(uvMap[u0], uvMap[u1], glm::distance(p0, p2), glm::distance(p1, p2));
        
        activeUniqueVertices.insert(u0); activeUniqueVertices.insert(u1); activeUniqueVertices.insert(u2);
        // [關鍵] 鎖定中心三角形的三個頂點，作為平滑的錨點
        lockedVertices.insert(u0); lockedVertices.insert(u1); lockedVertices.insert(u2);

        glm::vec3 triNormal = glm::normalize(glm::cross(p1 - p0, p2 - p0));
        glm::vec3 worldUp = (std::abs(glm::dot(triNormal, glm::vec3(0,1,0))) > 0.9f) ? glm::vec3(0,0,1) : glm::vec3(0,1,0);
        glm::vec3 projUp = glm::normalize(worldUp - triNormal * glm::dot(worldUp, triNormal));
        glm::vec3 edge0Dir = glm::normalize(p1 - p0);
        float angle = std::acos(glm::clamp(glm::dot(edge0Dir, projUp), -1.0f, 1.0f));
        if (glm::dot(glm::cross(edge0Dir, projUp), triNormal) < 0) angle = -angle;
        float correctionAngle = 1.5707963f - angle; 
        float cosR = std::cos(correctionAngle), sinR = std::sin(correctionAngle);

        glm::vec3 bary = Barycentric(mouseWorldPos, p0, p1, p2);
        glm::vec2 mouseRawUV = bary.x * uvMap[u0] + bary.y * uvMap[u1] + bary.z * uvMap[u2];

        std::priority_queue<DijkstraNode, std::vector<DijkstraNode>, std::greater<DijkstraNode>> pq;
        pq.push({seedIdx, 0.0f});
        processedTris.insert(seedIdx);
        std::set<int> selectionSet(selectedTriangles.begin(), selectedTriangles.end());

        while (!pq.empty()) {
            DijkstraNode current = pq.top(); pq.pop();
            int currTri = current.triIndex;

            for (int neighbor : topology.neighbors[currTri]) {
                if (selectionSet.count(neighbor) && processedTris.find(neighbor) == processedTris.end()) {
                    int nIdx[3] = { (int)indices[neighbor*3], (int)indices[neighbor*3+1], (int)indices[neighbor*3+2] };
                    int uIds[3] = { uniqueIndexMap[nIdx[0]], uniqueIndexMap[nIdx[1]], uniqueIndexMap[nIdx[2]] };
                    bool has[3] = { uvMap.count(uIds[0]) > 0, uvMap.count(uIds[1]) > 0, uvMap.count(uIds[2]) > 0 };

                    bool success = false;
                    if (has[0] && has[1] && !has[2]) {
                        float d02 = glm::distance(vertices[nIdx[0]].Position, vertices[nIdx[2]].Position);
                        float d12 = glm::distance(vertices[nIdx[1]].Position, vertices[nIdx[2]].Position);
                        uvMap[uIds[2]] = UnfoldVertex(uvMap[uIds[0]], uvMap[uIds[1]], d02, d12);
                        activeUniqueVertices.insert(uIds[2]); success = true;
                    }
                    else if (has[1] && has[2] && !has[0]) {
                        float d10 = glm::distance(vertices[nIdx[1]].Position, vertices[nIdx[0]].Position);
                        float d20 = glm::distance(vertices[nIdx[2]].Position, vertices[nIdx[0]].Position);
                        uvMap[uIds[0]] = UnfoldVertex(uvMap[uIds[1]], uvMap[uIds[2]], d10, d20);
                        activeUniqueVertices.insert(uIds[0]); success = true;
                    }
                    else if (has[2] && has[0] && !has[1]) {
                        float d21 = glm::distance(vertices[nIdx[2]].Position, vertices[nIdx[1]].Position);
                        float d01 = glm::distance(vertices[nIdx[0]].Position, vertices[nIdx[1]].Position);
                        uvMap[uIds[1]] = UnfoldVertex(uvMap[uIds[2]], uvMap[uIds[0]], d21, d01);
                        activeUniqueVertices.insert(uIds[1]); success = true;
                    }
                    else if (has[0] && has[1] && has[2]) success = true;

                    if (success) {
                        processedTris.insert(neighbor);
                        float newDist = glm::distance(topology.centroids[seedIdx], topology.centroids[neighbor]);
                        pq.push({neighbor, newDist});
                    }
                }
            }
        }

        // [執行] 執行 10 次迭代
        SmoothUVs(uvMap, vertices, activeUniqueVertices, lockedVertices, 100);

        for (int tri : selectedTriangles) {
            for (int k = 0; k < 3; k++) {
                int vIdx = indices[tri * 3 + k];
                int uIdx = uniqueIndexMap[vIdx];
                glm::vec2 rawUV = (uvMap.count(uIdx)) ? uvMap[uIdx] : glm::vec2(0.0f);
                glm::vec2 centeredUV = rawUV - mouseRawUV; 
                
                glm::vec2 rotatedUV; 
                rotatedUV.x = centeredUV.x * cosR - centeredUV.y * sinR;
                rotatedUV.y = centeredUV.x * sinR + centeredUV.y * cosR;
                
                flatResult.push_back({ rotatedUV, vIdx });
            }
        }
        return flatResult;
    }
};

#endif