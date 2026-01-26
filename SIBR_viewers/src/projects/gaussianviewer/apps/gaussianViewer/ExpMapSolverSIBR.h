#ifndef EXPMAP_SOLVER_SIBR_H
#define EXPMAP_SOLVER_SIBR_H

#include <core/graphics/Mesh.hpp>
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <vector>
#include <queue>
#include <set>
#include <map>
#include <algorithm>
#include <numeric>
#include <limits> // For infinity
#include <imgui/imgui.h>

inline glm::vec3 toGlm(const sibr::Vector3f& v) { return glm::vec3(v.x(), v.y(), v.z()); }

// ==========================================
// 1. TangentFrame
// ==========================================
struct TangentFrame {
    glm::vec3 origin = {0, 0, 0};
    glm::mat3 axes = glm::mat3(1);

    TangentFrame() = default;
    
    TangentFrame(const glm::vec3& pos, const glm::vec3& normal) {
        origin = pos;
        glm::vec3 n = glm::normalize(normal);
        glm::vec3 x;
        if (std::abs(n.x) >= std::abs(n.y) && std::abs(n.x) >= std::abs(n.z)) {
            x = glm::normalize(glm::vec3(-n.y, n.x, 0.0f));
        } else {
            x = glm::normalize(glm::vec3(0.0f, n.z, -n.y));
        }
        glm::vec3 y = glm::cross(n, x);
        axes = glm::mat3(x, y, n);
    }

    glm::vec3 toLocal(const glm::vec3& worldVec) const {
        return glm::transpose(axes) * worldVec;
    }

    void alignZAxis(const TangentFrame& target) {
        glm::vec3 fromZ = axes[2];
        glm::vec3 toZ = target.axes[2];
        glm::vec3 axis = glm::cross(fromZ, toZ);
        float cosAngle = glm::dot(fromZ, toZ);

        if (glm::length(axis) < 1e-6f) {
            if (cosAngle < 0) axes = glm::mat3(axes[0], -axes[1], -axes[2]);
            return;
        }
        axis = glm::normalize(axis);
        float angle = std::acos(glm::clamp(cosAngle, -1.0f, 1.0f));
        glm::mat3 rot = glm::mat3(glm::rotate(glm::mat4(1.0f), angle, axis));
        axes = rot * axes;
    }
};

struct ExpVertex {
    int id;
    int parentId = -1;
    float dist = 1e9f;
    glm::vec2 uv = {0,0};
    bool frozen = false;
};

// ==========================================
// 2. ExpMap Solver
// ==========================================
class ExpMapSolverSIBR {
public:
    void Init(const sibr::Mesh* mesh) {
        _mesh = mesh;
        buildAdjacency();
    }

    void RegisterAppPointCloud(const std::vector<float>& pointData) { _appCloudData = pointData; }
    void RegisterGeoPointCloud(const std::vector<float>& pointData) { _geoCloudData = pointData; }

    const std::vector<sibr::Vector3u>& GetActiveTriangles() const { return _validTriangles; }

    void Compute(const sibr::Vector3f& hitPos, const sibr::Vector3f& hitNormal, float radius) {
        if(!_mesh) return;

        // Reset
        _vertexData.clear();
        _displayUVs.clear();
        _validTriangles.clear();
        _projectedAppPoints.clear(); 
        _projectedGeoPoints.clear();
        
        // [Dijkstra Reset] 清除上次的路徑選擇
        _startNode = -1; _endNode = -1;
        _dijkstraPath.clear();

        glm::vec3 target = toGlm(hitPos);
        int seedIdx = -1;
        float minD = 1e9f;
        const auto& verts = _mesh->vertices();
        const auto& normals = _mesh->normals();
        
        for(size_t i=0; i<verts.size(); ++i) {
            float d = glm::distance(target, toGlm(verts[i]));
            if(d < minD) { minD = d; seedIdx = (int)i; }
        }

        if(seedIdx == -1) return;

        auto comp = [&](int a, int b){ return _vertexData[a].dist > _vertexData[b].dist; };
        std::priority_queue<int, std::vector<int>, decltype(comp)> pq(comp);

        ExpVertex vSeed;
        vSeed.id = seedIdx;
        vSeed.dist = 0.0f;
        vSeed.uv = {0.0f, 0.0f};
        _vertexData[seedIdx] = vSeed;
        pq.push(seedIdx);

        glm::vec3 seedN = (normals.size() > seedIdx) ? toGlm(normals[seedIdx]) : toGlm(hitNormal);
        _seedFrame = TangentFrame(toGlm(verts[seedIdx]), seedN);

        float maxDistFound = 0.0f;

        while(!pq.empty()) {
            int currIdx = pq.top(); pq.pop();
            if(_vertexData[currIdx].frozen) continue;
            _vertexData[currIdx].frozen = true;

            if(_vertexData[currIdx].dist > radius) continue;

            if (normals.size() > currIdx) {
                glm::vec3 currN = toGlm(normals[currIdx]);
                if (glm::dot(currN, seedN) <= 0.08f) continue; 
            }

            maxDistFound = std::max(maxDistFound, _vertexData[currIdx].dist);

            for(int neighbor : _adj[currIdx]) {
                if(_vertexData.find(neighbor) != _vertexData.end() && _vertexData[neighbor].frozen) continue;

                if(_vertexData.find(neighbor) == _vertexData.end()) {
                    ExpVertex vNew; vNew.id = neighbor; vNew.dist = 1e9f;
                    _vertexData[neighbor] = vNew;
                }
                propagate(currIdx, neighbor);
                pq.push(neighbor);
            }
        }
        
        if(maxDistFound > 1e-6f) {
            _viewScale = 1.0f;
            _viewOffset = ImVec2(0, 0);

            _uvScale = 0.45f / maxDistFound;
            for(const auto& pair : _vertexData) {
                if(pair.second.frozen) {
                    _displayUVs[pair.first] = pair.second.uv * _uvScale + glm::vec2(0.5f, 0.5f);
                }
            }

            const auto& tris = _mesh->triangles();
            std::vector<float> ratios;
            
            for(const auto& t : tris) {
                if (_displayUVs.count(t[0]) && _displayUVs.count(t[1]) && _displayUVs.count(t[2])) {
                    _validTriangles.push_back(t);
                    
                    glm::vec2 uv1 = _displayUVs[t[0]];
                    glm::vec2 uv2 = _displayUVs[t[1]];
                    float dUV = glm::distance(uv1, uv2) / _uvScale;
                    
                    glm::vec3 p1 = toGlm(_mesh->vertices()[t[0]]);
                    glm::vec3 p2 = toGlm(_mesh->vertices()[t[1]]);
                    float d3D = glm::distance(p1, p2);

                    if (d3D > 1e-5f) ratios.push_back(dUV / d3D);
                }
            }

            if (!ratios.empty()) {
                float sum = std::accumulate(ratios.begin(), ratios.end(), 0.0f);
                float mean = sum / ratios.size();
                _autoThreshold = std::max(1.5f, mean * 2.5f); 
            } else {
                _autoThreshold = 3.0f;
            }

            ProjectAllPointClouds(target, radius * 1.2f);
        }
    }

    void RenderUI() {
        ImGui::SetNextWindowSize(ImVec2(400, 450), ImGuiCond_FirstUseEver);
        if(ImGui::Begin("ExpMap UV Result")) {
            
            if (_displayUVs.empty()) {
                ImGui::TextColored(ImVec4(1,1,0,1), "Right-click on the mesh to compute UV.");
                ImGui::End();
                return;
            }

            ImGui::Text("Tris: %lu | App: %lu | Geo: %lu", _validTriangles.size(), _projectedAppPoints.size(), _projectedGeoPoints.size());
            ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), "Hold CTRL + Left Click to pick Start/End Points");
            
            if (_dijkstraPath.size() > 0) {
                ImGui::TextColored(ImVec4(0,1,0,1), "Path Length: %lu hops", _dijkstraPath.size());
            }

            ImGui::SameLine();
            if (ImGui::Button("Reset View")) { _viewScale = 1.0f; _viewOffset = ImVec2(0,0); }
            
            ImGui::Checkbox("Fill", &_drawFilled);
            ImGui::SameLine();
            ImGui::Checkbox("Cull", &_cullBackFace);
            
            // [修改] 新增 App 與 Geo 點雲顯示開關
            ImGui::SameLine();
            ImGui::Checkbox("App Pts", &_showAppPoints);
            ImGui::SameLine();
            ImGui::Checkbox("Geo Pts", &_showGeoPoints);

            ImVec2 p = ImGui::GetCursorScreenPos();
            ImVec2 sz = ImGui::GetContentRegionAvail();
            if(sz.x < 50) sz.x = 50; 
            if(sz.y < 50) sz.y = 50;
            float dim = std::min(sz.x, sz.y);
            
            // Mouse Interaction
            bool isHovered = ImGui::IsWindowHovered();
            ImVec2 mousePos = ImGui::GetMousePos();

            // Zoom & Pan
            if (isHovered) {
                float wheel = ImGui::GetIO().MouseWheel;
                if (wheel != 0.0f) {
                    float zoomFactor = 1.1f;
                    if (wheel < 0.0f) _viewScale /= zoomFactor;
                    else              _viewScale *= zoomFactor;
                }
                if (ImGui::IsMouseDragging(2) || ImGui::IsMouseDragging(1)) {
                    ImVec2 delta = ImGui::GetIO().MouseDelta;
                    _viewOffset.x += delta.x;
                    _viewOffset.y += delta.y;
                }
            }

            // [Dijkstra Picking Logic]
            if (isHovered && ImGui::IsMouseClicked(0) && ImGui::GetIO().KeyCtrl) {
                float lx = mousePos.x - (p.x + sz.x * 0.5f + _viewOffset.x);
                float ly = mousePos.y - (p.y + sz.y * 0.5f + _viewOffset.y);
                float uvX = (lx / (dim * _viewScale)) + 0.5f;
                float uvY = 1.0f - ((ly / (dim * _viewScale)) + 0.5f);

                int closestID = -1;
                float minUVDist = 1e9f;
                
                for (auto const& [id, uv] : _displayUVs) {
                    float d = glm::distance(glm::vec2(uvX, uvY), uv);
                    if (d < minUVDist) {
                        minUVDist = d;
                        closestID = id;
                    }
                }

                if (closestID != -1 && minUVDist < 0.05f / _viewScale) {
                    if (_startNode == -1) {
                        _startNode = closestID;
                        _dijkstraPath.clear();
                    } else if (_endNode == -1) {
                        _endNode = closestID;
                        ComputeDijkstra(_startNode, _endNode);
                    } else {
                        _startNode = closestID;
                        _endNode = -1;
                        _dijkstraPath.clear();
                    }
                }
            }

            ImDrawList* dl = ImGui::GetWindowDrawList();
            dl->PushClipRect(p, ImVec2(p.x + sz.x, p.y + sz.y), true);
            dl->AddRectFilled(p, ImVec2(p.x + sz.x, p.y + sz.y), IM_COL32(40, 40, 40, 255));
            
            auto TransformUV = [&](const glm::vec2& uv) -> ImVec2 {
                float lx = (uv.x - 0.5f) * dim * _viewScale;
                float ly = (1.0f - uv.y - 0.5f) * dim * _viewScale; 
                return ImVec2(p.x + sz.x*0.5f + lx + _viewOffset.x, p.y + sz.y*0.5f + ly + _viewOffset.y);
            };

            // 1. 畫網格
            for(const auto& t : _validTriangles) {
                glm::vec2 uv1 = _displayUVs[t[0]];
                glm::vec2 uv2 = _displayUVs[t[1]];
                glm::vec2 uv3 = _displayUVs[t[2]];

                ImVec2 ip1 = TransformUV(uv1);
                ImVec2 ip2 = TransformUV(uv2);
                ImVec2 ip3 = TransformUV(uv3);

                if ((ip1.x < p.x && ip2.x < p.x && ip3.x < p.x) || (ip1.x > p.x+sz.x && ip2.x > p.x+sz.x)) continue;

                if (_cullBackFace) {
                    float area = (ip2.x - ip1.x) * (ip3.y - ip1.y) - (ip3.x - ip1.x) * (ip2.y - ip1.y);
                    if (area > 0.0f) continue;
                }
                
                float dUV = glm::distance(uv1, uv2) / _uvScale;
                float d3D = glm::distance(toGlm(_mesh->vertices()[t[0]]), toGlm(_mesh->vertices()[t[1]]));
                if (d3D > 1e-6f && (dUV / d3D) > _autoThreshold) continue;

                if (_drawFilled) {
                    dl->AddTriangleFilled(ip1, ip2, ip3, IM_COL32(255, 215, 0, 80));
                    dl->AddTriangle(ip1, ip2, ip3, IM_COL32(255, 255, 0, 150), 1.0f);
                } else {
                    dl->AddTriangle(ip1, ip2, ip3, IM_COL32(255, 215, 0, 200), 1.0f);
                }
            }

            // 2. 畫投影點雲 (加入開關判斷)
            // [修改] 只有當 _showAppPoints 為真時才繪製
            if (_showAppPoints) {
                for (const auto& pt : _projectedAppPoints) {
                    ImVec2 pos = TransformUV(pt);
                    if (pos.x >= p.x && pos.x <= p.x + sz.x && pos.y >= p.y && pos.y <= p.y + sz.y)
                        dl->AddCircleFilled(pos, 3.0f, IM_COL32(0, 255, 255, 255)); 
                }
            }

            // [修改] 只有當 _showGeoPoints 為真時才繪製
            if (_showGeoPoints) {
                for (const auto& pt : _projectedGeoPoints) {
                    ImVec2 pos = TransformUV(pt);
                    if (pos.x >= p.x && pos.x <= p.x + sz.x && pos.y >= p.y && pos.y <= p.y + sz.y)
                        dl->AddCircleFilled(pos, 3.0f, IM_COL32(255, 50, 50, 255)); 
                }
            }

            // 3. [Dijkstra] 繪製選取的點與路徑
            if (_startNode != -1 && _displayUVs.count(_startNode)) {
                dl->AddCircleFilled(TransformUV(_displayUVs[_startNode]), 6.0f, IM_COL32(0, 255, 0, 255)); 
            }
            if (_endNode != -1 && _displayUVs.count(_endNode)) {
                dl->AddCircleFilled(TransformUV(_displayUVs[_endNode]), 6.0f, IM_COL32(255, 0, 0, 255)); 
            }

            if (_dijkstraPath.size() > 1) {
                for (size_t i = 0; i < _dijkstraPath.size() - 1; ++i) {
                    int idA = _dijkstraPath[i];
                    int idB = _dijkstraPath[i+1];
                    if (_displayUVs.count(idA) && _displayUVs.count(idB)) {
                        dl->AddLine(TransformUV(_displayUVs[idA]), TransformUV(_displayUVs[idB]), IM_COL32(0, 255, 0, 255), 3.0f);
                    }
                }
            }

            dl->PopClipRect();
        }
        ImGui::End();
    }

private:
    void buildAdjacency() {
        _adj.clear();
        _adj.resize(_mesh->vertices().size());
        const auto& tris = _mesh->triangles();
        for(const auto& t : tris) {
            auto addE = [&](int a, int b){ 
                if(std::find(_adj[a].begin(), _adj[a].end(), b) == _adj[a].end()) _adj[a].push_back(b);
                if(std::find(_adj[b].begin(), _adj[b].end(), a) == _adj[b].end()) _adj[b].push_back(a);
            };
            addE(t.x(), t.y()); addE(t.y(), t.z()); addE(t.z(), t.x());
        }
    }

    void propagate(int parentIdx, int currIdx) {
        ExpVertex& parent = _vertexData[parentIdx];
        ExpVertex& curr = _vertexData[currIdx];

        glm::vec3 parentPos = toGlm(_mesh->vertices()[parentIdx]);
        glm::vec3 currPos = toGlm(_mesh->vertices()[currIdx]);
        glm::vec3 parentN = (_mesh->normals().size() > parentIdx) ? toGlm(_mesh->normals()[parentIdx]) : glm::vec3(0,1,0);
        
        TangentFrame parentFrame(parentPos, parentN);
        TangentFrame alignedSeedFrame = _seedFrame;
        alignedSeedFrame.alignZAxis(parentFrame);

        glm::vec3 parentX = parentFrame.axes[0];
        glm::vec3 seedX = alignedSeedFrame.axes[0];
        float cosTheta = glm::clamp(glm::dot(parentX, seedX), -1.0f, 1.0f);
        float sinTheta = std::sqrt(std::max(0.0f, 1.0f - cosTheta * cosTheta));
        if (glm::dot(glm::cross(parentX, seedX), parentFrame.axes[2]) < 0) {
            sinTheta = -sinTheta;
        }

        glm::mat2 rot(cosTheta, -sinTheta, sinTheta, cosTheta);

        glm::vec3 vec = currPos - parentPos;
        float dist = glm::length(vec);
        glm::vec3 pNorm = parentFrame.axes[2];
        glm::vec3 vecOnPlane = vec - pNorm * glm::dot(vec, pNorm);
        if (glm::length(vecOnPlane) > 1e-6f) {
            vecOnPlane = glm::normalize(vecOnPlane) * dist;
        } else {
            vecOnPlane = glm::vec3(0,0,0);
        }

        glm::vec3 localVec3 = parentFrame.toLocal(vecOnPlane);
        glm::vec2 localVec2(localVec3.x, localVec3.y);
        localVec2 *= -1.0f; 

        glm::vec2 newUV = parent.uv + (rot * localVec2);
        float newDist = parent.dist + dist;

        if(newDist < curr.dist) {
            curr.dist = newDist;
            curr.uv = newUV;
            curr.parentId = parentIdx;
        }
    }

    void ProjectAllPointClouds(const glm::vec3& center, float radius) {
        ProcessCloudProjection(_appCloudData, _projectedAppPoints, center, radius);
        ProcessCloudProjection(_geoCloudData, _projectedGeoPoints, center, radius);
    }

    void ProcessCloudProjection(const std::vector<float>& cloudData, std::vector<glm::vec2>& outPoints, const glm::vec3& center, float radius) {
        if (cloudData.empty()) return;

        size_t numPoints = cloudData.size() / 3;
        float r2 = radius * radius;

        for (size_t i = 0; i < numPoints; ++i) {
            glm::vec3 pt(cloudData[i*3], cloudData[i*3+1], cloudData[i*3+2]);
            
            glm::vec3 diff = pt - center;
            if (glm::dot(diff, diff) > r2) continue;

            float bestDist = 1e9f;
            glm::vec2 bestUV(0,0);
            bool found = false;

            for (const auto& t : _validTriangles) {
                glm::vec3 v0 = toGlm(_mesh->vertices()[t[0]]);
                glm::vec3 v1 = toGlm(_mesh->vertices()[t[1]]);
                glm::vec3 v2 = toGlm(_mesh->vertices()[t[2]]);

                glm::vec3 normal = glm::normalize(glm::cross(v1 - v0, v2 - v0));
                float distPlane = std::abs(glm::dot(pt - v0, normal));
                
                if (distPlane > 0.05f) continue; 

                glm::vec3 pProj = pt - normal * glm::dot(pt - v0, normal);
                
                glm::vec3 v0v1 = v1 - v0; glm::vec3 v0v2 = v2 - v0; glm::vec3 pVec = pProj - v0;
                float d00 = glm::dot(v0v1, v0v1); float d01 = glm::dot(v0v1, v0v2);
                float d11 = glm::dot(v0v2, v0v2); float d20 = glm::dot(pVec, v0v1);
                float d21 = glm::dot(pVec, v0v2);
                float denom = d00 * d11 - d01 * d01;
                
                if (std::abs(denom) < 1e-6f) continue;
                
                float v = (d11 * d20 - d01 * d21) / denom;
                float w = (d00 * d21 - d01 * d20) / denom;
                float u = 1.0f - v - w;

                if (u >= -0.1f && v >= -0.1f && w >= -0.1f && u <= 1.1f && v <= 1.1f && w <= 1.1f) {
                    glm::vec2 uv0 = _displayUVs[t[0]];
                    glm::vec2 uv1 = _displayUVs[t[1]];
                    glm::vec2 uv2 = _displayUVs[t[2]];
                    
                    glm::vec2 finalUV = u * uv0 + v * uv1 + w * uv2;
                    float d = glm::distance(pt, pProj); 
                    
                    if (d < bestDist) {
                        bestDist = d;
                        bestUV = finalUV;
                        found = true;
                    }
                }
            }

            if (found) {
                outPoints.push_back(bestUV);
            }
        }
    }

    void ComputeDijkstra(int startIdx, int endIdx) {
        _dijkstraPath.clear();
        if (startIdx == -1 || endIdx == -1) return;
        if (!_displayUVs.count(startIdx) || !_displayUVs.count(endIdx)) return;

        std::map<int, float> minDist;
        std::map<int, int> prev;
        
        for (auto const& [id, uv] : _displayUVs) {
            minDist[id] = std::numeric_limits<float>::max();
        }

        minDist[startIdx] = 0.0f;
        
        using PElement = std::pair<float, int>;
        std::priority_queue<PElement, std::vector<PElement>, std::greater<PElement>> pq;
        pq.push({0.0f, startIdx});

        const auto& verts = _mesh->vertices();

        while (!pq.empty()) {
            float d = pq.top().first;
            int u = pq.top().second;
            pq.pop();

            if (u == endIdx) break; 
            if (d > minDist[u]) continue;

            for (int v : _adj[u]) {
                if (minDist.find(v) == minDist.end()) continue;

                float weight = glm::distance(toGlm(verts[u]), toGlm(verts[v]));
                
                if (minDist[u] + weight < minDist[v]) {
                    minDist[v] = minDist[u] + weight;
                    prev[v] = u;
                    pq.push({minDist[v], v});
                }
            }
        }

        if (prev.find(endIdx) != prev.end()) {
            int curr = endIdx;
            while (curr != startIdx) {
                _dijkstraPath.push_back(curr);
                curr = prev[curr];
            }
            _dijkstraPath.push_back(startIdx);
            std::reverse(_dijkstraPath.begin(), _dijkstraPath.end());
        }
    }

    const sibr::Mesh* _mesh = nullptr;
    std::vector<std::vector<int>> _adj;
    std::map<int, ExpVertex> _vertexData;
    TangentFrame _seedFrame;
    std::map<int, glm::vec2> _displayUVs;
    std::vector<sibr::Vector3u> _validTriangles;
    float _uvScale = 1.0f;
    float _autoThreshold = 2.5f;

    float _viewScale = 1.0f;
    ImVec2 _viewOffset = {0.0f, 0.0f};
    bool _cullBackFace = true;
    bool _drawFilled = false;
    
    // [修改] 新增點雲顯示控制變數
    bool _showAppPoints = true;
    bool _showGeoPoints = true;

    std::vector<float> _appCloudData; 
    std::vector<glm::vec2> _projectedAppPoints;

    std::vector<float> _geoCloudData; 
    std::vector<glm::vec2> _projectedGeoPoints;

    int _startNode = -1;
    int _endNode = -1;
    std::vector<int> _dijkstraPath;
};

#endif