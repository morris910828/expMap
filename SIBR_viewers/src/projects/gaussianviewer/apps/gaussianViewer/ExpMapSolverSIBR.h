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
#include <limits>
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
        float sinAngle = glm::length(axis);
        float cosAngle = glm::dot(fromZ, toZ);

        if (sinAngle < 1e-6f) {
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
    float cost = 1e9f;
    glm::vec2 uv = {0,0};
    glm::vec3 tangentX = {0, 0, 0}; 
    bool frozen = false;
};

// ==========================================
// 2. ExpMap Solver
// ==========================================
class ExpMapSolverSIBR {
public:
    void Init(const sibr::Mesh* mesh) {
        _mesh = mesh;
        buildBaseAdjacency();
    }

    void RegisterAppPointCloud(const std::vector<float>& pointData, const std::vector<int>& fids) { 
        _appCloudData = pointData; 
        _appCloudFids = fids;
    }
    void RegisterGeoPointCloud(const std::vector<float>& pointData) { _geoCloudData = pointData; }

    const std::vector<sibr::Vector3u>& GetActiveTriangles() const { return _validTriangles; }

    void Compute(const sibr::Vector3f& hitPos, const sibr::Vector3f& hitNormal, float radius) {
        if(!_mesh) return;

        // Reset
        _vertexData.clear();
        _displayUVs.clear();
        _validTriangles.clear();
        _validTriIDs.clear();
        _validTriangleIndicesSet.clear();
        _projectedAppPoints.clear(); 
        _projectedGeoPoints.clear();
        _extraNodePositions.clear(); 
        _extraNodeNormals.clear(); // [New] 清空虛擬法線
        
        _adj = _baseAdj; 
        _startNode = -1; _endNode = -1;
        _dijkstraPath.clear();

        // 1. ExpMap Core
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

        auto comp = [&](int a, int b){ return _vertexData[a].cost > _vertexData[b].cost; };
        std::priority_queue<int, std::vector<int>, decltype(comp)> pq(comp);

        glm::vec3 seedN = (normals.size() > seedIdx) ? toGlm(normals[seedIdx]) : toGlm(hitNormal);
        _seedFrame = TangentFrame(toGlm(verts[seedIdx]), seedN);

        ExpVertex vSeed;
        vSeed.id = seedIdx;
        vSeed.cost = 0.0f;
        vSeed.uv = {0.0f, 0.0f};
        vSeed.tangentX = _seedFrame.axes[0]; 

        _vertexData[seedIdx] = vSeed;
        pq.push(seedIdx);

        float maxCostFound = 0.0f;

        while(!pq.empty()) {
            int currIdx = pq.top(); pq.pop();
            if(_vertexData[currIdx].frozen) continue;
            _vertexData[currIdx].frozen = true;

            if(glm::length(_vertexData[currIdx].uv) > radius) continue;

            maxCostFound = std::max(maxCostFound, glm::length(_vertexData[currIdx].uv));

            for(int neighbor : _adj[currIdx]) {
                if(_vertexData.find(neighbor) != _vertexData.end() && _vertexData[neighbor].frozen) continue;

                if(_vertexData.find(neighbor) == _vertexData.end()) {
                    ExpVertex vNew; vNew.id = neighbor; vNew.cost = 1e9f;
                    _vertexData[neighbor] = vNew;
                }
                propagate(currIdx, neighbor);
                pq.push(neighbor);
            }
        }
        
        if(maxCostFound > 1e-6f) {
            _viewScale = 1.0f;
            _viewOffset = ImVec2(0, 0);

            _uvScale = 0.45f / maxCostFound;
            for(const auto& pair : _vertexData) {
                if(pair.second.frozen) {
                    _displayUVs[pair.first] = pair.second.uv * _uvScale + glm::vec2(0.5f, 0.5f);
                }
            }

            const auto& tris = _mesh->triangles();
            std::vector<float> ratios;
            
            for(size_t i = 0; i < tris.size(); ++i) {
                const auto& t = tris[i];
                if (!_displayUVs.count(t[0]) || !_displayUVs.count(t[1]) || !_displayUVs.count(t[2])) continue;

                glm::vec2 uv0 = _displayUVs[t[0]];
                glm::vec2 uv1 = _displayUVs[t[1]];
                glm::vec2 uv2 = _displayUVs[t[2]];
                
                glm::vec3 p0 = toGlm(_mesh->vertices()[t[0]]);
                glm::vec3 p1 = toGlm(_mesh->vertices()[t[1]]);
                glm::vec3 p2 = toGlm(_mesh->vertices()[t[2]]);

                float signedAreaUV = (uv1.x - uv0.x) * (uv2.y - uv0.y) - (uv1.y - uv0.y) * (uv2.x - uv0.x);
                if (signedAreaUV < 1e-7f) continue;

                float dUV_01 = glm::distance(uv0, uv1) / _uvScale; float d3D_01 = glm::distance(p0, p1);
                float dUV_12 = glm::distance(uv1, uv2) / _uvScale; float d3D_12 = glm::distance(p1, p2);
                float dUV_20 = glm::distance(uv2, uv0) / _uvScale; float d3D_20 = glm::distance(p2, p0);
                
                float maxRatio = 2.0f; 
                float minRatio = 0.5f;

                bool distorted = false;
                auto checkEdge = [&](float duv, float d3d) {
                    if (d3d > 1e-6f) {
                        float r = duv / d3d;
                        if (r > maxRatio || r < minRatio) return true;
                    } else if (duv > 1e-4f) { return true; }
                    return false;
                };

                if (checkEdge(dUV_01, d3D_01)) distorted = true;
                if (checkEdge(dUV_12, d3D_12)) distorted = true;
                if (checkEdge(dUV_20, d3D_20)) distorted = true;

                if (distorted) continue;

                _validTriangles.push_back(t);
                _validTriIDs.push_back((int)i); 
                _validTriangleIndicesSet.insert((int)i);
                
                if (d3D_01 > 1e-5f) ratios.push_back(dUV_01 / d3D_01);
            }

            if (!ratios.empty()) {
                float sum = std::accumulate(ratios.begin(), ratios.end(), 0.0f);
                float mean = sum / ratios.size();
                _autoThreshold = std::max(1.5f, mean * 2.5f); 
            } else {
                _autoThreshold = 3.0f;
            }

            ProjectAndInsertClouds(target, radius * 1.2f);
            
            // [關鍵新增] 為虛擬節點計算真實的測地線成本
            ComputeExtraNodeCosts();
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
            
            // [V3 新增] 點顯示控制
            ImGui::Checkbox("Show App Points", &_showAppPoints);
            ImGui::SameLine();
            ImGui::Checkbox("Show Geo Points", &_showGeoPoints);

            ImVec2 p = ImGui::GetCursorScreenPos();
            ImVec2 sz = ImGui::GetContentRegionAvail();
            if(sz.x < 50) sz.x = 50; 
            if(sz.y < 50) sz.y = 50;
            float dim = std::min(sz.x, sz.y);
            
            bool isHovered = ImGui::IsWindowHovered();
            ImVec2 mousePos = ImGui::GetMousePos();

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

            if (isHovered && ImGui::IsMouseClicked(0) && ImGui::GetIO().KeyCtrl) {
                float lx = mousePos.x - (p.x + sz.x * 0.5f + _viewOffset.x);
                float ly = mousePos.y - (p.y + sz.y * 0.5f + _viewOffset.y);
                float uvX = (lx / (dim * _viewScale)) + 0.5f;
                float uvY = 1.0f - ((ly / (dim * _viewScale)) + 0.5f); 
                glm::vec2 mouseUV(uvX, uvY);

                int closestID = -1;
                float minPickDist = 0.05f / _viewScale; 
                size_t meshSize = _mesh->vertices().size();

                for (auto const& [id, uv] : _displayUVs) {
                    float d = glm::distance(mouseUV, uv);
                    float priority = (id >= (int)meshSize) ? 0.5f : 1.0f;
                    if (d * priority < minPickDist) {
                        minPickDist = d * priority;
                        closestID = id;
                    }
                }

                if (closestID != -1) {
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

            for(const auto& t : _validTriangles) {
                glm::vec2 uv1 = _displayUVs[t[0]];
                glm::vec2 uv2 = _displayUVs[t[1]];
                glm::vec2 uv3 = _displayUVs[t[2]];
                ImVec2 ip1 = TransformUV(uv1); ImVec2 ip2 = TransformUV(uv2); ImVec2 ip3 = TransformUV(uv3);

                if ((ip1.x < p.x && ip2.x < p.x && ip3.x < p.x) || (ip1.x > p.x+sz.x && ip2.x > p.x+sz.x)) continue;
                if (_cullBackFace) {
                    float area = (ip2.x - ip1.x) * (ip3.y - ip1.y) - (ip3.x - ip1.x) * (ip2.y - ip1.y);
                    if (area > 0.0f) continue;
                }
                float dUV = glm::distance(uv1, uv2) / _uvScale;
                float d3D = glm::distance(toGlm(_mesh->vertices()[t[0]]), toGlm(_mesh->vertices()[t[1]]));
                if (d3D > 1e-6f && (dUV / d3D) > _autoThreshold) continue;

                if (_drawFilled) dl->AddTriangleFilled(ip1, ip2, ip3, IM_COL32(255, 215, 0, 80));
                else dl->AddTriangle(ip1, ip2, ip3, IM_COL32(255, 215, 0, 200), 1.0f);
            }

            size_t meshSize = _mesh->vertices().size();
            for (auto const& [id, uv] : _displayUVs) {
                if (id < (int)meshSize) continue; 
                ImVec2 pos = TransformUV(uv);
                bool isApp = (id < (int)(meshSize + _projectedAppPoints.size()));
                
                // [V3 修改] 根據開關顯示點
                if (isApp && _showAppPoints) {
                    dl->AddCircleFilled(pos, 3.0f, IM_COL32(0, 255, 255, 255)); 
                } else if (!isApp && _showGeoPoints) {
                    dl->AddCircleFilled(pos, 3.0f, IM_COL32(255, 50, 50, 255)); 
                }
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

            if (_startNode != -1 && _displayUVs.count(_startNode)) dl->AddCircleFilled(TransformUV(_displayUVs[_startNode]), 5.0f, IM_COL32(0, 255, 0, 255)); 
            if (_endNode != -1 && _displayUVs.count(_endNode)) dl->AddCircleFilled(TransformUV(_displayUVs[_endNode]), 5.0f, IM_COL32(255, 0, 0, 255)); 

            dl->PopClipRect();
        }
        ImGui::End();
    }

private:
    void buildBaseAdjacency() {
        _baseAdj.clear();
        _baseAdj.resize(_mesh->vertices().size());
        const auto& tris = _mesh->triangles();
        for(const auto& t : tris) {
            auto addE = [&](int a, int b){ 
                if(std::find(_baseAdj[a].begin(), _baseAdj[a].end(), b) == _baseAdj[a].end()) _baseAdj[a].push_back(b);
                if(std::find(_baseAdj[b].begin(), _baseAdj[b].end(), a) == _baseAdj[b].end()) _baseAdj[b].push_back(a);
            };
            addE(t.x(), t.y()); addE(t.y(), t.z()); addE(t.z(), t.x());
        }
    }

    void propagate(int parentIdx, int currIdx) {
        ExpVertex& parent = _vertexData[parentIdx];
        ExpVertex& curr = _vertexData[currIdx];

        glm::vec3 parentPos = getPos(parentIdx);
        glm::vec3 currPos = getPos(currIdx);
        
        float dist = glm::distance(currPos, parentPos);
        float weightedDist = dist;

        // [修改] 對所有擴展 (包括 Mesh 內部和邊緣) 都應用曲率檢查
        glm::vec3 parentN = getNormal(parentIdx); // [New] Helper function
        glm::vec3 currN = getNormal(currIdx);
        
        float dotNormal = glm::dot(parentN, currN);
        if (dotNormal < 0.5f) return; // 角度過大則停止 (防止穿透尖銳邊緣)

        float curvaturePenalty = 1.0f + 5.0f * (1.0f - dotNormal);
        weightedDist = dist * curvaturePenalty;

        float newCost = parent.cost + weightedDist;

        if(newCost < curr.cost) {
            curr.cost = newCost;
            curr.parentId = parentIdx;
            
            // UV 傳播邏輯 (僅對 Mesh 頂點)
            bool isExtraNode = (parentIdx >= _mesh->vertices().size()) || (currIdx >= _mesh->vertices().size());
            if (!isExtraNode) {
                glm::vec3 parentN = (_mesh->normals().size() > parentIdx) ? toGlm(_mesh->normals()[parentIdx]) : glm::vec3(0,1,0);
                glm::vec3 currN = (_mesh->normals().size() > currIdx) ? toGlm(_mesh->normals()[currIdx]) : glm::vec3(0,1,0);
                glm::vec3 edge = currPos - parentPos;
                glm::vec3 edgeProj = edge - parentN * glm::dot(edge, parentN);
                if (glm::length(edgeProj) > 1e-6f) edgeProj = glm::normalize(edgeProj) * dist; else edgeProj = glm::vec3(0);
                
                glm::vec3 parentTangentY = glm::cross(parentN, parent.tangentX);
                float dU = glm::dot(edgeProj, parent.tangentX);
                float dV = glm::dot(edgeProj, parentTangentY);
                curr.uv = parent.uv + glm::vec2(dU, dV);

                glm::vec3 axis = glm::cross(parentN, currN);
                float cosA = glm::dot(parentN, currN);
                glm::vec3 newTangentX;
                if (glm::length(axis) < 1e-6f) newTangentX = (cosA < 0) ? -parent.tangentX : parent.tangentX;
                else {
                    axis = glm::normalize(axis);
                    float angle = std::acos(glm::clamp(cosA, -1.0f, 1.0f));
                    glm::mat4 rotMat = glm::rotate(glm::mat4(1.0f), angle, axis);
                    newTangentX = glm::vec3(rotMat * glm::vec4(parent.tangentX, 0.0f));
                }
                curr.tangentX = glm::normalize(newTangentX - currN * glm::dot(newTangentX, currN));
            }
        }
    }

    void ProjectAndInsertClouds(const glm::vec3& center, float radius) {
        std::map<int, std::vector<int>> triToExtraNodes;

        auto process = [&](const std::vector<float>& cloud, const std::vector<int>& fids, std::vector<glm::vec2>& outPts) {
             if (cloud.empty()) return;
             size_t numPoints = cloud.size() / 3;
             float r2 = radius * radius;
             size_t meshSize = _mesh->vertices().size();
             bool use_fids = !fids.empty() && fids.size() == numPoints;

             for (size_t i = 0; i < numPoints; ++i) {
                glm::vec3 pt(cloud[i*3], cloud[i*3+1], cloud[i*3+2]);
                if (glm::dot(pt - center, pt - center) > r2) continue;

                float bestDist = 1e9f; glm::vec2 bestUV(0,0); int bestTriGlobalID = -1; bool found = false;
                glm::vec3 bestProjPos(0,0,0);
                glm::vec3 bestProjNormal(0,1,0); // [New] 儲存投影點法線

                if (use_fids) {
                    int face_id = fids[i];
                    if (_validTriangleIndicesSet.count(face_id)) {
                        const auto& t = _mesh->triangles()[face_id];
                        glm::vec3 v0 = toGlm(_mesh->vertices()[t[0]]);
                        glm::vec3 v1 = toGlm(_mesh->vertices()[t[1]]);
                        glm::vec3 v2 = toGlm(_mesh->vertices()[t[2]]);
                        glm::vec3 pVec = pt - v0;
                        glm::vec3 v0v1 = v1 - v0; glm::vec3 v0v2 = v2 - v0;
                        float d00 = glm::dot(v0v1, v0v1); float d01 = glm::dot(v0v1, v0v2); float d11 = glm::dot(v0v2, v0v2);
                        float d20 = glm::dot(pVec, v0v1); float d21 = glm::dot(pVec, v0v2); float denom = d00 * d11 - d01 * d01;
                        if (std::abs(denom) > 1e-9f) {
                            float v = (d11 * d20 - d01 * d21) / denom; float w = (d00 * d21 - d01 * d20) / denom; float u = 1.0f - v - w;
                            bestUV = u * _displayUVs.at(t[0]) + v * _displayUVs.at(t[1]) + w * _displayUVs.at(t[2]);
                            bestProjPos = u * v0 + v * v1 + w * v2; 
                            
                            // [New] Interpolate Normal
                            glm::vec3 n0 = toGlm(_mesh->normals()[t[0]]);
                            glm::vec3 n1 = toGlm(_mesh->normals()[t[1]]);
                            glm::vec3 n2 = toGlm(_mesh->normals()[t[2]]);
                            bestProjNormal = glm::normalize(u * n0 + v * n1 + w * n2);

                            bestTriGlobalID = face_id; found = true;
                        }
                    }
                } else {
                    for (int globalTIdx : _validTriIDs) {
                        const auto& t = _mesh->triangles()[globalTIdx];
                        glm::vec3 v0 = toGlm(_mesh->vertices()[t[0]]);
                        glm::vec3 v1 = toGlm(_mesh->vertices()[t[1]]);
                        glm::vec3 v2 = toGlm(_mesh->vertices()[t[2]]);
                        glm::vec3 n = glm::cross(v1 - v0, v2 - v0); if (glm::length(n) < 1e-9) continue; n = glm::normalize(n);
                        if (std::abs(glm::dot(pt - v0, n)) > 0.1f * radius) continue; 
                        glm::vec3 pProj = pt - n * glm::dot(pt - v0, n);
                        
                        glm::vec3 v0v1 = v1 - v0; glm::vec3 v0v2 = v2 - v0; glm::vec3 pVec = pProj - v0;
                        float d00 = glm::dot(v0v1, v0v1); float d01 = glm::dot(v0v1, v0v2); float d11 = glm::dot(v0v2, v0v2);
                        float d20 = glm::dot(pVec, v0v1); float d21 = glm::dot(pVec, v0v2); float denom = d00 * d11 - d01 * d01;
                        if (std::abs(denom) > 1e-9f) {
                            float v = (d11 * d20 - d01 * d21) / denom; float w = (d00 * d21 - d01 * d20) / denom; float u = 1.0f - v - w;
                            if (u >= -0.01f && v >= -0.01f && w >= -0.01f && u <= 1.01f && v <= 1.01f && w <= 1.01f) {
                                float dist = glm::distance(pt, pProj);
                                if (dist < bestDist) { 
                                    bestDist = dist; 
                                    bestUV = u * _displayUVs.at(t[0]) + v * _displayUVs.at(t[1]) + w * _displayUVs.at(t[2]); 
                                    bestProjPos = pProj; bestTriGlobalID = globalTIdx; found = true; 
                                    
                                    // [New] Interpolate Normal
                                    glm::vec3 n0 = toGlm(_mesh->normals()[t[0]]);
                                    glm::vec3 n1 = toGlm(_mesh->normals()[t[1]]);
                                    glm::vec3 n2 = toGlm(_mesh->normals()[t[2]]);
                                    bestProjNormal = glm::normalize(u * n0 + v * n1 + w * n2);
                                }
                            }
                        }
                    }
                }

                if (found) {
                    outPts.push_back(bestUV);
                    int newNodeID = (int)meshSize + (int)_extraNodePositions.size();
                    _extraNodePositions.push_back(bestProjPos);
                    _extraNodeNormals.push_back(bestProjNormal); // [New] Store normal
                    _displayUVs[newNodeID] = bestUV;
                    
                    // [關鍵新增] 為虛擬節點創建 ExpVertex
                    ExpVertex vExtra;
                    vExtra.id = newNodeID;
                    vExtra.cost = 1e9f;  // 初始化為高成本，稍後計算真實值
                    vExtra.uv = bestUV;
                    vExtra.frozen = false;
                    _vertexData[newNodeID] = vExtra;
                    
                    if (_adj.size() <= newNodeID) _adj.resize(newNodeID + 1);

                    const auto& t = _mesh->triangles()[bestTriGlobalID];
                    int v[3] = { (int)t[0], (int)t[1], (int)t[2] };
                    
                    // [修正] 只連接最近的 2 個頂點
                    std:: vector<std::pair<float, int>> distVerts;
                    for(int k=0; k<3; ++k) {
                        float d = glm::distance(bestProjPos, toGlm(_mesh->vertices()[v[k]]));
                        distVerts.push_back({d, v[k]});
                    }
                    std::sort(distVerts.begin(), distVerts.end());
                    
                    // [修正] 只連接最近的 2 個頂點，且距離不能太遠
                    float maxConnectionDist = 0.15f; // 可調整
                    int connected = 0;
                    for(int k=0; k<3 && connected < 2; ++k) {
                        if (distVerts[k].first > maxConnectionDist) continue;
                        int vertID = distVerts[k].second;
                        _adj[newNodeID].push_back(vertID);
                        _adj[vertID].push_back(newNodeID);
                        connected++;
                    }

                    // [關鍵修正V3] 移除同一三角形內虛擬節點的互連
                    // 虛擬節點之間不應該有直連，必須通過網格頂點
                    
                    if (triToExtraNodes.count(bestTriGlobalID)) {
                        for (int siblingID : triToExtraNodes[bestTriGlobalID]) {
                            _adj[newNodeID].push_back(siblingID);
                            _adj[siblingID].push_back(newNodeID);
                        }
                    }
                    
                    triToExtraNodes[bestTriGlobalID].push_back(newNodeID);
                }
             }
        };

        process(_appCloudData, _appCloudFids, _projectedAppPoints);
        process(_geoCloudData, {}, _projectedGeoPoints);

        // [關鍵修正V3] 移除共享邊三角形的虛擬節點互連
        // 這是導致路徑跨越三角形的主要原因
        /*
        std::map<std::pair<int, int>, std::vector<int>> edgeToTriangles;
        for (auto const& [globalTIdx, nodes] : triToExtraNodes) {
            const auto& t = _mesh->triangles()[globalTIdx];
            int v[3] = { (int)t[0], (int)t[1], (int)t[2] };
            auto addEdge = [&](int a, int b) { 
                if (a > b) std::swap(a, b); 
                edgeToTriangles[{a, b}].push_back(globalTIdx); 
            };
            addEdge(v[0], v[1]); addEdge(v[1], v[2]); addEdge(v[2], v[0]);
        }

        for (auto const& kv : edgeToTriangles) {
            const auto& tris = kv.second;
            for (size_t i = 0; i < tris.size(); ++i) {
                for (size_t j = i + 1; j < tris.size(); ++j) {
                    int tA = tris[i];
                    int tB = tris[j];
                    for (int nA : triToExtraNodes[tA]) {
                        for (int nB : triToExtraNodes[tB]) {
                             _adj[nA].push_back(nB);
                             _adj[nB].push_back(nA);
                        }
                    }
                }
            }
        }
        */
    }

    glm::vec3 getPos(int id) const {
        if (id < _mesh->vertices().size()) return toGlm(_mesh->vertices()[id]);
        else return _extraNodePositions[id - _mesh->vertices().size()];
    }

    // [New] Helper to get Normal for ANY node
    glm::vec3 getNormal(int id) const {
        if (id < _mesh->vertices().size()) return toGlm(_mesh->normals()[id]);
        else return _extraNodeNormals[id - _mesh->vertices().size()];
    }

    void ComputeExtraNodeCosts() {
        size_t meshSize = _mesh->vertices().size();
        
        // 對每個虛擬節點，從其連接的網格頂點計算最小成本
        for (auto& [nodeID, vData] : _vertexData) {
            if (nodeID < (int)meshSize) continue; // 跳過原始網格頂點
            
            float minCost = 1e9f;
            int bestParent = -1;
            
            // 遍歷所有鄰居（應該是網格頂點）
            if (nodeID < (int)_adj.size()) {
                for (int neighborID : _adj[nodeID]) {
                    if (neighborID >= (int)meshSize) continue; // 只考慮網格頂點
                    if (_vertexData.find(neighborID) == _vertexData.end()) continue;
                    if (!_vertexData[neighborID].frozen) continue;
                    
                    // 計算從鄰居到這個虛擬節點的成本
                    float dist = glm::distance(getPos(nodeID), getPos(neighborID));
                    
                    // 應用曲率懲罰
                    glm::vec3 nodeN = getNormal(nodeID);
                    glm::vec3 neighborN = getNormal(neighborID);
                    float dotNormal = glm::dot(nodeN, neighborN);
                    
                    if (dotNormal < 0.3f) continue; // 角度太大則跳過
                    
                    float penalty = 1.0f + 5.0f * (1.0f - dotNormal);
                    float edgeCost = dist * penalty;
                    float totalCost = _vertexData[neighborID].cost + edgeCost;
                    
                    if (totalCost < minCost) {
                        minCost = totalCost;
                        bestParent = neighborID;
                    }
                }
            }
            
            if (bestParent != -1) {
                vData.cost = minCost;
                vData.parentId = bestParent;
                vData.frozen = true;
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

        while (!pq.empty()) {
            float d = pq.top().first;
            int u = pq.top().second;
            pq.pop();

            if (u == endIdx) break; 
            if (d > minDist[u]) continue;

            if (u >= _adj.size()) continue;

            for (int v : _adj[u]) {
                if (minDist.find(v) == minDist.end()) continue;

                float dist = glm::distance(getPos(u), getPos(v));
                
                // [關鍵修正] 在 Dijkstra 路徑搜尋時也應用曲率懲罰
                // 這能防止路徑 "鑽地" (穿透法線差異大的面)
                glm::vec3 uN = getNormal(u);
                glm::vec3 vN = getNormal(v);
                float dotNormal = glm::dot(uN, vN);
                
                // [修正] 更嚴格的角度閾值:只允許 < 60° (cos 60° = 0.5)
                if (dotNormal < 0.85f) continue; 

                // [新增] 檢查邊長異常(可能是錯誤連接)
                float maxEdgeLength = 0.25f; // 根據場景調整
                if (dist > maxEdgeLength) continue;

                // [修正] 更大的懲罰係數來避免穿透
                float penalty = 1.0f + 20.0f * (1.0f - dotNormal);
                float weight = dist * penalty;
                
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
    std::vector<std::vector<int>> _baseAdj; 
    std::vector<std::vector<int>> _adj;     
    
    std::map<int, ExpVertex> _vertexData;
    TangentFrame _seedFrame;
    std::map<int, glm::vec2> _displayUVs;
    std::vector<sibr::Vector3u> _validTriangles;
    std::vector<int> _validTriIDs;
    std::set<int> _validTriangleIndicesSet;
    
    std::vector<glm::vec3> _extraNodePositions; 
    std::vector<glm::vec3> _extraNodeNormals; // [New]

    float _uvScale = 1.0f;
    float _autoThreshold = 2.5f;
    float _viewScale = 1.0f;
    ImVec2 _viewOffset = {0.0f, 0.0f};
    bool _drawFilled = false;
    bool _cullBackFace = true;
    bool _showAppPoints = true;
    bool _showGeoPoints = true;

    std::vector<float> _appCloudData; 
    std::vector<int> _appCloudFids;
    std::vector<glm::vec2> _projectedAppPoints;

    std::vector<float> _geoCloudData; 
    std::vector<glm::vec2> _projectedGeoPoints;

    int _startNode = -1;
    int _endNode = -1;
    std::vector<int> _dijkstraPath;
};

#endif