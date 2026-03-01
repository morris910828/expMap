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
#include <iostream>
#include <imgui/imgui.h>
#include "texture.h"

inline glm::vec3 toGlm(const sibr::Vector3f& v) { return glm::vec3(v.x(), v.y(), v.z()); }

// ==========================================
// 统计信息结构
// ==========================================
struct ProjectionStats {
    // APP点统计
    int appTotal = 0;
    int appOutOfRadius = 0;
    int appInvalidFid = 0;
    int appFidOutOfRange = 0;
    int appProjected = 0;
    
    // GEO点统计
    int geoTotal = 0;
    int geoOutOfRadius = 0;
    int geoProjected = 0;
};

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
        if (glm::abs(n.x) >= glm::abs(n.y) && glm::abs(n.x) >= glm::abs(n.z)) {
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
    // Public methods
    const std::vector<int>& GetDijkstraPath() const { return _dijkstraPath; }

    
    // Aliases and additional public methods for compatibility
    const std::vector<sibr::Vector3u>& GetActiveTris() const { return _validTriangles; }
    const std::set<int>& GetActiveTriIndices() const { return _validTriangleIndicesSet; }
    
    // NEW: Get triangles within UV selection box
    const std::vector<sibr::Vector3u>& GetSelectedTris() const { return _selectedTriangles; }
    
    // NEW: Public accessors for texture projection
    const std::map<int, glm::vec2>& GetDisplayUVs() const { return _displayUVs; }
    const TextureLoader& GetTextureLoader() const { return _textureLoader; }
    TextureLoader& GetTextureLoader() { return _textureLoader; }

    // ================================================================
    // 获取统计信息
    // ================================================================
    const ProjectionStats& GetProjectionStats() const { return _projectionStats; }

    // ----------------------------------------------------------------
    // Multi-slot API
    // ----------------------------------------------------------------
    int SaveCurrentSlot(const std::string& name = "") {
        if (_validTriangles.empty() || !_textureLoader.getTexture()) return -1;
        TextureSlotData slot;
        slot.triangles   = _validTriangles;
        slot.uvMap       = _displayUVs;
        slot.texture     = _textureLoader.getTexture();
        slot.texturePath = _textureLoader.getPath();
        slot.alpha       = 1.f;
        slot.visible     = true;
        _slots.push_back(std::move(slot));
        return (int)_slots.size() - 1;
    }
    void RemoveSlot(int idx) {
        if (idx >= 0 && idx < (int)_slots.size())
            _slots.erase(_slots.begin() + idx);
    }
    void ClearAllSlots() { _slots.clear(); }
    const std::vector<TextureSlotData>& GetAllSlots() const { return _slots; }
    std::vector<TextureSlotData>&       GetAllSlots()       { return _slots; }
    
    void ClearRaycastState() {
        _validTriangles.clear();
        _validTriIDs.clear();
        _validTriangleIndicesSet.clear();
        _dijkstraPath.clear();
        _startNode = -1;
        _endNode = -1;
        // Clear UV map and projected point clouds so geo/app dots disappear
        _displayUVs.clear();
        _vertexData.clear();
        _projectedAppPoints.clear();
        _projectedGeoPoints.clear();
        _extraNodePositions.clear();
        _extraNodeNormals.clear();
        // Clear selected triangles
        _selectedTriangles.clear();
        _isSelecting = false;
        
        // 清空统计信息
        _projectionStats = ProjectionStats();
    }
    
    void OnRaycastHit(const sibr::Vector3f& hitPos, float radius, int hitTriID) {
        if (!_mesh) return;
        
        // Compute normal at hit position
        sibr::Vector3f hitNormal(0, 1, 0); // default
        if (hitTriID >= 0 && hitTriID < _mesh->triangles().size()) {
            const auto& tri = _mesh->triangles()[hitTriID];
            const auto& n0 = _mesh->normals()[tri[0]];
            const auto& n1 = _mesh->normals()[tri[1]];
            const auto& n2 = _mesh->normals()[tri[2]];
            hitNormal = (n0 + n1 + n2) / 3.0f;
            hitNormal.normalize();
        }
        
        // Call the main Compute method
        Compute(hitPos, hitNormal, radius);
    }


    void Init(const sibr::Mesh* mesh) {
        _mesh = mesh;
        buildBaseAdjacency();

        // Check if normals are all zero and compute if necessary
        bool normalsAreZero = true;
        if (!_mesh->normals().empty()) {
            for (const auto& n : _mesh->normals()) {
                if (n.norm() > 1e-6f) { // Check if normal is non-zero
                    normalsAreZero = false;
                    break;
                }
            }
        } else {
            // If normals vector is empty, it means no normals were loaded
            normalsAreZero = true; 
        }

        if (normalsAreZero) {
            std::cout << "[INFO] ExpMapSolverSIBR: Detected zero or missing normals. Computing vertex normals..." << std::endl;
            computeVertexNormals();
        }
    }

    void RegisterAppPointCloud(const std::vector<float>& pointData, const std::vector<int>& fids) { 
        _appCloudData = pointData; 
        _appCloudFids = fids;
    }
    void RegisterGeoPointCloud(const std::vector<float>& pointData) { _geoCloudData = pointData; }

    const std::vector<sibr::Vector3u>& GetActiveTriangles() const { return _validTriangles; }

    glm::vec3 getPos(int id) const {
        if (id < _mesh->vertices().size()) return toGlm(_mesh->vertices()[id]);
        else return _extraNodePositions[id - _mesh->vertices().size()];
    }

    glm::vec3 getNormal(int id) const {
        if (id < _mesh->vertices().size()) return toGlm(_mesh->normals()[id]);
        else return _extraNodeNormals[id - _mesh->vertices().size()];
    }

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
        _extraNodeNormals.clear();
        _projectionStats = ProjectionStats();
        
        _adj = _baseAdj; 
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
                if (!_displayUVs.count(t[0]) || !_displayUVs.count(t[1]) || !_displayUVs.count(t[2])) 
                    continue;

                glm::vec2 uv0 = _displayUVs[t[0]];
                glm::vec2 uv1 = _displayUVs[t[1]];
                glm::vec2 uv2 = _displayUVs[t[2]];
                
                glm::vec3 p0 = toGlm(_mesh->vertices()[t[0]]);
                glm::vec3 p1 = toGlm(_mesh->vertices()[t[1]]);
                glm::vec3 p2 = toGlm(_mesh->vertices()[t[2]]);

                // RELAXED THRESHOLD - Much more lenient than before
                float signedAreaUV = (uv1.x - uv0.x) * (uv2.y - uv0.y) - 
                                     (uv1.y - uv0.y) * (uv2.x - uv0.x);
                
                // Changed from 1e-7f to much larger threshold
                if (std::abs(signedAreaUV) < 1e-10f) continue;

                float dUV_01 = glm::distance(uv0, uv1) / _uvScale;
                float d3D_01 = glm::distance(p0, p1);
                float dUV_12 = glm::distance(uv1, uv2) / _uvScale;
                float d3D_12 = glm::distance(p1, p2);
                float dUV_20 = glm::distance(uv2, uv0) / _uvScale;
                float d3D_20 = glm::distance(p2, p0);

                if (d3D_01 < 1e-6f && d3D_12 < 1e-6f && d3D_20 < 1e-6f) continue;

                _validTriangles.push_back(t);
                _validTriIDs.push_back((int)i); 
                _validTriangleIndicesSet.insert((int)i);
                
                if (d3D_01 > 1e-6f) ratios.push_back(dUV_01 / d3D_01);
                if (d3D_12 > 1e-6f) ratios.push_back(dUV_12 / d3D_12);
                if (d3D_20 > 1e-6f) ratios.push_back(dUV_20 / d3D_20);
            }

            if (!ratios.empty()) {
                float sum = std::accumulate(ratios.begin(), ratios.end(), 0.0f);
                float mean = sum / ratios.size();
                _autoThreshold = std::max(1.5f, mean * 2.5f); 
            } else {
                _autoThreshold = 3.0f;
            }

            ProjectAndInsertClouds(target, radius * 1.2f);
            
            // 打印统计信息
            std::cout << "\n========== ProjectionStats ==========" << std::endl;
            std::cout << "APP Points:" << std::endl;
            std::cout << "  Total: " << _projectionStats.appTotal << std::endl;
            std::cout << "  Out of radius: " << _projectionStats.appOutOfRadius << std::endl;
            std::cout << "  Invalid face_id: " << _projectionStats.appInvalidFid << std::endl;
            std::cout << "  Face_id not in range: " << _projectionStats.appFidOutOfRange << std::endl;
            std::cout << "  Successfully projected: " << _projectionStats.appProjected << std::endl;
            std::cout << "\nGEO Points:" << std::endl;
            std::cout << "  Total: " << _projectionStats.geoTotal << std::endl;
            std::cout << "  Out of radius: " << _projectionStats.geoOutOfRadius << std::endl;
            std::cout << "  Successfully projected: " << _projectionStats.geoProjected << std::endl;
            std::cout << "====================================\n" << std::endl;
        }
    }

    void RenderUI() {
        ImGui::SetNextWindowSize(ImVec2(400, 450), ImGuiCond_FirstUseEver);
        if(ImGui::Begin("ExpMap UV Result", nullptr, ImGuiWindowFlags_MenuBar)) {
            
            if (ImGui::BeginMenuBar()) {
                if (ImGui::BeginMenu("File")) {
                    if (ImGui::BeginMenu("Load Background")) {
                        
                        // Load images from assets/texture directory
                        std::string targetDir = "D:/SGGaussians/SGGaussians/assets/texture/";
                        std::vector<std::string> imageFiles = _textureLoader.scanForImages(targetDir);
                        
                        if (imageFiles.empty()) {
                            ImGui::MenuItem("(No images in assets/texture)", NULL, false, false);
                        } else {
                            for (const auto& imagePath : imageFiles) {
                                // Extract filename for display (simple logic to remove path)
                                std::string displayName = imagePath;
                                size_t lastSlash = displayName.find_last_of("/\\");
                                if (lastSlash != std::string::npos) {
                                    displayName = displayName.substr(lastSlash + 1);
                                }

                                if (ImGui::MenuItem(displayName.c_str())) {
                                    _textureLoader.LoadImage(imagePath);
                                }
                            }
                        }
                        ImGui::EndMenu();
                    }
                    ImGui::EndMenu();
                }
                ImGui::EndMenuBar();
            }

            // ---- Save / manage slots ----
            if (!_validTriangles.empty() && _textureLoader.getTexture()) {
                if (ImGui::Button("Save Current Slot")) {
                    int idx = SaveCurrentSlot();
                    std::cout << "[INFO] Saved texture slot " << idx << std::endl;
                }
                ImGui::SameLine();
            }
            if (!_slots.empty()) {
                if (ImGui::Button("Clear All Slots")) ClearAllSlots();
                ImGui::Separator();
                ImGui::Text("Saved Slots (%lu):", _slots.size());
                for (int i = 0; i < (int)_slots.size(); ++i) {
                    ImGui::PushID(i);
                    ImGui::Checkbox("##vis", &_slots[i].visible);
                    ImGui::SameLine();
                    // Extract filename for display
                    std::string lbl = _slots[i].texturePath;
                    size_t sl = lbl.find_last_of("/\\");
                    if (sl != std::string::npos) lbl = lbl.substr(sl + 1);
                    lbl = "Slot " + std::to_string(i + 1) + ": " + lbl;
                    ImGui::TextUnformatted(lbl.c_str());
                    ImGui::SameLine();
                    ImGui::PushItemWidth(80.f);
                    ImGui::SliderFloat("##alpha", &_slots[i].alpha, 0.f, 1.f);
                    ImGui::PopItemWidth();
                    ImGui::SameLine();
                    if (ImGui::SmallButton("X")) { RemoveSlot(i); ImGui::PopID(); break; }
                    ImGui::PopID();
                }
                ImGui::Separator();
            }

            if (_displayUVs.empty()) {
                ImGui::TextColored(ImVec4(1,1,0,1), "Right-click on the mesh to compute UV.");
                ImGui::End();
                return;
            }

            // ================================================================
            // 显示统计信息
            // ================================================================
            ImGui::Text("Tris: %lu | App: %d/%d | Geo: %d/%d", 
                        _validTriangles.size(),
                        _projectionStats.appProjected,
                        _projectionStats.appTotal,
                        _projectionStats.geoProjected,
                        _projectionStats.geoTotal);
            
            // 显示详细的统计信息（可折叠）
            if (ImGui::TreeNode("Projection Statistics")) {
                ImGui::TextColored(ImVec4(0, 1, 1, 1), "APP Points:");
                ImGui::Text("  Total: %d", _projectionStats.appTotal);
                ImGui::Text("  Projected: %d", _projectionStats.appProjected);
                ImGui::Text("  Out of radius: %d", _projectionStats.appOutOfRadius);
                ImGui::Text("  Invalid face_id: %d", _projectionStats.appInvalidFid);
                ImGui::Text("  Face_id not in range: %d", _projectionStats.appFidOutOfRange);
                
                if (_projectionStats.appTotal > 0) {
                    float appRate = (float)_projectionStats.appProjected / _projectionStats.appTotal * 100.0f;
                    ImGui::TextColored(ImVec4(1, 1, 0, 1), "  Projection rate: %.1f%%", appRate);
                }
                
                ImGui::TextColored(ImVec4(1, 0.5, 0.5, 1), "GEO Points:");
                ImGui::Text("  Total: %d", _projectionStats.geoTotal);
                ImGui::Text("  Projected: %d", _projectionStats.geoProjected);
                ImGui::Text("  Out of radius: %d", _projectionStats.geoOutOfRadius);
                
                if (_projectionStats.geoTotal > 0) {
                    float geoRate = (float)_projectionStats.geoProjected / _projectionStats.geoTotal * 100.0f;
                    ImGui::TextColored(ImVec4(1, 1, 0, 1), "  Projection rate: %.1f%%", geoRate);
                }
                
                ImGui::TreePop();
            }
            
            ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), "Hold CTRL + Left Click to pick Start/End Points");
            // NEW: Hint for selection box feature
            ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "Hold SHIFT + Drag to select area");
            
            if (_dijkstraPath.size() > 0) {
                ImGui::TextColored(ImVec4(0,1,0,1), "Path Length: %lu hops", _dijkstraPath.size());
            }

            ImGui::SameLine();
            if (ImGui::Button("Reset View")) { _viewScale = 1.0f; _viewOffset = ImVec2(0,0); }
            
            ImGui::Checkbox("Fill", &_drawFilled);
            ImGui::SameLine();
            ImGui::Checkbox("Cull", &_cullBackFace);
            ImGui::SameLine();
            ImGui::Checkbox("BG", &_showBackgroundTexture);  // NEW: Show/Hide background texture
            ImGui::Checkbox("Show App", &_showAppPoints);
            ImGui::SameLine();
            ImGui::Checkbox("Show Geo", &_showGeoPoints);

            ImVec2 p = ImGui::GetCursorScreenPos();
            ImVec2 sz = ImGui::GetContentRegionAvail();
            if(sz.x < 50) sz.x = 50; 
            if(sz.y < 50) sz.y = 50;
            float dim = std::min(sz.x, sz.y);

            // 用 InvisibleButton 佔位，讓 ImGui 知道這塊畫布的範圍，
            // 才能正確做 hit-test；IsItemHovered 只在真正的畫布區域內才回 true
            ImGui::InvisibleButton("##uvcanvas", sz);
            bool isHovered = ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenBlockedByActiveItem);
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

            // NEW: Handle selection box dragging
            if (isHovered && ImGui::GetIO().KeyShift) {
                if (ImGui::IsMouseClicked(0)) {
                    _isSelecting = true;
                    float lx = mousePos.x - (p.x + sz.x * 0.5f + _viewOffset.x);
                    float ly = mousePos.y - (p.y + sz.y * 0.5f + _viewOffset.y);
                    _selectionStart.x = (lx / (dim * _viewScale)) + 0.5f;
                    _selectionStart.y = 1.0f - ((ly / (dim * _viewScale)) + 0.5f);
                    _selectionEnd = _selectionStart;
                }
                if (_isSelecting && ImGui::IsMouseDragging(0)) {
                    float lx = mousePos.x - (p.x + sz.x * 0.5f + _viewOffset.x);
                    float ly = mousePos.y - (p.y + sz.y * 0.5f + _viewOffset.y);
                    _selectionEnd.x = (lx / (dim * _viewScale)) + 0.5f;
                    _selectionEnd.y = 1.0f - ((ly / (dim * _viewScale)) + 0.5f);
                }
                if (_isSelecting && ImGui::IsMouseReleased(0)) {
                    _isSelecting = false;
                    UpdateSelectedTriangles();
                }
            } else {
                _isSelecting = false;
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

            // Draw background: either texture or plain color
            if (_showBackgroundTexture) {
                const auto& bgTex = _textureLoader.getTexture();
                if (bgTex && bgTex->handle() != 0) {
                    dl->AddImage((void*)(intptr_t)bgTex->handle(), p, ImVec2(p.x + sz.x, p.y + sz.y), ImVec2(0, 1), ImVec2(1, 0));
                } else {
                    dl->AddRectFilled(p, ImVec2(p.x + sz.x, p.y + sz.y), IM_COL32(40, 40, 40, 255));
                }
            } else {
                // Show plain gray background without texture
                dl->AddRectFilled(p, ImVec2(p.x + sz.x, p.y + sz.y), IM_COL32(40, 40, 40, 255));
            }
            
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

            // ================================================================
            // 改进的点显示逻辑 - 修复APP/GEO识别问题
            // ================================================================
            size_t meshSize = _mesh->vertices().size();
            size_t appPointCount = _projectionStats.appProjected;  // ← 使用实际投影数量，不是原始数量
            
            for (auto const& [id, uv] : _displayUVs) {
                if (id < (int)meshSize) continue;  // 跳过网格顶点
                
                // 检查UV范围是否合法
                if (uv.x < -1.0f || uv.x > 2.0f || uv.y < -1.0f || uv.y > 2.0f) {
                    // 异常UV，跳过显示
                    continue;
                }
                
                ImVec2 pos = TransformUV(uv);
                
                // 改进的识别逻辑：使用实际投影的数量
                // APP点在[meshSize, meshSize + appPointCount)范围内
                // GEO点在[meshSize + appPointCount, meshSize + appPointCount + geoPointCount)范围内
                int extraNodeIndex = id - (int)meshSize;
                bool isApp = (extraNodeIndex < (int)appPointCount);
                
                if (isApp && _showAppPoints) {
                    dl->AddCircleFilled(pos, 3.0f, IM_COL32(0, 255, 255, 255));  // 青色APP点
                } else if (!isApp && _showGeoPoints) {
                    dl->AddCircleFilled(pos, 3.0f, IM_COL32(255, 50, 50, 255));   // 红色GEO点
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

            // NEW: Draw selection box
            if (_isSelecting) {
                ImVec2 rect_min = TransformUV(_selectionStart);
                ImVec2 rect_max = TransformUV(_selectionEnd);
                if (rect_min.x > rect_max.x) std::swap(rect_min.x, rect_max.x);
                if (rect_min.y > rect_max.y) std::swap(rect_min.y, rect_max.y);
                dl->AddRect(rect_min, rect_max, IM_COL32(255, 255, 0, 255), 0.0f, 0, 2.0f);
                dl->AddRectFilled(rect_min, rect_max, IM_COL32(255, 255, 0, 50));
            }

            if (_startNode != -1 && _displayUVs.count(_startNode)) dl->AddCircleFilled(TransformUV(_displayUVs[_startNode]), 5.0f, IM_COL32(0, 255, 0, 255)); 
            if (_endNode != -1 && _displayUVs.count(_endNode)) dl->AddCircleFilled(TransformUV(_displayUVs[_endNode]), 5.0f, IM_COL32(255, 0, 0, 255)); 

            dl->PopClipRect();
        }
        ImGui::End();
    }

private:
    // NEW: Update selected triangles within selection box
    void UpdateSelectedTriangles() {
        _selectedTriangles.clear();
        
        if (!_mesh) return;
        
        float minX = std::min(_selectionStart.x, _selectionEnd.x);
        float maxX = std::max(_selectionStart.x, _selectionEnd.x);
        float minY = std::min(_selectionStart.y, _selectionEnd.y);
        float maxY = std::max(_selectionStart.y, _selectionEnd.y);
        
        for (const auto& t : _validTriangles) {
            if (!_displayUVs.count(t[0]) || !_displayUVs.count(t[1]) || !_displayUVs.count(t[2])) continue;
            
            glm::vec2 uv0 = _displayUVs[t[0]];
            glm::vec2 uv1 = _displayUVs[t[1]];
            glm::vec2 uv2 = _displayUVs[t[2]];
            
            // Check if any vertex is inside the selection box
            bool in0 = (uv0.x >= minX && uv0.x <= maxX && uv0.y >= minY && uv0.y <= maxY);
            bool in1 = (uv1.x >= minX && uv1.x <= maxX && uv1.y >= minY && uv1.y <= maxY);
            bool in2 = (uv2.x >= minX && uv2.x <= maxX && uv2.y >= minY && uv2.y <= maxY);
            
            // Add to selection if any vertex is inside
            if (in0 || in1 || in2) {
                _selectedTriangles.push_back(t);
            }
        }
        
        std::cout << "[INFO] Selected " << _selectedTriangles.size() << " triangles in UV selection box" << std::endl;
    }

    void computeVertexNormals() {
        if (!_mesh || _mesh->vertices().empty() || _mesh->triangles().empty()) {
            std::cerr << "[WARNING] Cannot compute normals: Mesh is invalid or empty." << std::endl;
            return;
        }

        // Ensure the normals vector is correctly sized and mutable
        std::vector<sibr::Vector3f>& normals = const_cast<std::vector<sibr::Vector3f>&>(_mesh->normals()); 
        if (normals.size() != _mesh->vertices().size()) {
             normals.assign(_mesh->vertices().size(), sibr::Vector3f(0.0f, 0.0f, 0.0f));
        } else {
            std::fill(normals.begin(), normals.end(), sibr::Vector3f(0.0f, 0.0f, 0.0f)); // Clear existing normals
        }

        const auto& vertices = _mesh->vertices();
        const auto& triangles = _mesh->triangles();

        for (const auto& tri : triangles) {
            // Ensure triangle indices are valid
            if (tri.x() >= vertices.size() || tri.y() >= vertices.size() || tri.z() >= vertices.size()) {
                std::cerr << "[WARNING] Invalid triangle index encountered during normal computation. Skipping triangle." << std::endl;
                continue;
            }

            const sibr::Vector3f& v0 = vertices[tri.x()];
            const sibr::Vector3f& v1 = vertices[tri.y()];
            const sibr::Vector3f& v2 = vertices[tri.z()];

            sibr::Vector3f edge1 = v1 - v0;
            sibr::Vector3f edge2 = v2 - v0;
            sibr::Vector3f faceNormal = edge1.cross(edge2);

            // Only add if face normal is not degenerate
            if (faceNormal.norm() > 1e-6f) {
                faceNormal.normalize();
                normals[tri.x()] += faceNormal;
                normals[tri.y()] += faceNormal;
                normals[tri.z()] += faceNormal;
            }
        }

        // Normalize accumulated vertex normals
        for (sibr::Vector3f& n : normals) {
            if (n.norm() > 1e-6f) {
                n.normalize();
            } else {
                // If a vertex has no incident non-degenerate faces, give it an arbitrary normal
                n = sibr::Vector3f(0.0f, 1.0f, 0.0f); 
            }
        }
        std::cout << "[INFO] ExpMapSolverSIBR: Computed missing vertex normals." << std::endl;
    }
    
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
        float newCost = parent.cost + dist;

        if(newCost >= curr.cost) return;

        curr.cost = newCost;
        curr.parentId = parentIdx;
        
        bool isExtraNode = (parentIdx >= _mesh->vertices().size()) || (currIdx >= _mesh->vertices().size());
        if (isExtraNode) return;

        // ===== IMPROVED Parallel Transport using Rodrigues' rotation formula =====
        glm::vec3 parentN = glm::normalize(getNormal(parentIdx));
        glm::vec3 currN = glm::normalize(getNormal(currIdx));
        
        glm::vec3 edge = currPos - parentPos;
        
        // Compute tangent Y
        glm::vec3 parentTangentX = parent.tangentX;
        glm::vec3 parentTangentY = glm::normalize(glm::cross(parentN, parentTangentX));
        
        // Project edge to tangent plane
        glm::vec3 edgeProj = edge - parentN * glm::dot(edge, parentN);
        float dU = glm::dot(edgeProj, parentTangentX);
        float dV = glm::dot(edgeProj, parentTangentY);
        
        curr.uv = parent.uv + glm::vec2(dU, dV);

        // ===== Rodrigues rotation formula for stable tangent propagation =====
        glm::vec3 rotAxis = glm::cross(parentN, currN);
        float rotAxisLen = glm::length(rotAxis);
        
        glm::vec3 newTangentX;
        
        if (rotAxisLen < 1e-6f) {
            float cosA = glm::dot(parentN, currN);
            newTangentX = (cosA >= 0.0f) ? parentTangentX : -parentTangentX;
        } else {
            rotAxis = glm::normalize(rotAxis);
            float cosA = glm::clamp(glm::dot(parentN, currN), -1.0f, 1.0f);
            float sinA = std::sqrt(std::max(0.0f, 1.0f - cosA * cosA));
            
            glm::vec3 term1 = parentTangentX * cosA;
            glm::vec3 term2 = glm::cross(rotAxis, parentTangentX) * sinA;
            glm::vec3 term3 = rotAxis * (glm::dot(rotAxis, parentTangentX) * (1.0f - cosA));
            
            newTangentX = term1 + term2 + term3;
        }
        
        newTangentX = newTangentX - currN * glm::dot(newTangentX, currN);
        float newTangentLen = glm::length(newTangentX);
        
        if (newTangentLen < 0.5f) {
            glm::vec3 arbVec = (std::abs(currN.z) < 0.9f) ? 
                               glm::vec3(0.0f, 0.0f, 1.0f) : glm::vec3(1.0f, 0.0f, 0.0f);
            newTangentX = glm::cross(currN, arbVec);
        }
        
        curr.tangentX = glm::normalize(newTangentX);
    }

    // ================================================================
    // ProjectAndInsertClouds - 完整重写，添加错误检查
    // ================================================================
    void ProjectAndInsertClouds(const glm::vec3& center, float radius) {
        std::map<int, std::vector<int>> triToExtraNodes;

        auto process = [&](const std::vector<float>& cloud, const std::vector<int>& fids, 
                          std::vector<glm::vec2>& outPts, const char* cloudName) {
            if (cloud.empty()) return;
            
            size_t numPoints = cloud.size() / 3;
            float r2 = radius * radius;
            size_t meshSize = _mesh->vertices().size();
            bool use_fids = !fids.empty() && fids.size() == numPoints;

            // 统计变量
            int outOfRadius = 0;
            int invalidFid = 0;
            int fidOutOfRange = 0;
            int projected = 0;

            std::cout << "\n[ProjectAndInsertClouds] Processing " << cloudName << " cloud" << std::endl;
            std::cout << "  Total points: " << numPoints << std::endl;
            std::cout << "  Has face_id: " << (use_fids ? "YES" : "NO") << std::endl;
            std::cout << "  Valid triangles in ExpMap: " << _validTriangleIndicesSet.size() << std::endl;

            for (size_t i = 0; i < numPoints; ++i) {
                glm::vec3 pt(cloud[i*3], cloud[i*3+1], cloud[i*3+2]);
                
                // 第一个检查：点是否在展开半径内
                if (glm::dot(pt - center, pt - center) > r2) {
                    outOfRadius++;
                    continue;
                }

                float bestDist = 1e9f; 
                glm::vec2 bestUV(0, 0); 
                int bestTriGlobalID = -1; 
                bool found = false;
                glm::vec3 bestProjPos(0, 0, 0);
                glm::vec3 bestProjNormal(0, 1, 0);

                if (use_fids) {
                    int face_id = fids[i];
                    
                    // 第二个检查：face_id范围检查
                    if (face_id < 0 || face_id >= (int)_mesh->triangles().size()) {
                        std::cerr << "[WARNING] " << cloudName << " point " << i 
                                  << " has INVALID face_id: " << face_id << std::endl;
                        invalidFid++;
                        continue;
                    }
                    
                    // 第三个检查：face_id是否在展开范围内
                    if (!_validTriangleIndicesSet.count(face_id)) {
                        fidOutOfRange++;
                        continue;
                    }
                    
                    // 投影到该三角形
                    const auto& t = _mesh->triangles()[face_id];
                    glm::vec3 v0 = toGlm(_mesh->vertices()[t[0]]);
                    glm::vec3 v1 = toGlm(_mesh->vertices()[t[1]]);
                    glm::vec3 v2 = toGlm(_mesh->vertices()[t[2]]);
                    glm::vec3 pVec = pt - v0;
                    glm::vec3 v0v1 = v1 - v0; 
                    glm::vec3 v0v2 = v2 - v0;
                    float d00 = glm::dot(v0v1, v0v1); 
                    float d01 = glm::dot(v0v1, v0v2); 
                    float d11 = glm::dot(v0v2, v0v2);
                    float d20 = glm::dot(pVec, v0v1); 
                    float d21 = glm::dot(pVec, v0v2); 
                    float denom = d00 * d11 - d01 * d01;
                    
                    if (std::abs(denom) > 1e-9f) {
                        float v = (d11 * d20 - d01 * d21) / denom; 
                        float w = (d00 * d21 - d01 * d20) / denom; 
                        float u = 1.0f - v - w;
                        
                        bestUV = u * _displayUVs.at(t[0]) + v * _displayUVs.at(t[1]) + w * _displayUVs.at(t[2]);
                        bestProjPos = u * v0 + v * v1 + w * v2;
                        
                        glm::vec3 n0 = toGlm(_mesh->normals()[t[0]]);
                        glm::vec3 n1 = toGlm(_mesh->normals()[t[1]]);
                        glm::vec3 n2 = toGlm(_mesh->normals()[t[2]]);
                        bestProjNormal = glm::normalize(u * n0 + v * n1 + w * n2);

                        bestTriGlobalID = face_id; 
                        found = true;
                    }
                } else {
                    // GEO点没有face_id，搜索所有三角形
                    for (int globalTIdx : _validTriIDs) {
                        const auto& t = _mesh->triangles()[globalTIdx];
                        glm::vec3 v0 = toGlm(_mesh->vertices()[t[0]]);
                        glm::vec3 v1 = toGlm(_mesh->vertices()[t[1]]);
                        glm::vec3 v2 = toGlm(_mesh->vertices()[t[2]]);
                        glm::vec3 n = glm::cross(v1 - v0, v2 - v0); 
                        if (glm::length(n) < 1e-9) continue; 
                        n = glm::normalize(n);
                        if (std::abs(glm::dot(pt - v0, n)) > 0.1f * radius) continue; 
                        glm::vec3 pProj = pt - n * glm::dot(pt - v0, n);
                        
                        glm::vec3 v0v1 = v1 - v0; 
                        glm::vec3 v0v2 = v2 - v0; 
                        glm::vec3 pVec = pProj - v0;
                        float d00 = glm::dot(v0v1, v0v1); 
                        float d01 = glm::dot(v0v1, v0v2); 
                        float d11 = glm::dot(v0v2, v0v2);
                        float d20 = glm::dot(pVec, v0v1); 
                        float d21 = glm::dot(pVec, v0v2); 
                        float denom = d00 * d11 - d01 * d01;
                        
                        if (std::abs(denom) > 1e-9f) {
                            float v = (d11 * d20 - d01 * d21) / denom; 
                            float w = (d00 * d21 - d01 * d20) / denom; 
                            float u = 1.0f - v - w;
                            if (u >= -0.01f && v >= -0.01f && w >= -0.01f && 
                                u <= 1.01f && v <= 1.01f && w <= 1.01f) {
                                
                                float dist = glm::distance(pt, pProj);
                                if (dist < bestDist) { 
                                    bestDist = dist; 
                                    bestUV = u * _displayUVs.at(t[0]) + v * _displayUVs.at(t[1]) + w * _displayUVs.at(t[2]); 
                                    bestProjPos = pProj; 
                                    bestTriGlobalID = globalTIdx; 
                                    found = true; 
                                    
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
                    // 第四个检查：UV范围检查
                    if (bestUV.x < 0.0f || bestUV.x > 1.0f || bestUV.y < 0.0f || bestUV.y > 1.0f) {
                        std::cerr << "[WARNING] " << cloudName << " point " << i 
                                  << " projected to out-of-range UV: (" << bestUV.x << ", " << bestUV.y << ")" << std::endl;
                        // 继续处理，但记录警告
                    }
                    
                    outPts.push_back(bestUV);
                    int newNodeID = (int)meshSize + (int)_extraNodePositions.size();
                    _extraNodePositions.push_back(bestProjPos);
                    _extraNodeNormals.push_back(bestProjNormal);
                    _displayUVs[newNodeID] = bestUV;
                    
                    ExpVertex vExtra;
                    vExtra.id = newNodeID;
                    vExtra.cost = 1e9f;
                    vExtra.uv = bestUV;
                    vExtra.frozen = false;
                    _vertexData[newNodeID] = vExtra;
                    
                    if (_adj.size() <= newNodeID) _adj.resize(newNodeID + 1);

                    const auto& t = _mesh->triangles()[bestTriGlobalID];
                    int v[3] = { (int)t[0], (int)t[1], (int)t[2] };
                    for(int k = 0; k < 3; ++k) {
                        _adj[newNodeID].push_back(v[k]);
                        _adj[v[k]].push_back(newNodeID);
                    }

                    if (triToExtraNodes.count(bestTriGlobalID)) {
                        for (int siblingID : triToExtraNodes[bestTriGlobalID]) {
                            _adj[newNodeID].push_back(siblingID);
                            _adj[siblingID].push_back(newNodeID);
                        }
                    }
                    triToExtraNodes[bestTriGlobalID].push_back(newNodeID);
                    
                    projected++;
                }
            }
            
            // 记录统计信息
            if (strcmp(cloudName, "APP") == 0) {
                _projectionStats.appTotal = numPoints;
                _projectionStats.appOutOfRadius = outOfRadius;
                _projectionStats.appInvalidFid = invalidFid;
                _projectionStats.appFidOutOfRange = fidOutOfRange;
                _projectionStats.appProjected = projected;
            } else {
                _projectionStats.geoTotal = numPoints;
                _projectionStats.geoOutOfRadius = outOfRadius;
                _projectionStats.geoProjected = projected;
            }
            
            // 打印统计结果
            std::cout << "  Out of radius: " << outOfRadius << std::endl;
            if (use_fids) {
                std::cout << "  Invalid face_id: " << invalidFid << std::endl;
                std::cout << "  Face_id not in ExpMap range: " << fidOutOfRange << std::endl;
            }
            std::cout << "  Successfully projected: " << projected << std::endl;
            if (numPoints > 0) {
                float rate = (float)projected / numPoints * 100.0f;
                std::cout << "  Projection rate: " << rate << "%" << std::endl;
            }
            std::cout << std::endl;
        };

        process(_appCloudData, _appCloudFids, _projectedAppPoints, "APP");
        process(_geoCloudData, {}, _projectedGeoPoints, "GEO");
    }
    
    void ComputeExtraNodeCosts() {
        size_t meshSize = _mesh->vertices().size();
        for (auto& [nodeID, vData] : _vertexData) {
            if (nodeID < (int)meshSize) continue; 
            
            float minCost = 1e9f;
            int bestParent = -1;
            
            if (nodeID < (int)_adj.size()) {
                for (int neighborID : _adj[nodeID]) {
                    if (neighborID >= (int)meshSize) continue; 
                    if (_vertexData.find(neighborID) == _vertexData.end()) continue;
                    if (!_vertexData[neighborID].frozen) continue;
                    
                    float dist = glm::distance(getPos(nodeID), getPos(neighborID));
                    glm::vec3 nodeN = getNormal(nodeID);
                    glm::vec3 neighborN = getNormal(neighborID);
                    float dotNormal = glm::dot(nodeN, neighborN);
                    
                    if (dotNormal < 0.3f) continue; 
                    
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

        size_t meshVertCount = _mesh->vertices().size();

        while (!pq.empty()) {
            float d = pq.top().first;
            int u = pq.top().second;
            pq.pop();

            if (u == endIdx) break; 
            if (d > minDist[u]) continue;
            if (u >= _adj.size()) continue;

            for (int v : _adj[u]) {
                if (minDist.find(v) == minDist.end()) continue;
                bool isVExtra = (v >= meshVertCount);
                if (isVExtra && v != endIdx) {
                    continue; 
                }

                float dist = glm::distance(getPos(u), getPos(v));
                
                glm::vec3 uN = getNormal(u);
                glm::vec3 vN = getNormal(v);
                float dotNormal = glm::dot(uN, vN);
                
                if (dotNormal < 0.5f) continue; 

                bool isUExtra = (u >= meshVertCount);

                if (!isUExtra && !isVExtra) {
                    float maxEdgeLength = 0.25f; 
                    if (dist > maxEdgeLength) continue;
                }

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
            std::cout << "[DEBUG] ComputeDijkstra: Path found with " << _dijkstraPath.size() << " nodes." << std::endl;
        } else {
            std::cout << "[DEBUG] ComputeDijkstra: No path found between " << startIdx << " and " << endIdx << "." << std::endl;
        }
    }

    const sibr::Mesh*             _mesh = nullptr;
    std::vector<std::vector<int>> _baseAdj;
    std::vector<std::vector<int>> _adj;
    std::map<int, ExpVertex>      _vertexData;
    TangentFrame                  _seedFrame;

    std::vector<sibr::Vector3u>   _validTriangles;
    std::vector<int>              _validTriIDs;
    std::set<int>                 _validTriangleIndicesSet;

    std::vector<glm::vec3>        _extraNodePositions;
    std::vector<glm::vec3>        _extraNodeNormals;

    // UV viewer state
    float  _uvScale       = 1.0f;
    ImVec2 _viewOffset    = ImVec2(0, 0);
    float  _viewScale     = 1.0f;
    bool   _drawFilled    = true;
    bool   _cullBackFace  = true;
    bool   _showBackgroundTexture = false;
    
    bool   _showAppPoints = true;
    bool   _showGeoPoints = true;

    std::vector<glm::vec2> _projectedAppPoints;
    std::vector<glm::vec2> _projectedGeoPoints;
    
    std::vector<int> _dijkstraPath;
    int _startNode = -1;
    int _endNode = -1;

    std::map<int, glm::vec2> _displayUVs;
    std::vector<sibr::Vector3u> _selectedTriangles;
    glm::vec2 _selectionStart, _selectionEnd;
    bool _isSelecting = false;
    
    TextureLoader _textureLoader;
    std::vector<TextureSlotData> _slots;
    
    std::vector<float> _appCloudData;
    std::vector<int>   _appCloudFids;
    std::vector<float> _geoCloudData;

    float _autoThreshold = 3.0f;
    
    ProjectionStats _projectionStats;
};

#endif