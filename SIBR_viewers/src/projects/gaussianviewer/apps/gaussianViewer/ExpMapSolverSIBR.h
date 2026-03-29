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

// Helper to convert SIBR vector to GLM
inline glm::vec3 toGlm(const sibr::Vector3f& v) { 
    return glm::vec3(v.x(), v.y(), v.z()); 
}

struct ProjectionStats {
    int appTotal = 0;
    int appOutOfRadius = 0;
    int appInvalidFid = 0;
    int appFidOutOfRange = 0;
    int appProjected = 0;
    
    int geoTotal = 0;
    int geoOutOfRadius = 0;
    int geoProjected = 0;
};

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


class ExpMapSolverSIBR {
public:
    ~ExpMapSolverSIBR() {
        ClearAllSlots();
        if (_liveSSBO) glDeleteBuffers(1, &_liveSSBO);
    }

    const std::vector<int>& GetDijkstraPath() const { return _dijkstraPath; }

    const std::vector<sibr::Vector3u>& GetActiveTris() const { return _validTriangles; }
    const std::set<int>& GetActiveTriIndices() const { return _validTriangleIndicesSet; }
    
    const std::vector<sibr::Vector3u>& GetSelectedTris() const { return _selectedTriangles; }
    
    const std::vector<ProjectedGaussian>& GetProjectedGeoPoints() const { return _projectedGeoPoints; }
    const std::vector<ProjectedGaussian>& GetProjectedAppPoints() const { return _projectedAppPoints; }
    const std::vector<int>& GetSelectedGeoIndices() const { return _selectedGeoIndices; }
    const std::vector<int>& GetSelectedAppIndices() const { return _selectedAppIndices; }
    
    const std::map<int, glm::vec2>& GetDisplayUVs() const { return _displayUVs; }
    const TextureLoader& GetTextureLoader() const { return _textureLoader; }
    TextureLoader& GetTextureLoader() { return _textureLoader; }

    bool ConsumeTextureDirty() {
        if (_textureDirty) { _textureDirty = false; return true; }
        return false;
    }

    const ProjectionStats& GetProjectionStats() const { return _projectionStats; }

    // Call this from main.cpp after building _texGaussians so that
    // SaveCurrentSlot stores the same main-cloud Gaussians that the live
    // renderer uses, making before/after Save visually identical.
    void SetMainGaussiansForNextSave(const std::vector<ProjectedGaussian>& g) {
        _pendingMainGaussians = g;
    }

    // Set the UV positions of Gaussians that are inside the sphere but
    // failed to receive a texture UV (suppressed / no-triangle-hit).
    // These will be shown as green dots on the UV canvas.
    void SetSuppressedUVs(const std::vector<glm::vec2>& uvs) {
        _suppressedUVs = uvs;
    }
    void ClearSuppressedUVs() { _suppressedUVs.clear(); }

    int SaveCurrentSlot(const std::string& name = "") {
        if (_validTriangles.empty() || !_textureLoader.getTexture()) return -1;
        TextureSlotData slot;
        slot.triangles   = _validTriangles;
        slot.uvMap       = _displayUVs;
        slot.texture     = _textureLoader.getTexture();
        slot.texturePath = _textureLoader.getPath();
        slot.alpha       = 1.f;
        slot.visible     = true;

        // Use the main-cloud Gaussians (set via SetMainGaussiansForNextSave)
        // instead of the aux geo/app point clouds, so the saved slot renders
        // identically to what was visible before saving.
        if (!_pendingMainGaussians.empty()) {
            slot.geoPoints = _pendingMainGaussians;
            slot.appPoints = {};
        } else {
            slot.geoPoints = _projectedGeoPoints;
            slot.appPoints = _projectedAppPoints;
        }
        slot.ssbo    = 0;
        slot.isDirty = true;
        
        _slots.push_back(std::move(slot));
        return (int)_slots.size() - 1;
    }
    void RemoveSlot(int idx) {
        if (idx >= 0 && idx < (int)_slots.size()) {
            if (_slots[idx].ssbo) glDeleteBuffers(1, &_slots[idx].ssbo);
            _slots.erase(_slots.begin() + idx);
        }
    }
    void ClearAllSlots() { 
        for (auto& s : _slots) {
            if (s.ssbo) glDeleteBuffers(1, &s.ssbo);
        }
        _slots.clear(); 
    }
    
    std::vector<TextureSlotData>& GetAllSlotsRef() { return _slots; }
    const std::vector<TextureSlotData>& GetAllSlots() const { return _slots; }
    
    GLuint& GetLiveSSBO() { return _liveSSBO; }
    bool&   GetLiveDirty() { return _liveDirty; }
    
    void ClearRaycastState() {
        _validTriangles.clear();
        _validTriIDs.clear();
        _validTriangleIndicesSet.clear();
        _dijkstraPath.clear();
        _startNode = -1;
        _endNode = -1;
        _displayUVs.clear();
        _vertexData.clear();
        
        _projectedAppPoints.clear();
        _projectedGeoPoints.clear();
        _selectedGeoIndices.clear();
        _selectedAppIndices.clear();
        
        _extraNodePositions.clear();
        _extraNodeNormals.clear();
        _selectedTriangles.clear();
        _isSelecting = false;
        
        _projectionStats = ProjectionStats();

        if (_liveSSBO) { glDeleteBuffers(1, &_liveSSBO); _liveSSBO = 0; }
        _liveDirty = true;
        _suppressedUVs.clear();
    }
    
    void OnRaycastHit(const sibr::Vector3f& hitPos, float radius, int hitTriID) {
        if (!_mesh) return;
        
        sibr::Vector3f hitNormal(0, 1, 0);
        if (hitTriID >= 0 && hitTriID < _mesh->triangles().size()) {
            const auto& tri = _mesh->triangles()[hitTriID];
            const auto& n0 = _mesh->normals()[tri[0]];
            const auto& n1 = _mesh->normals()[tri[1]];
            const auto& n2 = _mesh->normals()[tri[2]];
            hitNormal = (n0 + n1 + n2) / 3.0f;
            hitNormal.normalize();
        }
        
        Compute(hitPos, hitNormal, radius, hitTriID);
    }


    void Init(const sibr::Mesh* mesh) {
        _mesh = mesh;
        buildBaseAdjacency();

        bool normalsAreZero = true;
        if (!_mesh->normals().empty()) {
            for (const auto& n : _mesh->normals()) {
                if (n.norm() > 1e-6f) {
                    normalsAreZero = false;
                    break;
                }
            }
        } else {
            normalsAreZero = true; 
        }

        if (normalsAreZero) {
            std::cout << "[INFO] ExpMapSolverSIBR: Detected zero or missing normals. Computing vertex normals..." << std::endl;
            computeVertexNormals();
        }
    }

    void RegisterAppPointCloud(const std::vector<float>& pointData, const std::vector<int>& fids,
                               const std::vector<GaussianProps>& props) {
        _appCloudData = pointData;
        _appCloudFids = fids;
        _appGaussianProps = props;
    }
    void RegisterGeoPointCloud(const std::vector<float>& pointData,
                               const std::vector<GaussianProps>& props) {
        _geoCloudData = pointData;
        _geoGaussianProps = props;
    }

    const std::vector<sibr::Vector3u>& GetActiveTriangles() const { return _validTriangles; }

    glm::vec3 getPos(int id) const {
        if (id < _mesh->vertices().size()) return toGlm(_mesh->vertices()[id]);
        else return _extraNodePositions[id - _mesh->vertices().size()];
    }

    glm::vec3 getNormal(int id) const {
        if (id < _mesh->vertices().size()) return toGlm(_mesh->normals()[id]);
        else return _extraNodeNormals[id - _mesh->vertices().size()];
    }

    void Compute(const sibr::Vector3f& hitPos, const sibr::Vector3f& hitNormal, float radius, int hitTriID = -1) {
        if(!_mesh) return;

        _vertexData.clear();
        _displayUVs.clear();
        _validTriangles.clear();
        _validTriIDs.clear();
        _validTriangleIndicesSet.clear();
        
        _projectedAppPoints.clear(); 
        _projectedGeoPoints.clear();
        _selectedGeoIndices.clear();
        _selectedAppIndices.clear();
        
        _extraNodePositions.clear(); 
        _extraNodeNormals.clear();
        _projectionStats = ProjectionStats();
        
        _adj = _baseAdj; 
        _startNode = -1; _endNode = -1;
        _dijkstraPath.clear();

        glm::vec3 target = toGlm(hitPos);

        auto comp = [&](int a, int b){ return _vertexData[a].cost > _vertexData[b].cost; };
        std::priority_queue<int, std::vector<int>, decltype(comp)> pq(comp);

        _seedFrame = TangentFrame(target, toGlm(hitNormal));

        if (hitTriID >= 0 && hitTriID < _mesh->triangles().size()) {
            const auto& tri = _mesh->triangles()[hitTriID];
            for (int k = 0; k < 3; ++k) {
                int vIdx = (int)tri[k];
                glm::vec3 vPos = toGlm(_mesh->vertices()[vIdx]);
                
                ExpVertex vSeed;
                vSeed.id = vIdx;
                vSeed.cost = glm::distance(target, vPos);
                
                glm::vec3 localPos = _seedFrame.toLocal(vPos - target);
                vSeed.uv = glm::vec2(localPos.x, localPos.y);
                vSeed.tangentX = _seedFrame.axes[0]; 
                
                _vertexData[vIdx] = vSeed;
                pq.push(vIdx);
            }
        } else {
            return;
        }

        float maxCostFound = 0.0f;

        while(!pq.empty()) {
            int currIdx = pq.top(); pq.pop();
            if(_vertexData[currIdx].frozen) continue;
            _vertexData[currIdx].frozen = true;

            if(_vertexData[currIdx].cost > radius) continue;

            // [事實修正 1] 嚴格禁止展開路徑越過與種子點法線夾角超過 81 度 (cos=0.15) 的區域。
            // 從物理源頭斬斷拓撲摺疊，強制放棄曲率過大的背面展開。
            glm::vec3 currN = glm::normalize(getNormal(currIdx));
            if (glm::dot(currN, toGlm(hitNormal)) < 0.15f) continue;

            maxCostFound = std::max(maxCostFound, _vertexData[currIdx].cost);

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

            refineUVsTriangleUnfolding(3);

            // 1. 計算點擊中心在局部展開空間中的座標 (作為原點偏移)
            glm::vec2 exactHitUV(0.0f, 0.0f);
            if (hitTriID >= 0 && hitTriID < _mesh->triangles().size()) {
                const auto& t = _mesh->triangles()[hitTriID];
                if (_vertexData.count(t[0]) && _vertexData.count(t[1]) && _vertexData.count(t[2])) {
                    glm::vec3 v0 = toGlm(_mesh->vertices()[t[0]]);
                    glm::vec3 v1 = toGlm(_mesh->vertices()[t[1]]);
                    glm::vec3 v2 = toGlm(_mesh->vertices()[t[2]]);
                    
                    glm::vec3 v0v1 = v1 - v0, v0v2 = v2 - v0, pVec = target - v0;
                    float d00 = glm::dot(v0v1, v0v1), d01 = glm::dot(v0v1, v0v2), d11 = glm::dot(v0v2, v0v2);
                    float d20 = glm::dot(pVec, v0v1), d21 = glm::dot(pVec, v0v2);
                    float denom = d00 * d11 - d01 * d01;
                    
                    if (std::abs(denom) > 1e-9f) {
                        float v = (d11 * d20 - d01 * d21) / denom;
                        float w = (d00 * d21 - d01 * d20) / denom;
                        float u = 1.0f - v - w;
                        exactHitUV = u * _vertexData[t[0]].uv + v * _vertexData[t[1]].uv + w * _vertexData[t[2]].uv;
                    }
                }
            }

            // 2. 將所有頂點 UV 移至以點擊點為中心，並計算展開形狀的包圍盒 (Bounding Box)
            glm::vec2 uvMin(1e9f), uvMax(-1e9f);
            for (auto& [id, vd] : _vertexData) {
                if (vd.frozen) {
                    vd.uv -= exactHitUV; 
                    uvMin = glm::min(uvMin, vd.uv);
                    uvMax = glm::max(uvMax, vd.uv);
                }
            }

            // 3. 計算自動縮放係數，確保解算範圍撐滿貼圖
            float uvWidth  = uvMax.x - uvMin.x;
            float uvHeight = uvMax.y - uvMin.y;
            float maxDim   = std::max(uvWidth, uvHeight);
            
            // 使用 0.95f 作為目標覆蓋率，保留 5% 邊界餘量以利 GUI 觀察
            _uvScale = (maxDim > 1e-7f) ? (0.95f / maxDim) : 1.0f;

            // 4. 計算形狀幾何中心，並映射至 [0, 1] 畫布空間
            glm::vec2 uvCenter = (uvMin + uvMax) * 0.5f;

            for (auto& [id, vd] : _vertexData) {
                if (vd.frozen) {
                    // 將形狀中心對齊畫布中心 (0.5, 0.5)
                    _displayUVs[id] = (vd.uv - uvCenter) * _uvScale + glm::vec2(0.5f, 0.5f);
                }
            }

            const auto& tris = _mesh->triangles();

            struct TriCandidate {
                sibr::Vector3u t;
                int   idx;
                float signedAreaUV;
                float maxEdgeRatio;
            };
            std::vector<TriCandidate> cands;
            cands.reserve(2048);

            for (size_t i = 0; i < tris.size(); ++i) {
                const auto& t = tris[i];
                if (!_displayUVs.count(t[0]) || !_displayUVs.count(t[1]) || !_displayUVs.count(t[2]))
                    continue;

                glm::vec2 uv0 = _displayUVs[t[0]], uv1 = _displayUVs[t[1]], uv2 = _displayUVs[t[2]];
                glm::vec3 p0  = toGlm(_mesh->vertices()[t[0]]);
                glm::vec3 p1  = toGlm(_mesh->vertices()[t[1]]);
                glm::vec3 p2  = toGlm(_mesh->vertices()[t[2]]);

                float d01 = glm::distance(p0,p1), d12 = glm::distance(p1,p2), d20 = glm::distance(p2,p0);
                if (d01 < 1e-6f && d12 < 1e-6f && d20 < 1e-6f) continue;

                float signedAreaUV = (uv1.x-uv0.x)*(uv2.y-uv0.y) - (uv1.y-uv0.y)*(uv2.x-uv0.x);
                if (std::abs(signedAreaUV) < 1e-6f) continue;

                float r01 = (d01 > 1e-6f) ? (glm::distance(uv0,uv1)/_uvScale) / d01 : 0.f;
                float r12 = (d12 > 1e-6f) ? (glm::distance(uv1,uv2)/_uvScale) / d12 : 0.f;
                float r20 = (d20 > 1e-6f) ? (glm::distance(uv2,uv0)/_uvScale) / d20 : 0.f;
                float maxR = std::max({r01, r12, r20});

                cands.push_back({t, (int)i, signedAreaUV, maxR});
            }

            int posW = 0, negW = 0;
            for (auto& c : cands) {
                if (c.signedAreaUV > 0.f) ++posW; else ++negW;
            }
            bool expectPos = (posW >= negW);

            std::vector<float> goodRatios;
            goodRatios.reserve(cands.size());
            for (auto& c : cands) {
                bool ok = expectPos ? (c.signedAreaUV > 0.f) : (c.signedAreaUV < 0.f);
                if (ok) goodRatios.push_back(c.maxEdgeRatio);
            }
            float medRatio = 1.0f;
            if (!goodRatios.empty()) {
                std::sort(goodRatios.begin(), goodRatios.end());
                medRatio = goodRatios[goodRatios.size() / 2];
            }
            _autoThreshold = std::min(std::max(1.5f, medRatio * 2.0f), 6.0f);

            for (auto& c : cands) {
                bool correctWind = expectPos ? (c.signedAreaUV > 0.f) : (c.signedAreaUV < 0.f);
                if (!correctWind)              continue;
                if (c.maxEdgeRatio > _autoThreshold) continue;
                _validTriangles.push_back(c.t);
                _validTriIDs.push_back(c.idx);
                _validTriangleIndicesSet.insert(c.idx);
            }

            ProjectAndInsertClouds(target, radius * 1.2f);
        }
    }

    void RenderUI() {
        ImGui::SetNextWindowSize(ImVec2(400, 450), ImGuiCond_FirstUseEver);
        if(ImGui::Begin("ExpMap UV Result", nullptr, ImGuiWindowFlags_MenuBar)) {
            
            if (ImGui::BeginMenuBar()) {
                if (ImGui::BeginMenu("File")) {
                    if (ImGui::BeginMenu("Load Background")) {
                        std::string targetDir = "D:/SGGaussians/SGGaussians/assets/texture/";
                        std::vector<std::string> imageFiles = _textureLoader.scanForImages(targetDir);
                        
                        if (imageFiles.empty()) {
                            ImGui::MenuItem("(No images in assets/texture)", NULL, false, false);
                        } else {
                            for (const auto& imagePath : imageFiles) {
                                std::string displayName = imagePath;
                                size_t lastSlash = displayName.find_last_of("/\\");
                                if (lastSlash != std::string::npos) {
                                    displayName = displayName.substr(lastSlash + 1);
                                }

                                if (ImGui::MenuItem(displayName.c_str())) {
                                    _textureLoader.LoadImage(imagePath);
                                    _textureDirty = true;
                                }
                            }
                        }
                        ImGui::EndMenu();
                    }
                    ImGui::EndMenu();
                }
                ImGui::EndMenuBar();
            }

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
            } else {
            ImGui::Text("Tris: %lu | App: %d/%d | Geo: %d/%d", 
                        _validTriangles.size(),
                        _projectionStats.appProjected,
                        _projectionStats.appTotal,
                        _projectionStats.geoProjected,
                        _projectionStats.geoTotal);
            
            
            ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), "Hold CTRL + Left Click to pick Start/End Points");
            ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "Hold SHIFT + Drag to select area");
            
            if (_dijkstraPath.size() > 0) {
                ImGui::TextColored(ImVec4(0,1,0,1), "Path Length: %lu hops", _dijkstraPath.size());
            }

            ImGui::SameLine();
            if (ImGui::Button("Reset View")) { _viewScale = 1.0f; _viewOffset = ImVec2(0,0); }
            } // end else (_displayUVs not empty)
            
            ImGui::Checkbox("Fill", &_drawFilled);
            ImGui::SameLine();
            ImGui::Checkbox("Cull", &_cullBackFace);
            ImGui::SameLine();
            ImGui::Checkbox("BG", &_showBackgroundTexture);
            ImGui::Checkbox("Show App", &_showAppPoints);
            ImGui::SameLine();
            ImGui::Checkbox("Show Geo", &_showGeoPoints);
            ImGui::SameLine();
            ImGui::Checkbox("Show Suppressed", &_showSuppressedPoints);

            ImVec2 p = ImGui::GetCursorScreenPos();
            ImVec2 sz = ImGui::GetContentRegionAvail();
            if(sz.x < 50) sz.x = 50; 
            if(sz.y < 50) sz.y = 50;
            float dim = std::min(sz.x, sz.y);

            ImGui::InvisibleButton("##uvcanvas", sz);
            bool isHovered = ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenBlockedByActiveItem);
            ImVec2 mousePos = ImGui::GetMousePos();

            if (isHovered) {
                float wheel = ImGui::GetIO().MouseWheel;
                if (wheel != 0.0f) {
                    // 1. 記錄縮放前滑鼠相對於畫布中心的距離
                    float lx = mousePos.x - (p.x + sz.x * 0.5f + _viewOffset.x);
                    float ly = mousePos.y - (p.y + sz.y * 0.5f + _viewOffset.y);
                    
                    // 2. 進行縮放
                    float zoomFactor = 1.1f;
                    float oldScale = _viewScale;
                    if (wheel < 0.0f) _viewScale /= zoomFactor;
                    else              _viewScale *= zoomFactor;
                    
                    // 3. 計算縮放比例並補償 Offset，確保滑鼠指著的 UV 點在縮放後依然在滑鼠下方
                    float ratio = _viewScale / oldScale;
                    _viewOffset.x -= lx * (ratio - 1.0f);
                    _viewOffset.y -= ly * (ratio - 1.0f);
                }
            }

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
                    UpdateSelectedGaussians(); 
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

            if (_showBackgroundTexture) {
                const auto& bgTex = _textureLoader.getTexture();
                if (bgTex && bgTex->handle() != 0) {
                    dl->AddImage((void*)(intptr_t)bgTex->handle(), p, ImVec2(p.x + sz.x, p.y + sz.y), ImVec2(0, 1), ImVec2(1, 0));
                } else {
                    dl->AddRectFilled(p, ImVec2(p.x + sz.x, p.y + sz.y), IM_COL32(40, 40, 40, 255));
                }
            } else {
                dl->AddRectFilled(p, ImVec2(p.x + sz.x, p.y + sz.y), IM_COL32(40, 40, 40, 255));
            }
            
            auto TransformUV = [&](const glm::vec2& uv) -> ImVec2 {
                float lx = (uv.x - 0.5f) * dim * _viewScale;
                float ly = (1.0f - uv.y - 0.5f) * dim * _viewScale; 
                return ImVec2(p.x + sz.x*0.5f + lx + _viewOffset.x, p.y + sz.y*0.5f + ly + _viewOffset.y);
            };

            // Safe margin beyond canvas edge at which we still allow drawing
            // (triangles that straddle the border need to be rendered).
            // Beyond CANVAS_GUARD pixels we hard-clamp to prevent ImGui
            // DrawList artifacts (white streaks, text ghosts) that occur when
            // vertex coordinates are extremely large (thousands of pixels away).
            const float CANVAS_GUARD = sz.x + sz.y;
            auto ClampPx = [&](ImVec2 v) -> ImVec2 {
                float xMin = p.x - CANVAS_GUARD, xMax = p.x + sz.x + CANVAS_GUARD;
                float yMin = p.y - CANVAS_GUARD, yMax = p.y + sz.y + CANVAS_GUARD;
                v.x = std::max(xMin, std::min(xMax, v.x));
                v.y = std::max(yMin, std::min(yMax, v.y));
                return v;
            };

            for(const auto& t : _validTriangles) {
                glm::vec2 uv1 = _displayUVs[t[0]];
                glm::vec2 uv2 = _displayUVs[t[1]];
                glm::vec2 uv3 = _displayUVs[t[2]];
                ImVec2 ip1 = TransformUV(uv1); ImVec2 ip2 = TransformUV(uv2); ImVec2 ip3 = TransformUV(uv3);

                // Full 4-side canvas cull: skip only when ALL 3 vertices are
                // outside the same edge of the canvas (completely off-screen).
                float canvasX0 = p.x, canvasX1 = p.x + sz.x;
                float canvasY0 = p.y, canvasY1 = p.y + sz.y;
                if (ip1.x < canvasX0 && ip2.x < canvasX0 && ip3.x < canvasX0) continue;
                if (ip1.x > canvasX1 && ip2.x > canvasX1 && ip3.x > canvasX1) continue;
                if (ip1.y < canvasY0 && ip2.y < canvasY0 && ip3.y < canvasY0) continue;
                if (ip1.y > canvasY1 && ip2.y > canvasY1 && ip3.y > canvasY1) continue;

                if (_cullBackFace) {
                    float area = (ip2.x - ip1.x) * (ip3.y - ip1.y) - (ip3.x - ip1.x) * (ip2.y - ip1.y);
                    if (area > 0.0f) continue;
                }
                {
                    auto edgeRatio = [&](unsigned int a, unsigned int b) -> float {
                        float d3D = glm::distance(toGlm(_mesh->vertices()[a]),
                                                  toGlm(_mesh->vertices()[b]));
                        if (d3D < 1e-6f) return 0.f;
                        return (glm::distance(_displayUVs[(int)a], _displayUVs[(int)b]) / _uvScale) / d3D;
                    };
                    float maxR = std::max({edgeRatio(t[0],t[1]),
                                          edgeRatio(t[1],t[2]),
                                          edgeRatio(t[2],t[0])});
                    if (maxR > _autoThreshold) continue;
                }

                // Clamp to guard region before submitting to ImGui DrawList.
                // This prevents the "white streaks / text ghost" artifact that
                // ImGui produces when triangle vertices are thousands of pixels
                // outside the window (a known ImGui DrawList limitation).
                ip1 = ClampPx(ip1); ip2 = ClampPx(ip2); ip3 = ClampPx(ip3);

                if (_drawFilled) dl->AddTriangleFilled(ip1, ip2, ip3, IM_COL32(255, 215, 0, 80));
                else dl->AddTriangle(ip1, ip2, ip3, IM_COL32(255, 215, 0, 200), 1.0f);
            }

            static bool showEllipses = true;
            static float ellipseAlpha = 0.1f;
            static float ellipseSigma = 3.0f; // 顯示 3-Sigma 分布範圍
            
            ImGui::Checkbox("Show Covariance Ellipses", &showEllipses);
            ImGui::SameLine();
            ImGui::PushItemWidth(100.0f);
            ImGui::SliderFloat("Alpha##ell", &ellipseAlpha, 0.0f, 1.0f);
            ImGui::PopItemWidth();
            ImGui::SameLine();
            ImGui::PushItemWidth(100.0f);
            ImGui::SliderFloat("Sigma##ell", &ellipseSigma, 1.0f, 5.0f);
            ImGui::PopItemWidth();

            auto drawRotatedEllipse = [&](ImDrawList* dl, ImVec2 center, float a, float b, float theta, ImU32 color, ImU32 fillColor) {
                // [事實修正 1：最小可視保證] 
                // 移除原有的 return 丟棄機制。若數值崩潰 (NaN) 或極小，強制賦予 0.5 像素的最小半徑。
                if (std::isnan(a) || std::isnan(b)) { a = 0.5f; b = 0.5f; }
                a = std::max(a, 0.5f);
                b = std::max(b, 0.5f);
                
                int num_segments = 6;
                float max_r = std::max(a, b);
                if (max_r > 4.0f)  num_segments = 10;
                if (max_r > 12.0f) num_segments = 16;
                if (max_r > 30.0f) num_segments = 24;
                if (max_r > 100.0f) num_segments = 36; 

                float cos_th = cosf(theta), sin_th = sinf(theta);
                std::vector<ImVec2> points(num_segments);
                for (int i = 0; i < num_segments; ++i) {
                    float phi = 2.0f * IM_PI * (float)i / (float)num_segments;
                    float ex = a * cosf(phi), ey = b * sinf(phi);
                    points[i] = ImVec2(center.x + ex * cos_th - ey * sin_th, center.y + ex * sin_th + ey * cos_th);
                }
                dl->AddConvexPolyFilled(points.data(), num_segments, fillColor);
                dl->AddPolyline(points.data(), num_segments, color, true, 1.0f);
            };

            auto computeUVEllipse = [&](const ProjectedGaussian& pt, float& a_uv, float& b_uv, float& angle) -> bool {
                glm::vec3 s = glm::exp(pt.scale);
                float r = pt.rotation.x, x = pt.rotation.y, y = pt.rotation.z, z = pt.rotation.w; 
                glm::mat3 R = glm::mat3(
                    1.f-2.f*(y*y+z*z), 2.f*(x*y+r*z), 2.f*(x*z-r*y),
                    2.f*(x*y-r*z), 1.f-2.f*(x*x+z*z), 2.f*(y*z+r*x),
                    2.f*(x*z+r*y), 2.f*(y*z-r*x), 1.f-2.f*(x*x+y*y)
                );
                glm::mat3 Sigma3D = R * glm::mat3(s.x*s.x, 0, 0,  0, s.y*s.y, 0,  0, 0, s.z*s.z) * glm::transpose(R);

                glm::vec3 N = glm::cross(pt.dU, pt.dV);
                float len2N = glm::dot(N, N);
                glm::vec3 rowU(0.f), rowV(0.f);
                
                if (len2N > 1e-12f) {
                    rowU = glm::cross(pt.dV, N) / len2N;
                    rowV = glm::cross(N, pt.dU) / len2N;
                } else {
                    float lU = glm::dot(pt.dU, pt.dU), lV = glm::dot(pt.dV, pt.dV);
                    rowU = (lU > 1e-12f) ? pt.dU / lU : glm::vec3(0.f);
                    rowV = (lV > 1e-12f) ? pt.dV / lV : glm::vec3(0.f);
                }

                float c00 = glm::dot(rowU, Sigma3D * rowU);
                float c01 = glm::dot(rowU, Sigma3D * rowV);
                float c11 = glm::dot(rowV, Sigma3D * rowV);

                float Trace = c00 + c11, Det = c00 * c11 - c01 * c01;
                float detSqrt = std::sqrt(std::max(0.0f, Trace * Trace / 4.0f - Det));
                a_uv = std::sqrt(std::max(0.0f, Trace / 2.0f + detSqrt));
                b_uv = std::sqrt(std::max(0.0f, Trace / 2.0f - detSqrt));
                angle = 0.5f * std::atan2(2.0f * c01, c00 - c11);

                a_uv = std::min(a_uv, 5.0f);
                b_uv = std::min(b_uv, 5.0f);
                
                return true;
            };

            // 取得畫布實體邊界
            float canvasX0 = p.x, canvasX1 = p.x + sz.x;
            float canvasY0 = p.y, canvasY1 = p.y + sz.y;

            // 繪製 Geo 點 (紅色) 與橢圓
            if (_showGeoPoints) {
                for (const auto& pt : _projectedGeoPoints) {
                    // [絕對事實 3：嚴格 UV 範圍剔除] 飛出貼圖邊界的無效點直接捨棄
                    // 寬鬆 UV 範圍：邊緣投影點可能稍微超出 [0,1]，仍需顯示
                    if (pt.uv.x < -0.15f || pt.uv.x > 1.15f || pt.uv.y < -0.15f || pt.uv.y > 1.15f) continue;
                    
                    float a_uv = 0.f, b_uv = 0.f, angle = 0.f;
                    bool validEllipse = false;
                    if (showEllipses) {
                        validEllipse = computeUVEllipse(pt, a_uv, b_uv, angle);
                    }

                    float canvasScale = dim * _viewScale * ellipseSigma;
                    float maxRadiusPx = validEllipse ? (std::max(a_uv, b_uv) * canvasScale) : 3.0f;
                    
                    ImVec2 pos = TransformUV(pt.uv);
                    
                    if (pos.x + maxRadiusPx < canvasX0 || pos.x - maxRadiusPx > canvasX1 ||
                        pos.y + maxRadiusPx < canvasY0 || pos.y - maxRadiusPx > canvasY1) {
                        continue;
                    }
                    
                    bool isSelected = std::find(_selectedGeoIndices.begin(), _selectedGeoIndices.end(), pt.originalIndex) != _selectedGeoIndices.end();
                    ImU32 baseColor = isSelected ? IM_COL32(255, 255, 0, 255) : IM_COL32(255, 50, 50, 255);
                    
                    if (validEllipse) {
                        ImU32 strokeColor = isSelected ? IM_COL32(255, 255, 0, 200) : IM_COL32(255, 50, 50, 200);
                        ImU32 fillColor   = isSelected ? IM_COL32(255, 255, 0, (int)(ellipseAlpha * 255)) : IM_COL32(255, 50, 50, (int)(ellipseAlpha * 255));
                        drawRotatedEllipse(dl, pos, a_uv * canvasScale, b_uv * canvasScale, -angle, strokeColor, fillColor);
                    }
                    dl->AddCircleFilled(pos, isSelected ? 3.0f : 1.5f, baseColor);
                }
            }

            // 繪製 App 點 (藍色) 與橢圓
            if (_showAppPoints) {
                for (const auto& pt : _projectedAppPoints) {
                    if (pt.uv.x < -0.15f || pt.uv.x > 1.15f || pt.uv.y < -0.15f || pt.uv.y > 1.15f) continue;
                    
                    float a_uv = 0.f, b_uv = 0.f, angle = 0.f;
                    bool validEllipse = false;
                    if (showEllipses) {
                        validEllipse = computeUVEllipse(pt, a_uv, b_uv, angle);
                    }

                    float canvasScale = dim * _viewScale * ellipseSigma;
                    float maxRadiusPx = validEllipse ? (std::max(a_uv, b_uv) * canvasScale) : 3.0f;
                    
                    ImVec2 pos = TransformUV(pt.uv);
                    
                    if (pos.x + maxRadiusPx < canvasX0 || pos.x - maxRadiusPx > canvasX1 ||
                        pos.y + maxRadiusPx < canvasY0 || pos.y - maxRadiusPx > canvasY1) {
                        continue;
                    }
                    
                    bool isSelected = std::find(_selectedAppIndices.begin(), _selectedAppIndices.end(), pt.originalIndex) != _selectedAppIndices.end();
                    ImU32 baseColor = isSelected ? IM_COL32(255, 255, 0, 255) : IM_COL32(0, 150, 255, 255);
                    
                    if (validEllipse) {
                        ImU32 strokeColor = isSelected ? IM_COL32(255, 255, 0, 200) : IM_COL32(0, 150, 255, 200);
                        ImU32 fillColor   = isSelected ? IM_COL32(255, 255, 0, (int)(ellipseAlpha * 255)) : IM_COL32(0, 150, 255, (int)(ellipseAlpha * 255));
                        drawRotatedEllipse(dl, pos, a_uv * canvasScale, b_uv * canvasScale, -angle, strokeColor, fillColor);
                    }
                    dl->AddCircleFilled(pos, isSelected ? 3.0f : 1.5f, baseColor);
                }
            }

            // Draw suppressed Gaussians as green dots on the UV canvas.
            // These are in-sphere Gaussians that failed triangle projection,
            // so they have no texture UV and were suppressed (opacity = 0).
            // Their UV is computed by clamping to the nearest triangle boundary.
            if (_showSuppressedPoints) {
                for (const auto& uv : _suppressedUVs) {
                    if (uv.x < -1.0f || uv.x > 2.0f || uv.y < -1.0f || uv.y > 2.0f) continue;
                    ImVec2 pos = TransformUV(uv);
                    // Outer ring (darker green) + inner fill (bright green)
                    dl->AddCircleFilled(pos, 4.0f, IM_COL32(0, 180, 0, 200));
                    dl->AddCircleFilled(pos, 2.5f, IM_COL32(0, 255, 80, 255));
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
            
            bool in0 = (uv0.x >= minX && uv0.x <= maxX && uv0.y >= minY && uv0.y <= maxY);
            bool in1 = (uv1.x >= minX && uv1.x <= maxX && uv1.y >= minY && uv1.y <= maxY);
            bool in2 = (uv2.x >= minX && uv2.x <= maxX && uv2.y >= minY && uv2.y <= maxY);
            
            if (in0 || in1 || in2) {
                _selectedTriangles.push_back(t);
            }
        }
    }

    void UpdateSelectedGaussians() {
        _selectedGeoIndices.clear();
        _selectedAppIndices.clear();

        float minX = std::min(_selectionStart.x, _selectionEnd.x);
        float maxX = std::max(_selectionStart.x, _selectionEnd.x);
        float minY = std::min(_selectionStart.y, _selectionEnd.y);
        float maxY = std::max(_selectionStart.y, _selectionEnd.y);

        for (const auto& geoPt : _projectedGeoPoints) {
            if (geoPt.uv.x >= minX && geoPt.uv.x <= maxX && 
                geoPt.uv.y >= minY && geoPt.uv.y <= maxY) {
                _selectedGeoIndices.push_back(geoPt.originalIndex);
            }
        }

        for (const auto& appPt : _projectedAppPoints) {
            if (appPt.uv.x >= minX && appPt.uv.x <= maxX && 
                appPt.uv.y >= minY && appPt.uv.y <= maxY) {
                _selectedAppIndices.push_back(appPt.originalIndex);
            }
        }

        std::cout << "[INFO] Selected GEO Points Count: " << _selectedGeoIndices.size() << "\n";
        std::cout << "[INFO] Selected APP Points Count: " << _selectedAppIndices.size() << std::endl;
    }

    void computeVertexNormals() {
        if (!_mesh || _mesh->vertices().empty() || _mesh->triangles().empty()) {
            std::cerr << "[WARNING] Cannot compute normals: Mesh is invalid or empty." << std::endl;
            return;
        }

        std::vector<sibr::Vector3f>& normals = const_cast<std::vector<sibr::Vector3f>&>(_mesh->normals()); 
        if (normals.size() != _mesh->vertices().size()) {
             normals.assign(_mesh->vertices().size(), sibr::Vector3f(0.0f, 0.0f, 0.0f));
        } else {
            std::fill(normals.begin(), normals.end(), sibr::Vector3f(0.0f, 0.0f, 0.0f)); 
        }

        const auto& vertices = _mesh->vertices();
        const auto& triangles = _mesh->triangles();

        for (const auto& tri : triangles) {
            if (tri.x() >= vertices.size() || tri.y() >= vertices.size() || tri.z() >= vertices.size()) {
                continue;
            }

            const sibr::Vector3f& v0 = vertices[tri.x()];
            const sibr::Vector3f& v1 = vertices[tri.y()];
            const sibr::Vector3f& v2 = vertices[tri.z()];

            sibr::Vector3f edge1 = v1 - v0;
            sibr::Vector3f edge2 = v2 - v0;
            sibr::Vector3f faceNormal = edge1.cross(edge2);

            if (faceNormal.norm() > 1e-6f) {
                faceNormal.normalize();
                normals[tri.x()] += faceNormal;
                normals[tri.y()] += faceNormal;
                normals[tri.z()] += faceNormal;
            }
        }

        for (sibr::Vector3f& n : normals) {
            if (n.norm() > 1e-6f) n.normalize();
            else n = sibr::Vector3f(0.0f, 1.0f, 0.0f); 
        }
    }
    
    void refineUVsTriangleUnfolding(int iterations = 8) {
        const int meshSize = (int)_mesh->vertices().size();
        std::vector<std::pair<float, int>> byDist;
        for (auto& [id, vd] : _vertexData) {
            if (id < meshSize && vd.frozen) byDist.push_back({vd.cost, id});
        }
        std::sort(byDist.begin(), byDist.end());

        for (int iter = 0; iter < iterations; ++iter) {
            float blend = 0.5f + 0.3f * (float)iter / (float)iterations;

            for (auto& [cost, vIdx] : byDist) {
                if (cost < 1e-7f) continue;
                if (vIdx >= (int)_vertexToTriangles.size()) continue;

                glm::vec2 uvAccum(0.f, 0.f);
                float     wAccum = 0.f;

                for (int triIdx : _vertexToTriangles[vIdx]) {
                    const auto& tri = _mesh->triangles()[triIdx];
                    int vA = -1, vB = -1;
                    for (int k = 0; k < 3; ++k) {
                        if ((int)tri[k] != vIdx) {
                            if (vA < 0) vA = (int)tri[k]; else vB = (int)tri[k];
                        }
                    }
                    
                    auto itA = _vertexData.find(vA);
                    auto itB = _vertexData.find(vB);
                    if (itA == _vertexData.end() || itB == _vertexData.end() || 
                        !itA->second.frozen || !itB->second.frozen) continue;

                    
                    glm::vec3 pV = getPos(vIdx), pA = getPos(vA), pB = getPos(vB);
                    float dAB = glm::distance(pA, pB);
                    float dVA = glm::distance(pV, pA);
                    float dVB = glm::distance(pV, pB);
                    if (dAB < 1e-7f) continue;

                    float cosA = glm::clamp((dVA*dVA + dAB*dAB - dVB*dVB) / (2.f * dVA * dAB), -1.f, 1.f);
                    float sinA = std::sqrt(1.f - cosA * cosA);

                    glm::vec2 uvA = itA->second.uv, uvB = itB->second.uv;
                    glm::vec2 edgeUV = uvB - uvA;
                    float L_uv = glm::length(edgeUV);
                    if (L_uv < 1e-8f) continue;

                    float localScale = L_uv / dAB;
                    glm::vec2 dir = edgeUV / L_uv;
                    glm::vec2 perp(-dir.y, dir.x);

                    glm::vec3 nA = glm::normalize(getNormal(vA));
                    float side = glm::dot(glm::cross(pB - pA, pV - pA), nA);
                    glm::vec2 uvEst = uvA + dir * (dVA * cosA * localScale) + 
                                    perp * (dVA * sinA * localScale * (side >= 0.f ? 1.f : -1.f));

                    float w = std::max(0.1f, glm::dot(nA, glm::normalize(getNormal(vIdx))));
                    uvAccum += uvEst * w;
                    wAccum += w;
                }

                if (wAccum > 1e-6f) {
                    _vertexData[vIdx].uv = glm::mix(_vertexData[vIdx].uv, uvAccum / wAccum, blend);
                }
            }
        }
    }

    void buildBaseAdjacency() {
        const int N = (int)_mesh->vertices().size();
        _baseAdj.assign(N, {});
        _vertexToTriangles.assign(N, {});

        const auto& tris = _mesh->triangles();
        for (size_t ti = 0; ti < tris.size(); ++ti) {
            int a = (int)tris[ti][0], b = (int)tris[ti][1], c = (int)tris[ti][2];

            auto addEdge = [&](int u, int v) {
                if (std::find(_baseAdj[u].begin(), _baseAdj[u].end(), v) == _baseAdj[u].end())
                    _baseAdj[u].push_back(v);
                if (std::find(_baseAdj[v].begin(), _baseAdj[v].end(), u) == _baseAdj[v].end())
                    _baseAdj[v].push_back(u);
            };
            addEdge(a, b); addEdge(b, c); addEdge(c, a);

            _vertexToTriangles[a].push_back((int)ti);
            _vertexToTriangles[b].push_back((int)ti);
            _vertexToTriangles[c].push_back((int)ti);
        }
    }

    void propagate(int parentIdx, int currIdx) {
        ExpVertex& parent = _vertexData[parentIdx];
        ExpVertex& curr   = _vertexData[currIdx];

        glm::vec3 posP = getPos(parentIdx);
        glm::vec3 posC = getPos(currIdx);

        float edgeLen = glm::distance(posP, posC);
        float newCost = parent.cost + edgeLen;
        if (newCost >= curr.cost) return;
        curr.cost     = newCost;
        curr.parentId = parentIdx;

        if (parentIdx >= (int)_mesh->vertices().size() || currIdx >= (int)_mesh->vertices().size()) return;

        glm::vec3 nP  = glm::normalize(getNormal(parentIdx));
        glm::vec3 txP = parent.tangentX; 
        glm::vec3 tyP = glm::normalize(glm::cross(nP, txP));

        glm::vec3 edge = posC - posP;
        
        glm::vec3 edgeProj = edge - nP * glm::dot(edge, nP);
        if (glm::length(edgeProj) > 1e-8f) {
            edgeProj = glm::normalize(edgeProj) * edgeLen; 
        }

        glm::vec2 deltaUV(glm::dot(edgeProj, txP), glm::dot(edgeProj, tyP));
        curr.uv = parent.uv + deltaUV;

        glm::vec3 nC      = glm::normalize(getNormal(currIdx));
        glm::vec3 rotAxis = glm::cross(nP, nC);
        float     axisLen = glm::length(rotAxis);

        glm::vec3 newTx;
        if (axisLen < 1e-8f) {
            newTx = (glm::dot(nP, nC) >= 0.0f) ? txP : -txP;
        } else {
            rotAxis /= axisLen;
            float cosA = glm::clamp(glm::dot(nP, nC), -1.0f, 1.0f);
            float sinA = std::sqrt(std::max(0.0f, 1.0f - cosA * cosA));
            
            newTx = txP * cosA + glm::cross(rotAxis, txP) * sinA + 
                    rotAxis * (glm::dot(rotAxis, txP) * (1.0f - cosA));
        }

        newTx -= nC * glm::dot(newTx, nC);
        curr.tangentX = glm::normalize(newTx);
    }

    void ProjectAndInsertClouds(const glm::vec3& center, float radius) {
        std::map<int, std::vector<int>> triToExtraNodes;

        auto process = [&](const std::vector<float>& cloud, const std::vector<int>& fids,
                          const std::vector<GaussianProps>& props,
                          std::vector<ProjectedGaussian>& outPts, const char* cloudName) {
            
            outPts.clear();
            if (cloud.empty()) return;
            
            size_t numPoints = cloud.size() / 3;
            float r2 = radius * radius;
            size_t meshSize = _mesh->vertices().size();
            bool use_fids = !fids.empty() && fids.size() == numPoints;

            int outOfRadius = 0;
            int invalidFid = 0;
            int fidOutOfRange = 0;
            int projected = 0;

            for (size_t i = 0; i < numPoints; ++i) {
                glm::vec3 pt(cloud[i*3], cloud[i*3+1], cloud[i*3+2]);
                
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

                // ── 共用：在所有有效三角形裡找最近投影 ───────────────────────────
                // 供 GEO 路徑直接使用，以及 APP 路徑在 fid 無效/超出範圍時 fallback 使用。
                // 平面距離容忍度改為 0.3 * radius（原為 0.1），讓稍微懸浮在表面上的
                // 邊緣 Gaussian 也能投影成功。
                auto searchNearestValidTri = [&]() {
                    for (int globalTIdx : _validTriIDs) {
                        const auto& t = _mesh->triangles()[globalTIdx];
                        glm::vec3 v0 = toGlm(_mesh->vertices()[t[0]]);
                        glm::vec3 v1 = toGlm(_mesh->vertices()[t[1]]);
                        glm::vec3 v2 = toGlm(_mesh->vertices()[t[2]]);
                        glm::vec3 n = glm::cross(v1 - v0, v2 - v0);
                        if (glm::length(n) < 1e-9f) continue;
                        n = glm::normalize(n);
                        if (std::abs(glm::dot(pt - v0, n)) > 0.3f * radius) continue;
                        glm::vec3 pProj = pt - n * glm::dot(pt - v0, n);

                        glm::vec3 v0v1 = v1 - v0, v0v2 = v2 - v0, pVec = pProj - v0;
                        float d00 = glm::dot(v0v1, v0v1), d01 = glm::dot(v0v1, v0v2), d11 = glm::dot(v0v2, v0v2);
                        float d20 = glm::dot(pVec, v0v1), d21 = glm::dot(pVec, v0v2);
                        float denom = d00 * d11 - d01 * d01;
                        if (std::abs(denom) < 1e-9f) continue;

                        float bv = (d11 * d20 - d01 * d21) / denom;
                        float bw = (d00 * d21 - d01 * d20) / denom;
                        float bu = 1.0f - bv - bw;
                        // 允許稍微超出三角形邊界（-0.05 容差），讓邊緣的 Gaussian 能投影
                        if (bu < -0.05f || bv < -0.05f || bw < -0.05f) continue;
                        // 夾回 [0,1] 保證 UV 插值不會飛出範圍
                        bu = glm::clamp(bu, 0.f, 1.f);
                        bv = glm::clamp(bv, 0.f, 1.f);
                        bw = glm::clamp(bw, 0.f, 1.f);
                        float bsum = bu + bv + bw;
                        if (bsum > 1e-9f) { bu /= bsum; bv /= bsum; bw /= bsum; }

                        float dist = glm::distance(pt, pProj);
                        if (dist < bestDist) {
                            bestDist      = dist;
                            bestUV        = bu * _displayUVs.at(t[0]) + bv * _displayUVs.at(t[1]) + bw * _displayUVs.at(t[2]);
                            bestProjPos   = pProj;
                            bestTriGlobalID = globalTIdx;
                            found         = true;
                            glm::vec3 n0 = toGlm(_mesh->normals()[t[0]]);
                            glm::vec3 n1 = toGlm(_mesh->normals()[t[1]]);
                            glm::vec3 n2 = toGlm(_mesh->normals()[t[2]]);
                            bestProjNormal = glm::normalize(bu * n0 + bv * n1 + bw * n2);
                        }
                    }
                };

                if (use_fids) {
                    int face_id = fids[i];

                    if (face_id < 0 || face_id >= (int)_mesh->triangles().size()) {
                        // fid 完全無效 → 搜尋最近有效三角形
                        invalidFid++;
                        searchNearestValidTri();
                    } else if (!_validTriangleIndicesSet.count(face_id)) {
                        // fid 指向的三角形被 edge-ratio / winding 過濾掉
                        // (邊緣/角落三角形最常發生) → fallback 搜尋最近有效三角形
                        fidOutOfRange++;
                        searchNearestValidTri();
                    } else {
                        // 正常路徑：直接用 fid 三角形做重心插值
                        const auto& t = _mesh->triangles()[face_id];
                        glm::vec3 v0 = toGlm(_mesh->vertices()[t[0]]);
                        glm::vec3 v1 = toGlm(_mesh->vertices()[t[1]]);
                        glm::vec3 v2 = toGlm(_mesh->vertices()[t[2]]);
                        glm::vec3 pVec = pt - v0;
                        glm::vec3 v0v1 = v1 - v0, v0v2 = v2 - v0;
                        float d00 = glm::dot(v0v1, v0v1), d01 = glm::dot(v0v1, v0v2), d11 = glm::dot(v0v2, v0v2);
                        float d20 = glm::dot(pVec, v0v1), d21 = glm::dot(pVec, v0v2);
                        float denom = d00 * d11 - d01 * d01;
                        if (std::abs(denom) > 1e-9f) {
                            float bv = (d11 * d20 - d01 * d21) / denom;
                            float bw = (d00 * d21 - d01 * d20) / denom;
                            float bu = 1.0f - bv - bw;
                            bestUV        = bu * _displayUVs.at(t[0]) + bv * _displayUVs.at(t[1]) + bw * _displayUVs.at(t[2]);
                            bestProjPos   = bu * v0 + bv * v1 + bw * v2;
                            glm::vec3 n0  = toGlm(_mesh->normals()[t[0]]);
                            glm::vec3 n1  = toGlm(_mesh->normals()[t[1]]);
                            glm::vec3 n2  = toGlm(_mesh->normals()[t[2]]);
                            bestProjNormal = glm::normalize(bu * n0 + bv * n1 + bw * n2);
                            bestTriGlobalID = face_id;
                            found = true;
                        }
                    }
                } else {
                    searchNearestValidTri();
                }

                if (found) {
                    const auto& t_orig = _mesh->triangles()[bestTriGlobalID];
                    glm::vec3 v0_tri = toGlm(_mesh->vertices()[t_orig[0]]);
                    glm::vec3 v1_tri = toGlm(_mesh->vertices()[t_orig[1]]);
                    glm::vec3 v2_tri = toGlm(_mesh->vertices()[t_orig[2]]);
                    glm::vec2 uv0_tri = _displayUVs.at(t_orig[0]);
                    glm::vec2 uv1_tri = _displayUVs.at(t_orig[1]);
                    glm::vec2 uv2_tri = _displayUVs.at(t_orig[2]);

                    glm::vec3 E1 = v1_tri - v0_tri;
                    glm::vec3 E2 = v2_tri - v0_tri;
                    glm::vec2 dUV1 = uv1_tri - uv0_tri;
                    glm::vec2 dUV2 = uv2_tri - uv0_tri;

                    float du1 = dUV1.x, dv1 = dUV1.y;
                    float du2 = dUV2.x, dv2 = dUV2.y;
                    float det2 = du1 * dv2 - du2 * dv1;
                    glm::vec3 dU(0.f), dV(0.f);
                    
                    // [必須還原] 刪除所有對 dU, dV 的長度鉗制
                    if (std::abs(det2) > 1e-9f) {
                        dU = ( dv2 * E1 - dv1 * E2) / det2;
                        dV = (-du2 * E1 + du1 * E2) / det2;
                    }
                    ProjectedGaussian pg;
                    pg.originalIndex = (int)i;
                    pg.uv            = bestUV;
                    pg.position      = bestProjPos; 
                    pg.originalPos   = pt;
                    pg.dU            = dU;          
                    pg.dV            = dV;          
                    
                    if (i < props.size()) {
                        pg.opacity   = props[i].opacity;
                        pg.scale     = glm::vec3(props[i].scale[0], props[i].scale[1], props[i].scale[2]);
                        pg.rotation  = glm::vec4(props[i].rot[0], props[i].rot[1], props[i].rot[2], props[i].rot[3]);
                    } else {
                        pg.opacity  = 0.0f;
                        pg.scale    = glm::vec3(1.0f);
                        pg.rotation = glm::vec4(1.0f, 0.0f, 0.0f, 0.0f);
                    }
                    outPts.push_back(pg);
                    
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
                    
                    projected++;
                }
            }
            
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
        };

        process(_geoCloudData, {}, _geoGaussianProps, _projectedGeoPoints, "GEO");
        process(_appCloudData, _appCloudFids, _appGaussianProps, _projectedAppPoints, "APP");

        _liveDirty = true;
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
        }
    }

    const sibr::Mesh* _mesh = nullptr;
    std::vector<std::vector<int>> _baseAdj;
    std::vector<std::vector<int>> _adj;
    std::vector<std::vector<int>> _vertexToTriangles;
    std::map<int, ExpVertex>      _vertexData;
    TangentFrame                  _seedFrame;

    std::vector<sibr::Vector3u>   _validTriangles;
    std::vector<int>              _validTriIDs;
    std::set<int>                 _validTriangleIndicesSet;

    std::vector<glm::vec3>        _extraNodePositions;
    std::vector<glm::vec3>        _extraNodeNormals;

    float  _uvScale       = 1.0f;
    ImVec2 _viewOffset    = ImVec2(0, 0);
    float  _viewScale     = 1.0f;
    bool   _drawFilled    = true;
    bool   _cullBackFace  = true;
    bool   _showBackgroundTexture = true;
    
    bool   _showAppPoints = true;
    bool   _showGeoPoints = true;

    std::vector<ProjectedGaussian> _projectedAppPoints;
    std::vector<ProjectedGaussian> _projectedGeoPoints;
    std::vector<int> _selectedGeoIndices;
    std::vector<int> _selectedAppIndices;
    
    std::vector<int> _dijkstraPath;
    int _startNode = -1;
    int _endNode = -1;

    std::map<int, glm::vec2> _displayUVs;
    std::vector<sibr::Vector3u> _selectedTriangles;
    glm::vec2 _selectionStart, _selectionEnd;
    bool _isSelecting = false;
    
    TextureLoader _textureLoader;
    bool _textureDirty = false;
    std::vector<TextureSlotData> _slots;
    std::vector<ProjectedGaussian> _pendingMainGaussians;
    
    std::vector<float>        _appCloudData;
    std::vector<int>          _appCloudFids;
    std::vector<GaussianProps> _appGaussianProps;
    std::vector<float>        _geoCloudData;
    std::vector<GaussianProps> _geoGaussianProps;

    float _autoThreshold = 3.0f;
    
    ProjectionStats _projectionStats;

    GLuint _liveSSBO = 0;
    bool   _liveDirty = true;

    // UV positions of Gaussians that are in-sphere but got no texture UV.
    // Shown as green dots on the UV canvas.
    std::vector<glm::vec2> _suppressedUVs;
    bool _showSuppressedPoints = true;
};

#endif