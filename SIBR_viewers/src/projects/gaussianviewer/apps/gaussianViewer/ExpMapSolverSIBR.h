#ifndef EXPMAP_SOLVER_SIBR_H
#define EXPMAP_SOLVER_SIBR_H

#include <core/graphics/Mesh.hpp>
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>
#include <queue>
#include <set>
#include <map>
#include <algorithm>
#include <numeric>
#include <limits>
#include <iostream>
#include <cstring>
#include <cmath>
#include <functional>
#include <imgui/imgui.h>
#include "texture.h"

#ifndef IM_PI
#define IM_PI 3.14159265358979323846f
#endif

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

    sibr::Texture2DRGBA::Ptr GetCudaTexture() const { return _cudaTexPtr; }

    bool ConsumeTextureDirty() {
        if (_textureDirty) { _textureDirty = false; return true; }
        return false;
    }

    float GetSurfaceBlend() const { return _surfaceBlend; }
    bool ConsumeSurfaceBlendDirty() {
        bool d = _surfaceBlendDirty;
        _surfaceBlendDirty = false;
        return d;
    }
    void ResetSurfaceBlend() { _surfaceBlend = 1.0f; _surfaceBlendDirty = false; }

    float GetSurfDistThreshold() const { return _surfDistThreshold; }
    bool ConsumeSurfDistThresholdDirty() {
        bool d = _surfDistThresholdDirty;
        _surfDistThresholdDirty = false;
        return d;
    }

    const ProjectionStats& GetProjectionStats() const { return _projectionStats; }

    void SetMainGaussiansForNextSave(const std::vector<ProjectedGaussian>& g) {
        _pendingMainGaussians = g;
    }

    void SetSuppressedUVs(const std::vector<glm::vec2>& uvs) {
        _suppressedUVs = uvs;
    }
    void ClearSuppressedUVs() { _suppressedUVs.clear(); }

    int SaveCurrentSlot(const std::string& name = "") {
        if (_validTriangles.empty() || !_textureLoader.getTexture()) return -1;
        TextureSlotData slot;
        slot.triangles   = _validTriangles;
        slot.uvMap       = _displayUVs;
        slot.texture     = _cudaTexPtr ? _cudaTexPtr : _textureLoader.getTexture();
        slot.texturePath = _textureLoader.getPath();
        slot.alpha       = 1.f;
        slot.visible     = true;

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

        _cudaTexPtr = nullptr;
        if (_liveSSBO) { glDeleteBuffers(1, &_liveSSBO); _liveSSBO = 0; }
        _liveDirty = true;
        _suppressedUVs.clear();

        _surfDistThreshold      = 1e9f;
        _computedMaxSurfDist    = 1.0f;
        _surfDistThresholdDirty = false;
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
            glm::vec3 currN = glm::normalize(getNormal(currIdx));
            if (glm::dot(currN, toGlm(hitNormal)) < 0.0f) continue;

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

            glm::vec2 uvMin(1e9f), uvMax(-1e9f);
            for (auto& [id, vd] : _vertexData) {
                if (vd.frozen) {
                    vd.uv -= exactHitUV; 
                    uvMin = glm::min(uvMin, vd.uv);
                    uvMax = glm::max(uvMax, vd.uv);
                }
            }

            float uvWidth  = uvMax.x - uvMin.x;
            float uvHeight = uvMax.y - uvMin.y;
            float maxDim   = std::max(uvWidth, uvHeight);
            
            _uvScale = (maxDim > 1e-7f) ? (0.95f / maxDim) : 1.0f;

            glm::vec2 uvCenter = (uvMin + uvMax) * 0.5f;

            for (auto& [id, vd] : _vertexData) {
                if (vd.frozen) {
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
            _autoThreshold = std::min(std::max(1.5f, medRatio * 2.0f), 12.0f);

            // Pass 1: filter triangles with wrong winding or too-large edge ratio
            struct PassedCand {
                sibr::Vector3u t;
                int   idx;
                float distFromCenter; // distance from UV centroid to center (0.5,0.5), for sorting
            };
            std::vector<PassedCand> passed;
            passed.reserve(cands.size());

            for (auto& c : cands) {
                bool correctWind = expectPos ? (c.signedAreaUV > 0.f) : (c.signedAreaUV < 0.f);
                if (!correctWind)              continue;
                if (c.maxEdgeRatio > _autoThreshold) continue;

                // UV centroid distance to (0.5, 0.5)
                glm::vec2 uv0 = _displayUVs[c.t[0]], uv1 = _displayUVs[c.t[1]], uv2 = _displayUVs[c.t[2]];
                glm::vec2 centroid = (uv0 + uv1 + uv2) / 3.f;
                float dist = glm::distance(centroid, glm::vec2(0.5f, 0.5f));
                passed.push_back({c.t, c.idx, dist});
            }

            // Sort near-to-far to prioritize triangles close to the hit point
            std::sort(passed.begin(), passed.end(),
                      [](const PassedCand& a, const PassedCand& b){ return a.distFromCenter < b.distFromCenter; });

            // Accept all triangles that passed winding + edge-ratio check.
            // UV overlap detection is O(n^2) and tends to misclassify adjacent
            // triangles on curved surfaces as overlapping, causing holes. Skip it.
            for (auto& pc : passed) {
                _validTriangles.push_back(pc.t);
                _validTriIDs.push_back(pc.idx);
                _validTriangleIndicesSet.insert(pc.idx);
            }

            ProjectAndInsertClouds(target, radius * 1.2f);
            
            _cudaTexPtr = _textureLoader.generateCudaTexture(_validTriangles, _displayUVs);
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
                                    _cudaTexPtr = _textureLoader.generateCudaTexture(_validTriangles, _displayUVs);
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
                
                
                if (_dijkstraPath.size() > 0) {
                    ImGui::TextColored(ImVec4(0,1,0,1), "Path Length: %lu hops", _dijkstraPath.size());
                }

                ImGui::SameLine();
                if (ImGui::Button("Reset View")) { _viewScale = 1.0f; _viewOffset = ImVec2(0,0); }
            }
            
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
            ImGui::SameLine();
            ImGui::Checkbox("Surf Depth", &_showSurfaceDepth);
            if (_showSurfaceDepth) {
                ImGui::SameLine();
                ImGui::PushItemWidth(100.f);
                ImGui::SliderFloat("Max", &_surfDepthMax, 0.01f, 2.0f, "%.2f");
                ImGui::PopItemWidth();
            }

            {
                float prev = _surfaceBlend;
                ImGui::PushItemWidth(-1.f);
                ImGui::SliderFloat("Surface Blend##sb", &_surfaceBlend, 0.0f, 1.0f, "Surface Blend %.2f");
                ImGui::PopItemWidth();
                if (_surfaceBlend != prev) _surfaceBlendDirty = true;
            }

            {
                float sliderMax = (_computedMaxSurfDist > 1e-6f) ? _computedMaxSurfDist : 1.0f;
                _surfDistThreshold = std::min(_surfDistThreshold, sliderMax);
                float prev = _surfDistThreshold;
                ImGui::PushItemWidth(-1.f);
                ImGui::SliderFloat("Dist Cutoff##sdc", &_surfDistThreshold, 0.0f, sliderMax, "Dist Cutoff %.4f");
                ImGui::PopItemWidth();
                if (_surfDistThreshold != prev) _surfDistThresholdDirty = true;
            }

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
                    float lx = mousePos.x - (p.x + sz.x * 0.5f + _viewOffset.x);
                    float ly = mousePos.y - (p.y + sz.y * 0.5f + _viewOffset.y);
                    
                    float zoomFactor = 1.1f;
                    float oldScale = _viewScale;
                    if (wheel < 0.0f) _viewScale /= zoomFactor;
                    else              _viewScale *= zoomFactor;
                    
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
                    float sc = dim * _viewScale;
                    _selectionStart.x = 0.5f - ly / sc;
                    _selectionStart.y = 0.5f - lx / sc;
                    _selectionEnd = _selectionStart;
                }
                if (_isSelecting && ImGui::IsMouseDragging(0)) {
                    float lx = mousePos.x - (p.x + sz.x * 0.5f + _viewOffset.x);
                    float ly = mousePos.y - (p.y + sz.y * 0.5f + _viewOffset.y);
                    float sc = dim * _viewScale;
                    _selectionEnd.x = 0.5f - ly / sc;
                    _selectionEnd.y = 0.5f - lx / sc;
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
                float sc = dim * _viewScale;
                float uvX = 0.5f - ly / sc;
                float uvY = 0.5f - lx / sc;
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
                float lx = (0.5f - uv.y) * dim * _viewScale;
                float ly = (0.5f - uv.x) * dim * _viewScale;
                return ImVec2(p.x + sz.x*0.5f + lx + _viewOffset.x, p.y + sz.y*0.5f + ly + _viewOffset.y);
            };

            const float CANVAS_GUARD = sz.x + sz.y;
            auto ClampPx = [&](ImVec2 v) -> ImVec2 {
                float xMin = p.x - CANVAS_GUARD, xMax = p.x + sz.x + CANVAS_GUARD;
                float yMin = p.y - CANVAS_GUARD, yMax = p.y + sz.y + CANVAS_GUARD;
                v.x = std::max(xMin, std::min(xMax, v.x));
                v.y = std::max(yMin, std::min(yMax, v.y));
                return v;
            };

            const int TRI_BUDGET  = 5000;
            const int PT_BUDGET   = 3000;
            int triStride = std::max(1, (int)_validTriangles.size()    / TRI_BUDGET);
            int geoStride = std::max(1, (int)_projectedGeoPoints.size() / PT_BUDGET);
            int appStride = std::max(1, (int)_projectedAppPoints.size() / PT_BUDGET);

            float canvasX0 = p.x, canvasX1 = p.x + sz.x;
            float canvasY0 = p.y, canvasY1 = p.y + sz.y;

            static float maxScale3DExp = 1.0f;
            static float minOpacity = 0.0f;

            ImGui::PushItemWidth(120.0f);
            ImGui::SliderFloat("MaxScale3D##ell", &maxScale3DExp, 0.001f, 5.0f, "%.3f");
            ImGui::PopItemWidth();
            ImGui::SameLine();
            ImGui::PushItemWidth(100.0f);
            ImGui::SliderFloat("MinOpac##ell", &minOpacity, 0.0f, 1.0f, "%.2f");
            ImGui::PopItemWidth();
            if (triStride > 1)
                ImGui::TextColored(ImVec4(1,0.6f,0,1), "Tri display thinned x%d", triStride);

            {
                struct GEntry {
                    const ProjectedGaussian* pg;
                    float absDist;
                    bool  isGeo;
                    bool  isSelected;
                };
                std::vector<GEntry> selEntries, nonSelEntries;

                glm::vec3 hitN = glm::normalize(_seedFrame.axes[2]);
                glm::vec3 hitO = _seedFrame.origin;

                auto passFilter = [&](const ProjectedGaussian& pt) {
                    if (pt.uv.x < -0.15f || pt.uv.x > 1.15f || pt.uv.y < -0.15f || pt.uv.y > 1.15f) return false;
                    float op = 1.f / (1.f + std::exp(-pt.opacity));
                    if (op < minOpacity) return false;
                    glm::vec3 sExp = glm::exp(pt.scale);
                    if (std::max({sExp.x, sExp.y, sExp.z}) > maxScale3DExp) return false;
                    float surfDist = glm::length(pt.originalPos - pt.position);
                    if (surfDist > _surfDistThreshold) return false;
                    return true;
                };

                auto buildEntries = [&](const std::vector<ProjectedGaussian>& pts,
                                        const std::vector<int>& selIds, bool isGeo) {
                    for (const auto& pt : pts) {
                        if (!passFilter(pt)) continue;
                        // When surface depth mode is on, sort by true surface distance
                        // so that far (blue) points are painted first and near (red)
                        // points sit on top — consistent with the colour mapping.
                        float d = _showSurfaceDepth
                            ? glm::length(pt.originalPos - pt.position)
                            : std::abs(glm::dot(pt.position - hitO, hitN));
                        bool sel = std::find(selIds.begin(), selIds.end(), pt.originalIndex) != selIds.end();
                        GEntry e{&pt, d, isGeo, sel};
                        if (sel) selEntries.push_back(e);
                        else     nonSelEntries.push_back(e);
                    }
                };

                if (_showGeoPoints) buildEntries(_projectedGeoPoints, _selectedGeoIndices, true);
                if (_showAppPoints) buildEntries(_projectedAppPoints, _selectedAppIndices, false);

                auto byCost = [](const GEntry& a, const GEntry& b){ return a.absDist > b.absDist; };
                std::sort(nonSelEntries.begin(), nonSelEntries.end(), byCost);
                std::sort(selEntries.begin(),    selEntries.end(),    byCost);

                float nsDistMin = nonSelEntries.empty() ? 0.f : nonSelEntries.back().absDist;
                float nsDistMax = nonSelEntries.empty() ? 1.f : nonSelEntries.front().absDist;
                float nsRange   = (nsDistMax - nsDistMin) > 1e-6f ? nsDistMax - nsDistMin : 1.f;

                auto drawEllipse = [&](const GEntry& e, ImU32 fillCol, ImU32 outlineCol, bool outlineOnly=false) -> bool {
                    constexpr int N_SEGS = 32;
                    const ProjectedGaussian& pg = *e.pg;
                    ImVec2 ctr = TransformUV(pg.uv);

                    if (glm::length(pg.dU) < 1e-8f || glm::length(pg.dV) < 1e-8f) {
                        if (ctr.x>=canvasX0&&ctr.x<=canvasX1&&ctr.y>=canvasY0&&ctr.y<=canvasY1)
                            dl->AddCircleFilled(ctr, e.isSelected?5.f:2.5f, outlineCol);
                        return true;
                    }
                    glm::vec3 t1 = glm::normalize(pg.dU);
                    glm::vec3 t2r = pg.dV - glm::dot(pg.dV, t1) * t1;
                    if (glm::length(t2r) < 1e-8f) {
                        if (ctr.x>=canvasX0&&ctr.x<=canvasX1&&ctr.y>=canvasY0&&ctr.y<=canvasY1)
                            dl->AddCircleFilled(ctr, e.isSelected?5.f:2.5f, outlineCol);
                        return true;
                    }
                    glm::vec3 t2 = glm::normalize(t2r);

                    float qw=pg.rotation.x,qx=pg.rotation.y,qy=pg.rotation.z,qz=pg.rotation.w;
                    glm::mat3 R(
                        1.f-2.f*(qy*qy+qz*qz),2.f*(qx*qy+qw*qz),    2.f*(qx*qz-qw*qy),
                        2.f*(qx*qy-qw*qz),    1.f-2.f*(qx*qx+qz*qz),2.f*(qy*qz+qw*qx),
                        2.f*(qx*qz+qw*qy),    2.f*(qy*qz-qw*qx),    1.f-2.f*(qx*qx+qy*qy)
                    );
                    float s0=std::exp(pg.scale.x),s1=std::exp(pg.scale.y),s2=std::exp(pg.scale.z);
                    float a0=s0*glm::dot(R[0],t1),a1=s1*glm::dot(R[1],t1),a2=s2*glm::dot(R[2],t1);
                    float b0=s0*glm::dot(R[0],t2),b1=s1*glm::dot(R[1],t2),b2=s2*glm::dot(R[2],t2);
                    float s11=a0*a0+a1*a1+a2*a2,s12=a0*b0+a1*b1+a2*b2,s22=b0*b0+b1*b1+b2*b2;
                    float tr=s11+s22,dif=s11-s22;
                    float disc=std::sqrt(std::max(0.f,dif*dif+4.f*s12*s12));
                    float sA=std::sqrt(std::max(0.f,(tr+disc)*0.5f));
                    float sB=std::sqrt(std::max(0.f,(tr-disc)*0.5f));
                    if (sA<1e-9f) return true;
                    float theta=0.5f*std::atan2(2.f*s12,dif);
                    float cT=std::cos(theta),sT=std::sin(theta);
                    glm::vec3 axA=sA*(cT*t1+sT*t2),axB=sB*(-sT*t1+cT*t2);

                    float jA=glm::dot(pg.dU,pg.dU),jB=glm::dot(pg.dU,pg.dV),jC=glm::dot(pg.dV,pg.dV);
                    float detJ=jA*jC-jB*jB;
                    if (std::abs(detJ)<1e-14f) return true;
                    auto toUV2=[&](const glm::vec3& v)->glm::vec2{
                        float du=glm::dot(pg.dU,v),dv=glm::dot(pg.dV,v);
                        return glm::vec2((jC*du-jB*dv)/detJ,(-jB*du+jA*dv)/detJ);
                    };
                    glm::vec2 uvA=toUV2(axA),uvB=toUV2(axB);

                    float sc = dim * _viewScale;
                    float pxAx=-uvA.y*sc, pxAy=-uvA.x*sc;
                    float pxBx=-uvB.y*sc, pxBy=-uvB.x*sc;
                    float mp=pxAx*pxAx+pxAy*pxAy;
                    float mq=pxAx*pxBx+pxAy*pxBy;
                    float mr=pxBx*pxBx+pxBy*pxBy;
                    float halfTr=(mp+mr)*0.5f;
                    float halfDiff=(mp-mr)*0.5f;
                    float discSVD=std::sqrt(halfDiff*halfDiff+mq*mq);
                    float ra=std::sqrt(std::max(0.f,halfTr+discSVD));
                    float rb=std::sqrt(std::max(0.f,halfTr-discSVD));
                    if (ra<1e-6f) return true;
                    float v1x=mq, v1y=halfTr+discSVD-mp;
                    float v1len=std::sqrt(v1x*v1x+v1y*v1y);
                    float ellRot;
                    if (v1len<1e-9f) {
                        ellRot=0.f;
                    } else {
                        v1x/=v1len; v1y/=v1len;
                        float u1x=pxAx*v1x+pxBx*v1y;
                        float u1y=pxAy*v1x+pxBy*v1y;
                        ellRot=std::atan2(u1y,u1x);
                    }
                    float cosR=std::cos(ellRot),sinR=std::sin(ellRot);

                    ImVec2 pts[N_SEGS];
                    bool anyVis=false;
                    for (int k=0;k<N_SEGS;++k){
                        float phi=2.f*IM_PI*k/N_SEGS;
                        float cx=ra*std::cos(phi)*cosR - rb*std::sin(phi)*sinR;
                        float cy=ra*std::cos(phi)*sinR + rb*std::sin(phi)*cosR;
                        pts[k]=ClampPx(ImVec2(ctr.x+cx, ctr.y+cy));
                        if(pts[k].x>=canvasX0&&pts[k].x<=canvasX1&&
                           pts[k].y>=canvasY0&&pts[k].y<=canvasY1) anyVis=true;
                    }
                    if(!anyVis&&(ctr.x<canvasX0||ctr.x>canvasX1||
                                 ctr.y<canvasY0||ctr.y>canvasY1)) return true;
                    if (!outlineOnly) dl->AddConvexPolyFilled(pts,N_SEGS,fillCol);
                    dl->AddPolyline(pts,N_SEGS,outlineCol,true,1.0f);
                };

                if (_showSurfaceDepth) {
                    for (int ei=0;ei<(int)nonSelEntries.size();ei++){
                        const GEntry& e=nonSelEntries[ei];
                        // Distance from surface, shrunk by blend factor (0=original, 1=on surface).
                        float d = glm::length(e.pg->originalPos - e.pg->position) * (1.0f - _surfaceBlend);
                        float t = glm::clamp(powf(d / _surfDepthMax, 0.3f), 0.f, 1.f);
                        int r = (int)(255.f * (1.f - t));
                        int b = (int)(255.f * t);
                        ImU32 fC = IM_COL32(r, 0, b, 150);
                        ImU32 oC = IM_COL32(r, 0, b, 255);
                        drawEllipse(e, fC, oC, false);
                    }

                    for (int ei=0;ei<(int)selEntries.size();ei++){
                        const GEntry& e=selEntries[ei];
                        drawEllipse(e, IM_COL32(255,255,0,150), IM_COL32(255,255,0,255));
                    }
                } else {
                    // Draw small dots: Geo = Red, App = Blue
                    for (const auto& e : nonSelEntries) {
                        ImVec2 ctr = TransformUV(e.pg->uv);
                        if (ctr.x >= canvasX0 && ctr.x <= canvasX1 && ctr.y >= canvasY0 && ctr.y <= canvasY1) {
                            ImU32 col = e.isGeo ? IM_COL32(255, 0, 0, 255) : IM_COL32(0, 0, 255, 255);
                            dl->AddCircleFilled(ctr, 3.0f, col);
                        }
                    }
                    for (const auto& e : selEntries) {
                        ImVec2 ctr = TransformUV(e.pg->uv);
                        if (ctr.x >= canvasX0 && ctr.x <= canvasX1 && ctr.y >= canvasY0 && ctr.y <= canvasY1) {
                            dl->AddCircleFilled(ctr, 3.0f, IM_COL32(255, 255, 0, 255));
                        }
                    }
                }
            }

            for(int ti = 0; ti < (int)_validTriangles.size(); ti += triStride) {
                const auto& t = _validTriangles[ti];
                glm::vec2 uv1 = _displayUVs[t[0]];
                glm::vec2 uv2 = _displayUVs[t[1]];
                glm::vec2 uv3 = _displayUVs[t[2]];
                ImVec2 ip1 = TransformUV(uv1); ImVec2 ip2 = TransformUV(uv2); ImVec2 ip3 = TransformUV(uv3);

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

                ip1 = ClampPx(ip1); ip2 = ClampPx(ip2); ip3 = ClampPx(ip3);

                if (_drawFilled) dl->AddTriangleFilled(ip1, ip2, ip3, IM_COL32(255, 215, 0, 80));
                else dl->AddTriangle(ip1, ip2, ip3, IM_COL32(255, 215, 0, 200), 1.0f);
            }

            if (_showSuppressedPoints) {
                for (const auto& uv : _suppressedUVs) {
                    if (uv.x < -1.0f || uv.x > 2.0f || uv.y < -1.0f || uv.y > 2.0f) continue;
                    ImVec2 pos = TransformUV(uv);
                    dl->AddCircleFilled(pos, 5.0f, IM_COL32(0,180,0,200));
                    dl->AddCircleFilled(pos, 3.5f, IM_COL32(0,255,80,255));
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
    
    static bool segmentsProperlyIntersect(const glm::vec2& p1, const glm::vec2& p2,
                                          const glm::vec2& p3, const glm::vec2& p4) {
        auto cross2D = [](const glm::vec2& a, const glm::vec2& b) {
            return a.x * b.y - a.y * b.x;
        };
        glm::vec2 d = p2 - p1, e = p4 - p3;
        float denom = cross2D(d, e);
        if (std::abs(denom) < 1e-12f) return false;
        glm::vec2 r = p3 - p1;
        float t = cross2D(r, e) / denom;
        float u = cross2D(r, d) / denom;
        const float eps = 1e-5f;
        return t > eps && t < 1.f - eps && u > eps && u < 1.f - eps;
    }

    static bool pointStrictlyInTriangle(const glm::vec2& p,
                                        const glm::vec2& a, const glm::vec2& b, const glm::vec2& c) {
        float d1 = (p.x - b.x) * (a.y - b.y) - (a.x - b.x) * (p.y - b.y);
        float d2 = (p.x - c.x) * (b.y - c.y) - (b.x - c.x) * (p.y - c.y);
        float d3 = (p.x - a.x) * (c.y - a.y) - (c.x - a.x) * (p.y - a.y);
        const float eps = 1e-6f;
        bool hasNeg = (d1 < -eps) || (d2 < -eps) || (d3 < -eps);
        bool hasPos = (d1 >  eps) || (d2 >  eps) || (d3 >  eps);
        if (hasNeg && hasPos) return false;
        return (std::abs(d1) > eps) && (std::abs(d2) > eps) && (std::abs(d3) > eps);
    }

    static bool uvTrianglesOverlap(const glm::vec2 a[3], const glm::vec2 b[3]) {
        // Check if any vertex of one triangle lies inside the other
        for (int i = 0; i < 3; ++i) {
            if (pointStrictlyInTriangle(a[i], b[0], b[1], b[2])) return true;
            if (pointStrictlyInTriangle(b[i], a[0], a[1], a[2])) return true;
        }
        // Check if any edge pair properly intersects
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                if (segmentsProperlyIntersect(a[i], a[(i+1)%3], b[j], b[(j+1)%3]))
                    return true;
            }
        }
        return false;
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

        glm::vec3 nP = glm::normalize(getNormal(parentIdx));

        TangentFrame centerFrame(posP, nP);

        TangentFrame seedAligned = _seedFrame;
        seedAligned.alignZAxis(centerFrame);

        glm::vec3 centerAxisX = centerFrame.axes[0];
        glm::vec3 seedAxisX   = seedAligned.axes[0];
        float cosTheta = glm::clamp(glm::dot(centerAxisX, seedAxisX), -1.f, 1.f);
        float fTmp     = std::max(0.f, 1.f - cosTheta * cosTheta);
        float sinTheta = std::sqrt(fTmp);
        glm::vec3 crossVec = glm::cross(centerAxisX, seedAxisX);
        if (glm::dot(crossVec, nP) < 0.f) sinTheta = -sinTheta;

        glm::mat2 matR(
            glm::vec2( cosTheta, -sinTheta),
            glm::vec2( sinTheta,  cosTheta)
        );

        glm::vec3 posC_proj = posC - nP * glm::dot(posC - posP, nP);
        glm::vec3 localVec  = centerFrame.toLocal(posC_proj - posP);

        curr.uv = parent.uv + matR * glm::vec2(localVec.x, localVec.y);
    }

    void ProjectAndInsertClouds(const glm::vec3& center, float radius) {
        std::map<int, std::vector<int>> triToExtraNodes;

        auto projectOntoTri = [&](const glm::vec3& pt, int triIdx, size_t ptIdx,
                                   const std::vector<GaussianProps>& props,
                                   ProjectedGaussian& pg) -> bool {
            const auto& t = _mesh->triangles()[triIdx];
            if (!_displayUVs.count(t[0]) || !_displayUVs.count(t[1]) || !_displayUVs.count(t[2]))
                return false;

            glm::vec3 v0 = toGlm(_mesh->vertices()[t[0]]);
            glm::vec3 v1 = toGlm(_mesh->vertices()[t[1]]);
            glm::vec3 v2 = toGlm(_mesh->vertices()[t[2]]);
            glm::vec3 e1 = v1 - v0, e2 = v2 - v0;
            glm::vec3 rawN = glm::cross(e1, e2);
            float lenN = glm::length(rawN);
            if (lenN < 1e-9f) return false;
            glm::vec3 n = rawN / lenN;

            float pDist   = glm::dot(pt - v0, n);
            glm::vec3 pProj = pt - n * pDist;
            glm::vec3 pVec  = pProj - v0;

            float d00 = glm::dot(e1, e1), d01 = glm::dot(e1, e2), d11 = glm::dot(e2, e2);
            float d20 = glm::dot(pVec, e1), d21 = glm::dot(pVec, e2);
            float denom = d00 * d11 - d01 * d01;

            glm::vec2 uv0 = _displayUVs.at(t[0]);
            glm::vec2 uv1 = _displayUVs.at(t[1]);
            glm::vec2 uv2 = _displayUVs.at(t[2]);

            glm::vec2 baryUV;
            if (std::abs(denom) > 1e-9f) {
                float bv = (d11 * d20 - d01 * d21) / denom;
                float bw = (d00 * d21 - d01 * d20) / denom;
                float bu = 1.0f - bv - bw;
                baryUV = bu * uv0 + bv * uv1 + bw * uv2;
            } else {
                baryUV = (uv0 + uv1 + uv2) / 3.0f;
            }

            pg.originalIndex = (int)ptIdx;
            pg.uv            = baryUV;
            pg.position      = pProj;   // surface-projected position
            pg.originalPos   = pt;      // original Gaussian position (for surf depth distance)

            float du1 = uv1.x-uv0.x, dv1 = uv1.y-uv0.y, du2 = uv2.x-uv0.x, dv2 = uv2.y-uv0.y;
            float det = du1*dv2 - du2*dv1;
            if (std::abs(det) > 1e-9f) {
                pg.dU = ( dv2*(v1-v0) - dv1*(v2-v0)) / det;
                pg.dV = (-du2*(v1-v0) + du1*(v2-v0)) / det;
            }
            {
                glm::vec2 uvMin = glm::min(uv0, glm::min(uv1, uv2));
                glm::vec2 uvMax = glm::max(uv0, glm::max(uv1, uv2));
                pg.uvMaxDelta = glm::length(uvMax - uvMin) * 2.0f; // Relaxed clamp to prevent cropping
            }
            if (ptIdx < props.size()) {
                pg.opacity  = props[ptIdx].opacity;
                pg.scale    = glm::vec3(props[ptIdx].scale[0], props[ptIdx].scale[1], props[ptIdx].scale[2]);
                pg.rotation = glm::vec4(props[ptIdx].rot[0], props[ptIdx].rot[1], props[ptIdx].rot[2], props[ptIdx].rot[3]);
            }
            return true;
        };

        auto process = [&](const std::vector<float>& cloud, const std::vector<int>& fids,
                          const std::vector<GaussianProps>& props,
                          std::vector<ProjectedGaussian>& outPts, const char* cloudName) {

            outPts.clear();
            if (cloud.empty()) return;

            size_t numPoints = cloud.size() / 3;
            float r2 = radius * radius;
            size_t meshSize = _mesh->vertices().size();

            bool isGeo    = (std::strcmp(cloudName, "GEO") == 0);
            bool is1to1   = isGeo && (numPoints == meshSize);
            bool hasFids  = !fids.empty() && ((int)fids.size() == (int)numPoints);

            int outOfRadius = 0;
            int projected   = 0;
            int fidHits     = 0;

            for (size_t i = 0; i < numPoints; ++i) {
                glm::vec3 pt(cloud[i*3], cloud[i*3+1], cloud[i*3+2]);

                if (is1to1) {
                    int vIdx = (int)i;
                    if (_displayUVs.count(vIdx)) {
                        glm::vec3 sumDU(0.f), sumDV(0.f);
                        float sumUVMaxDelta = 0.f;
                        int validCount = 0;
                        int firstTIdx = -1;
                        for (int tIdx : _vertexToTriangles[vIdx]) {
                            if (!_validTriangleIndicesSet.count(tIdx)) continue;
                            if (firstTIdx == -1) firstTIdx = tIdx;
                            const auto& t = _mesh->triangles()[tIdx];
                            if (!_displayUVs.count(t[0]) || !_displayUVs.count(t[1]) || !_displayUVs.count(t[2])) continue;
                            glm::vec3 v0 = toGlm(_mesh->vertices()[t[0]]), v1 = toGlm(_mesh->vertices()[t[1]]), v2 = toGlm(_mesh->vertices()[t[2]]);
                            glm::vec2 uv0 = _displayUVs.at(t[0]), uv1 = _displayUVs.at(t[1]), uv2 = _displayUVs.at(t[2]);
                            float du1 = uv1.x-uv0.x, dv1 = uv1.y-uv0.y, du2 = uv2.x-uv0.x, dv2 = uv2.y-uv0.y;
                            float det = du1*dv2 - du2*dv1;
                            if (std::abs(det) < 1e-12f) continue;
                            sumDU += ( dv2*(v1-v0) - dv1*(v2-v0)) / det;
                            sumDV += (-du2*(v1-v0) + du1*(v2-v0)) / det;
                            glm::vec2 uvMin = glm::min(uv0, glm::min(uv1, uv2));
                            glm::vec2 uvMax = glm::max(uv0, glm::max(uv1, uv2));
                            sumUVMaxDelta += glm::length(uvMax - uvMin) * 0.5f;
                            ++validCount;
                        }

                        if (firstTIdx != -1) {
                            ProjectedGaussian pg;
                            pg.originalIndex = vIdx;
                            pg.uv = _displayUVs.at(vIdx);
                            pg.position = pt;
                            pg.originalPos = pt;
                            if (validCount > 0) {
                                pg.dU = sumDU / float(validCount);
                                pg.dV = sumDV / float(validCount);
                                pg.uvMaxDelta = sumUVMaxDelta / float(validCount);
                            }
                            if (vIdx < (int)props.size()) {
                                pg.opacity  = props[vIdx].opacity;
                                pg.scale    = glm::vec3(props[vIdx].scale[0], props[vIdx].scale[1], props[vIdx].scale[2]);
                                pg.rotation = glm::vec4(props[vIdx].rot[0], props[vIdx].rot[1], props[vIdx].rot[2], props[vIdx].rot[3]);
                            }
                            outPts.push_back(pg);
                            projected++;
                            continue;
                        }
                    }
                }

                if (hasFids && !isGeo) {
                    int faceID = fids[i];
                    if (faceID >= 0 && faceID < (int)_mesh->triangles().size() &&
                        _validTriangleIndicesSet.count(faceID)) {
                        ProjectedGaussian pg;
                        if (projectOntoTri(pt, faceID, i, props, pg)) {
                            outPts.push_back(pg);
                            projected++;
                            fidHits++;
                            continue;
                        }
                    }
                    continue;
                }

                float distSq = glm::dot(pt - center, pt - center);
                if (distSq > r2) {
                    outOfRadius++;
                    continue;
                }

                float bestDist = 1e9f;
                glm::vec2 bestUV(0, 0);
                int bestTriGlobalID = -1;
                bool found = false;
                glm::vec3 bestProjPos(0, 0, 0);

                for (int globalTIdx : _validTriIDs) {
                    const auto& t = _mesh->triangles()[globalTIdx];
                    glm::vec3 v0 = toGlm(_mesh->vertices()[t[0]]);
                    glm::vec3 v1 = toGlm(_mesh->vertices()[t[1]]);
                    glm::vec3 v2 = toGlm(_mesh->vertices()[t[2]]);
                    glm::vec3 n = glm::normalize(glm::cross(v1 - v0, v2 - v0));

                    float pDist = glm::dot(pt - v0, n);
                    if (std::abs(pDist) > 0.05f * radius) continue;

                    glm::vec3 pProj = pt - n * pDist;
                    glm::vec3 v0v1 = v1 - v0, v0v2 = v2 - v0, pVec = pProj - v0;
                    float d00 = glm::dot(v0v1, v0v1), d01 = glm::dot(v0v1, v0v2), d11 = glm::dot(v0v2, v0v2);
                    float d20 = glm::dot(pVec, v0v1), d21 = glm::dot(pVec, v0v2);
                    float denom = d00 * d11 - d01 * d01;
                    if (std::abs(denom) < 1e-9f) continue;

                    float bv = (d11 * d20 - d01 * d21) / denom;
                    float bw = (d00 * d21 - d01 * d20) / denom;
                    float bu = 1.0f - bv - bw;

                    if (bu >= -0.001f && bv >= -0.001f && bw >= -0.001f) {
                        float d = std::abs(pDist);
                        if (d < bestDist) {
                            bestDist = d;
                            bestUV = bu * _displayUVs.at(t[0]) + bv * _displayUVs.at(t[1]) + bw * _displayUVs.at(t[2]);
                            bestProjPos = pProj;
                            bestTriGlobalID = globalTIdx;
                            found = true;
                        }
                    }
                }

                if (found) {
                    ProjectedGaussian pg;
                    pg.originalIndex = (int)i;
                    pg.uv            = bestUV;
                    pg.position      = bestProjPos; // surface-projected position
                    pg.originalPos   = pt;          // original Gaussian position (for surf depth distance)

                    const auto& t = _mesh->triangles()[bestTriGlobalID];
                    glm::vec3 v0 = toGlm(_mesh->vertices()[t[0]]), v1 = toGlm(_mesh->vertices()[t[1]]), v2 = toGlm(_mesh->vertices()[t[2]]);
                    glm::vec2 uv0 = _displayUVs.at(t[0]), uv1 = _displayUVs.at(t[1]), uv2 = _displayUVs.at(t[2]);
                    float du1 = uv1.x - uv0.x, dv1 = uv1.y - uv0.y, du2 = uv2.x - uv0.x, dv2 = uv2.y - uv0.y;
                    float det = du1 * dv2 - du2 * dv1;
                    if (std::abs(det) > 1e-9f) {
                        pg.dU = ( dv2 * (v1-v0) - dv1 * (v2-v0)) / det;
                        pg.dV = (-du2 * (v1-v0) + du1 * (v2-v0)) / det;
                    }
                    {
                        glm::vec2 uvMin = glm::min(uv0, glm::min(uv1, uv2));
                        glm::vec2 uvMax = glm::max(uv0, glm::max(uv1, uv2));
                        pg.uvMaxDelta = glm::length(uvMax - uvMin) * 2.0f; // Relaxed clamp to prevent cropping
                    }
                    if (i < props.size()) {
                        pg.opacity   = props[i].opacity;
                        pg.scale     = glm::vec3(props[i].scale[0], props[i].scale[1], props[i].scale[2]);
                        pg.rotation  = glm::vec4(props[i].rot[0], props[i].rot[1], props[i].rot[2], props[i].rot[3]);
                    }
                    outPts.push_back(pg);
                    projected++;
                }
            }

            if (std::strcmp(cloudName, "APP") == 0) {
                _projectionStats.appTotal     = (int)numPoints;
                _projectionStats.appProjected = projected;
                std::cout << "[ProjectAndInsertClouds] APP: total=" << numPoints
                          << " fid_hits=" << fidHits
                          << " projected=" << projected << "\n";
            } else {
                _projectionStats.geoTotal     = (int)numPoints;
                _projectionStats.geoProjected = projected;
            }
        };

        process(_geoCloudData, {}, _geoGaussianProps, _projectedGeoPoints, "GEO");
        process(_appCloudData, _appCloudFids, _appGaussianProps, _projectedAppPoints, "APP");
        _liveDirty = true;

        // Compute max surface distance for the dist-cutoff slider range
        float maxD = 0.f;
        for (const auto& pg : _projectedGeoPoints)
            maxD = std::max(maxD, glm::length(pg.originalPos - pg.position));
        for (const auto& pg : _projectedAppPoints)
            maxD = std::max(maxD, glm::length(pg.originalPos - pg.position));
        _computedMaxSurfDist = (maxD > 1e-6f) ? maxD * 1.1f : 1.0f;
        _surfDistThreshold = _computedMaxSurfDist;  // Default: show all
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
    
    bool   _showAppPoints    = true;
    bool   _showGeoPoints    = true;
    bool   _showSurfaceDepth = false;
    float  _surfDepthMax     = 0.6f;

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
    sibr::Texture2DRGBA::Ptr _cudaTexPtr = nullptr;
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

    std::vector<glm::vec2> _suppressedUVs;
    bool _showSuppressedPoints = true;

    float _surfaceBlend      = 1.0f;
    bool  _surfaceBlendDirty = false;

    float _surfDistThreshold      = 1e9f;
    float _computedMaxSurfDist    = 1.0f;
    bool  _surfDistThresholdDirty = false;
};

#endif