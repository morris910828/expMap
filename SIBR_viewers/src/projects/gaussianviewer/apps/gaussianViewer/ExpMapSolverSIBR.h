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
// Data Structures
// ==========================================
struct TangentFrame {
    glm::vec3 origin = {0, 0, 0};
    glm::mat3 axes = glm::mat3(1); // Columns: X, Y, Z (Z=Normal)

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
            if (cosAngle < 0) {
                axes = glm::mat3(axes[0], -axes[1], -axes[2]);
            }
            return;
        }

        axis = glm::normalize(axis);
        float angle = std::acos(glm::clamp(cosAngle, -1.0f, 1.0f));
        glm::mat3 rot = glm::mat3(glm::rotate(glm::mat4(1.0f), angle, axis));
        axes = rot * axes;
    }
};

struct ExpMapVertex {
    enum State { INACTIVE, ACTIVE, FROZEN };

    int id;
    State state = INACTIVE;
    float distance = std::numeric_limits<float>::max();
    glm::vec2 surfaceVector = {0, 0};
    TangentFrame frame;
    int nearestId = -1;
};


// ==========================================
// ExpMap Solver (Ported from ExpMapDemo & original file)
// ==========================================
class ExpMapSolverSIBR {
public:

    void Init(const sibr::Mesh* mesh) {
        _mesh = mesh;
        buildAdjacency();
    }

    const std::vector<sibr::Vector3u>& GetActiveTriangles() const { return _validTriangles; }
    void RegisterAppPointCloud(const std::vector<float>& pointData) { _appCloudData = pointData; }
    void RegisterGeoPointCloud(const std::vector<float>& pointData) { _geoCloudData = pointData; }

    void Compute(const sibr::Vector3f& hitPos, const sibr::Vector3f& hitNormal, float radius) {
        if (!_mesh || _mesh->vertices().empty()) return;

        // --- Reset State ---
        _vertexData.clear();
        _displayUVs.clear();
        _validTriangles.clear();
        _projectedAppPoints.clear();
        _projectedGeoPoints.clear();

        // --- 1. Find Seed Vertex ---
        glm::vec3 target = toGlm(hitPos);
        int seedIdx = -1;
        float minD = std::numeric_limits<float>::max();
        const auto& verts = _mesh->vertices();
        for (size_t i = 0; i < verts.size(); ++i) {
            float d = glm::distance(target, toGlm(verts[i]));
            if (d < minD) { minD = d; seedIdx = (int)i; }
        }
        if (seedIdx == -1) return;

        // --- 2. Find all vertices within the radius (the patch) ---
        std::set<int> patch_vertices;
        std::map<int, float> distances;
        auto cmp_dist = [&](int a, int b) { return distances[a] > distances[b]; };
        std::priority_queue<int, std::vector<int>, decltype(cmp_dist)> pq_dist(cmp_dist);

        distances[seedIdx] = 0.0f;
        pq_dist.push(seedIdx);
        
        float max_dist_in_patch = 0.f;

        while (!pq_dist.empty()) {
            int u = pq_dist.top();
            pq_dist.pop();

            if (distances[u] > radius) continue;
            patch_vertices.insert(u);
            max_dist_in_patch = std::max(max_dist_in_patch, distances[u]);

            for (int v_idx : _adj[u]) {
                float edge_dist = glm::distance(toGlm(verts[u]), toGlm(verts[v_idx]));
                if (distances.find(v_idx) == distances.end() || distances[u] + edge_dist < distances[v_idx]) {
                    distances[v_idx] = distances[u] + edge_dist;
                    pq_dist.push(v_idx);
                }
            }
        }
        if (patch_vertices.empty()) return;

        // --- 3. Initialize Per-Vertex Data for the ExpMap algorithm ---
        _vertexData.resize(verts.size());
        for (int idx : patch_vertices) {
            _vertexData[idx].id = idx;
            _vertexData[idx].state = ExpMapVertex::INACTIVE;
            _vertexData[idx].distance = std::numeric_limits<float>::max();
            _vertexData[idx].nearestId = -1;
        }

        // --- 4. Set up Seed Vertex ---
        auto& seed = _vertexData[seedIdx];
        seed.state = ExpMapVertex::FROZEN;
        seed.distance = 0.0f;
        seed.surfaceVector = glm::vec2(0, 0);

        glm::vec3 seedPos = toGlm(verts[seedIdx]);
        glm::vec3 seedN = (_mesh->normals().size() > seedIdx) ? toGlm(_mesh->normals()[seedIdx]) : toGlm(hitNormal);
        seed.frame = TangentFrame(seedPos, seedN);

        // --- 5. Initialize Priority Queue ---
        auto cmp = [&](int a, int b) { return _vertexData[a].distance > _vertexData[b].distance; };
        std::priority_queue<int, std::vector<int>, decltype(cmp)> pq(cmp);

        for (int neighborIdx : _adj[seedIdx]) {
            if (patch_vertices.count(neighborIdx)) {
                auto& nv = _vertexData[neighborIdx];
                nv.distance = glm::distance(toGlm(verts[neighborIdx]), seedPos);
                nv.nearestId = seedIdx;
                nv.state = ExpMapVertex::ACTIVE;
                pq.push(neighborIdx);
            }
        }
        
        // --- 6. Dijkstra Propagation (ExpMap Solver) ---
        while (!pq.empty()) {
            int curIdx = pq.top();
            pq.pop();

            auto& cur = _vertexData[curIdx];
            if (cur.state == ExpMapVertex::FROZEN) continue;
            cur.state = ExpMapVertex::FROZEN;

            propagateFrame(cur, seedIdx);

            glm::vec3 curPos = toGlm(verts[curIdx]);
            for (int neighborIdx : _adj[curIdx]) {
                if (!patch_vertices.count(neighborIdx)) continue;

                auto& nv = _vertexData[neighborIdx];
                if (nv.state == ExpMapVertex::FROZEN) continue;

                float edgeDist = glm::distance(toGlm(verts[neighborIdx]), curPos);
                float newDist = cur.distance + edgeDist;

                if (newDist < nv.distance) {
                    nv.distance = newDist;
                    nv.nearestId = curIdx;
                    nv.state = ExpMapVertex::ACTIVE;
                    pq.push(neighborIdx);
                }
            }
        }

        // --- 7. Normalize and Output UVs ---
        if (max_dist_in_patch < 1e-6f) max_dist_in_patch = 1.0f;
        
        float uvScale = 1.0f / (max_dist_in_patch * std::sqrt(2.0f));

        for (int idx : patch_vertices) {
            const auto& v = _vertexData[idx];
            if (v.state == ExpMapVertex::FROZEN) {
                _displayUVs[v.id] = v.surfaceVector * uvScale + glm::vec2(0.5f, 0.5f);
            }
        }

        const auto& tris = _mesh->triangles();
        for (const auto& t : tris) {
            if (_displayUVs.count(t[0]) && _displayUVs.count(t[1]) && _displayUVs.count(t[2])) {
                _validTriangles.push_back(t);
            }
        }

        // --- 8. Project Point Clouds ---
        ProjectAllPointClouds(target, radius * 1.2f);
    }

    void RenderUI() {
        ImGui::SetNextWindowSize(ImVec2(400, 450), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("ExpMap UV Result")) {
            if (_displayUVs.empty()) {
                ImGui::TextColored(ImVec4(1, 1, 0, 1), "Right-click on the mesh to compute UV.");
                ImGui::End();
                return;
            }

            ImGui::Text("Tris: %lu | Pts: %lu", _validTriangles.size(), _projectedAppPoints.size());
            ImGui::SameLine();
            if (ImGui::Button("Reset View")) { _viewScale = 1.0f; _viewOffset = ImVec2(0, 0); }
            
            ImGui::Checkbox("Fill", &_drawFilled);
            ImGui::SameLine();
            ImGui::Checkbox("App Pts", &_showAppPoints);
            ImGui::SameLine();
            ImGui::Checkbox("Geo Pts", &_showGeoPoints);

            ImVec2 p = ImGui::GetCursorScreenPos();
            ImVec2 sz = ImGui::GetContentRegionAvail();
            if (sz.x < 50) sz.x = 50;
            if (sz.y < 50) sz.y = 50;
            float dim = std::min(sz.x, sz.y);

            if (ImGui::IsWindowHovered()) {
                float wheel = ImGui::GetIO().MouseWheel;
                if (wheel != 0.0f) {
                    float zoomFactor = 1.1f;
                    _viewScale *= (wheel > 0.0f) ? zoomFactor : (1.0f / zoomFactor);
                }
                if (ImGui::IsMouseDragging(2) || ImGui::IsMouseDragging(1)) { // Using integer literals for mouse buttons
                    ImVec2 delta = ImGui::GetIO().MouseDelta;
                    _viewOffset.x += delta.x;
                    _viewOffset.y += delta.y;
                }
            }

            ImDrawList* dl = ImGui::GetWindowDrawList();
            dl->PushClipRect(p, ImVec2(p.x + sz.x, p.y + sz.y), true);
            dl->AddRectFilled(p, ImVec2(p.x + sz.x, p.y + sz.y), IM_COL32(40, 40, 40, 255));

            auto ToScreen = [&](glm::vec2 uv) {
                float x = (uv.x - 0.5f) * dim * _viewScale;
                float y = (1.0f - uv.y - 0.5f) * dim * _viewScale;
                return ImVec2(p.x + sz.x * 0.5f + x + _viewOffset.x, p.y + sz.y * 0.5f + y + _viewOffset.y);
            };

            // Draw Mesh
            for (const auto& t : _validTriangles) {
                ImVec2 p1 = ToScreen(_displayUVs[t[0]]);
                ImVec2 p2 = ToScreen(_displayUVs[t[1]]);
                ImVec2 p3 = ToScreen(_displayUVs[t[2]]);
                if (_drawFilled) {
                    dl->AddTriangleFilled(p1, p2, p3, IM_COL32(255, 215, 0, 80));
                }
                dl->AddTriangle(p1, p2, p3, IM_COL32(255, 215, 0, 200));
            }

            // Draw Points
            if (_showAppPoints) for (auto& pt : _projectedAppPoints) dl->AddCircleFilled(ToScreen(pt), 3.0f, IM_COL32(0, 255, 255, 255));
            if (_showGeoPoints) for (auto& pt : _projectedGeoPoints) dl->AddCircleFilled(ToScreen(pt), 3.0f, IM_COL32(255, 50, 50, 255));

            dl->PopClipRect();
        }
        ImGui::End();
    }

private:
    void buildAdjacency() {
        if (!_mesh) return;
        _adj.assign(_mesh->vertices().size(), {});
        for (const auto& t : _mesh->triangles()) {
            _adj[t.x()].push_back(t.y()); _adj[t.x()].push_back(t.z());
            _adj[t.y()].push_back(t.x()); _adj[t.y()].push_back(t.z());
            _adj[t.z()].push_back(t.x()); _adj[t.z()].push_back(t.y());
        }
        for (auto& neighbors : _adj) {
            std::sort(neighbors.begin(), neighbors.end());
            neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
        }
    }

    void propagateFrame(ExpMapVertex& current, int seedIdx) {
        if (current.nearestId == -1) return;

        auto& nearest = _vertexData[current.nearestId];
        const auto& all_verts = _mesh->vertices();

        glm::vec3 curPos = toGlm(all_verts[current.id]);
        glm::vec3 curNormal = toGlm(_mesh->normals()[current.id]);
        current.frame = TangentFrame(curPos, curNormal);

        TangentFrame& seedFrame = _vertexData[seedIdx].frame;
        TangentFrame alignedSeedFrame = seedFrame;
        alignedSeedFrame.alignZAxis(nearest.frame);

        glm::vec3 nearestX = nearest.frame.axes[0];
        glm::vec3 seedX = alignedSeedFrame.axes[0];

        float cosTheta = glm::clamp(glm::dot(nearestX, seedX), -1.0f, 1.0f);
        float sinTheta = std::sqrt(std::max(0.0f, 1.0f - cosTheta * cosTheta));
        glm::vec3 cross = glm::cross(nearestX, seedX);
        if (glm::dot(cross, nearest.frame.axes[2]) < 0) {
            sinTheta = -sinTheta;
        }
        
        glm::mat2 rot(cosTheta, -sinTheta, sinTheta, cosTheta);

        glm::vec3 nearestPos = toGlm(all_verts[nearest.id]);
        glm::vec3 nearestNormal = nearest.frame.axes[2];
        glm::vec3 projected = curPos - glm::dot(curPos - nearestPos, nearestNormal) * nearestNormal;
        
        glm::vec3 origVec = curPos - nearestPos;
        glm::vec3 projVec = projected - nearestPos;
        float projLen = glm::length(projVec);
        float scale = (projLen > 1e-8f) ? glm::length(origVec) / projLen : 1.0f;
        glm::vec3 planePoint = nearestPos + scale * projVec;

        glm::vec3 localVec = nearest.frame.toLocal(planePoint - nearestPos);
        glm::vec2 surfaceVec(-localVec.x, -localVec.y);

        current.surfaceVector = nearest.surfaceVector + rot * surfaceVec;

        float vecLen = glm::length(current.surfaceVector);
        float distSq = current.distance * current.distance;
        if (distSq > 1e-12f) {
            float distError = std::abs(vecLen * vecLen / distSq - 1.0f);
            if (distError > 0.75f && vecLen > 1e-8f) { // Using 0.75f threshold from ExpMapDemo
                current.surfaceVector = glm::normalize(current.surfaceVector) * current.distance;
            }
        }
    }

    void ProjectAllPointClouds(const glm::vec3& center, float radius) {
        ProcessCloud(_appCloudData, _projectedAppPoints, center, radius);
        ProcessCloud(_geoCloudData, _projectedGeoPoints, center, radius);
    }

    void ProcessCloud(const std::vector<float>& data, std::vector<glm::vec2>& outPts, const glm::vec3& center, float r) {
        outPts.clear();
        if (data.empty() || _displayUVs.empty()) return;

        const auto& all_verts = _mesh->vertices();
        size_t numPoints = data.size() / 3;
        float r2 = r * r;

        for (size_t i = 0; i < numPoints; ++i) {
            glm::vec3 pt(data[i * 3], data[i * 3 + 1], data[i * 3 + 2]);
            glm::vec3 diff = pt - center;
            if (glm::dot(diff, diff) > r2) continue;

            float minDist = std::numeric_limits<float>::max();
            glm::vec2 bestUV(0, 0);
            bool found = false;

            for (const auto& t : _validTriangles) {
                glm::vec3 v0 = toGlm(all_verts[t[0]]);
                glm::vec3 v1 = toGlm(all_verts[t[1]]);
                glm::vec3 v2 = toGlm(all_verts[t[2]]);

                glm::vec3 n = glm::cross(v1 - v0, v2 - v0);
                if (glm::length(n) < 1e-9) continue;
                n = glm::normalize(n);

                if (std::abs(glm::dot(pt - v0, n)) > 0.1f * r) continue; // FIXED: Stricter plane distance check

                glm::vec3 pProj = pt - n * glm::dot(pt - v0, n);

                glm::vec3 v0v1 = v1 - v0; glm::vec3 v0v2 = v2 - v0; glm::vec3 pVec = pProj - v0;
                float d00 = glm::dot(v0v1, v0v1); float d01 = glm::dot(v0v1, v0v2);
                float d11 = glm::dot(v0v2, v0v2); float d20 = glm::dot(pVec, v0v1);
                float d21 = glm::dot(pVec, v0v2);
                float denom = d00 * d11 - d01 * d01;

                if (std::abs(denom) > 1e-9f) {
                    float v = (d11 * d20 - d01 * d21) / denom;
                    float w = (d00 * d21 - d01 * d20) / denom;
                    float u = 1.0f - v - w;

                    // FIXED: Tighter barycentric coordinate check
                    if (u >= -0.01f && v >= -0.01f && w >= -0.01f && u <= 1.01f && v <= 1.01f && w <= 1.01f) {
                        float dist = glm::distance(pt, pProj);
                        if (dist < minDist) {
                            minDist = dist;
                            bestUV = u * _displayUVs.at(t[0]) + v * _displayUVs.at(t[1]) + w * _displayUVs.at(t[2]);
                            found = true;
                        }
                    }
                }
            }
            if (found) outPts.push_back(bestUV);
        }
    }

    // Member Variables
    const sibr::Mesh* _mesh = nullptr;
    std::vector<std::vector<int>> _adj;
    std::vector<ExpMapVertex> _vertexData;
    std::map<int, glm::vec2> _displayUVs;
    std::vector<sibr::Vector3u> _validTriangles;

    // UI
    float _viewScale = 1.0f;
    ImVec2 _viewOffset = {0,0};
    bool _drawFilled = false;
    bool _showAppPoints = true;
    bool _showGeoPoints = true;

    // Point Cloud Data
    std::vector<float> _appCloudData;
    std::vector<glm::vec2> _projectedAppPoints;
    std::vector<float> _geoCloudData;
    std::vector<glm::vec2> _projectedGeoPoints;
};

#endif