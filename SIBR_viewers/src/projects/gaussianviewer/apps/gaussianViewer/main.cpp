#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <filesystem>
#include <regex>
#include <array>
#include <cmath>

#include <core/graphics/Window.hpp>
#include <core/view/MultiViewManager.hpp>
#include <core/system/String.hpp>
#include "projects/gaussianviewer/renderer/GaussianView.hpp"
#include <core/renderer/DepthRenderer.hpp>
#include <core/raycaster/Raycaster.hpp>
#include <core/view/SceneDebugView.hpp>
#include <imgui/imgui_internal.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_inverse.hpp>

#include "ExpMapSolverSIBR.h"

namespace std_fs = std::filesystem;
using namespace sibr;

class MeshGaussianView : public sibr::ViewBase {
public:
    using Ptr = std::shared_ptr<MeshGaussianView>;

    MeshGaussianView(
        GaussianView::Ptr                   gaussianView,
        const sibr::Mesh*                   mesh,
        const std::string&                  geoPath,
        const std::string&                  appPath,
        sibr::InteractiveCameraHandler::Ptr camHandler
    )   : ViewBase(gaussianView->getResolution().x(), gaussianView->getResolution().y()),
          _gaussianView(gaussianView),
          _mesh(mesh),
          _camHandler(camHandler),
          _viewport(0, 0,
                    (float)gaussianView->getResolution().x(),
                    (float)gaussianView->getResolution().y())
    {
        if (std_fs::exists(geoPath)) {
            _geoRenderer.load(geoPath);
            _geoRenderer.loadGaussianProps(geoPath);
        }
        if (std_fs::exists(appPath)) {
            _appRenderer.load(appPath);
            _appRenderer.loadFids(appPath);
            _appRenderer.loadGaussianProps(appPath);
        }
        if (_mesh) {
            _expMapSolver.Init(_mesh);
            if (_geoRenderer.isLoaded())
                _expMapSolver.RegisterGeoPointCloud(_geoRenderer.getRawData(),
                                                    _geoRenderer.getGaussianProps());
            if (_appRenderer.isLoaded())
                _expMapSolver.RegisterAppPointCloud(_appRenderer.getRawData(),
                                                    _appRenderer.getFids(),
                                                    _appRenderer.getGaussianProps());
            _wireframeRenderer.uploadMesh(_mesh);
        }
        _meshColor = sibr::Vector3f(0.0f, 0.6f, 0.0f);
        _geoColor  = sibr::Vector3f(1.0f, 0.2f, 0.0f);
        _appColor  = sibr::Vector3f(0.0f, 0.4f, 1.0f);
    }

    ~MeshGaussianView() {
        if (_liveSSBO) glDeleteBuffers(1, &_liveSSBO);
    }

    void onRenderIBR(sibr::IRenderTarget& dst, const sibr::Camera& eye) override {
        if (!_gaussianView) return;

        // Single-pass textured Gaussian path:
        // CUDA now renders final textured output directly.
        _gaussianView->onRenderIBR(dst, eye);

        _viewport = Viewport(0, 0, (float)dst.w(), (float)dst.h());
        dst.bind();
        glViewport(0, 0, dst.w(), dst.h());

        const glm::mat4 mvp = glm::make_mat4(eye.viewproj().data());

        // Keep only debug / visualization overlays.
        _wireframeRenderer.render(_mesh, _expMapSolver.GetActiveTriIndices(), mvp, _showMesh);
        _pointCloudRenderer.render(
            mvp, _pointSize,
            _geoRenderer, _geoColor, _showGeo,
            _appRenderer, _appColor, _showApp);

        const auto& path = _expMapSolver.GetDijkstraPath();
        if (path.size() >= 2 && _mesh) {
            const auto& verts = _mesh->vertices();
            glUseProgram(0);
            glDisable(GL_DEPTH_TEST);
            glLineWidth(3.0f);
            glBegin(GL_LINE_STRIP);
            glColor3f(0.0f, 1.0f, 0.0f);
            for (int vid : path) {
                if (vid >= 0 && vid < (int)verts.size()) {
                    const auto& v = verts[vid];
                    glVertex3f(v.x(), v.y(), v.z());
                }
            }
            glEnd();
            glEnable(GL_DEPTH_TEST);
        }

        glUseProgram(0);
        dst.unbind();
    }

    void onUpdate(sibr::Input& input) override {
        if (!_gaussianView) return;
        _gaussianView->onUpdate(input);
        if (input.mouseButton().isReleased(sibr::Mouse::Right))
            performRaycast(input);
    }

    void onGUI() override {
        if (!_gaussianView) return;
        _gaussianView->onGUI();

        ImGui::Begin("Mesh & Controls");
        ImGui::Checkbox("Show Mesh",         &_showMesh);
        ImGui::Checkbox("Show UV Wireframe", &_wireframeRenderer._showYellowWireframe);
        ImGui::Checkbox("Show Geo Points",   &_showGeo);
        ImGui::Checkbox("Show App Points",   &_showApp);
        ImGui::SliderFloat("Point Size",     &_pointSize,    1.0f, 3.0f);
        ImGui::SliderFloat("ExpMap Radius",  &_expMapRadius, 0.05f, 2.0f);

        if (ImGui::Button("Clear ExpMap")) {
            _gaussianView->restoreOpacities();   // restore suppressed Gaussian opacities
            _texGaussians.clear();
            _liveSSBODirty = true;
            _expMapSolver.ClearRaycastState();
            _texPtr = nullptr;

            std::vector<sibr::Vector2f> empty_uvs(_gaussianView->getCount(), sibr::Vector2f(-1.f, -1.f));
            std::vector<sibr::Vector3f> empty_dUs(_gaussianView->getCount(), sibr::Vector3f(0.f, 0.f, 0.f));
            std::vector<sibr::Vector3f> empty_dVs(_gaussianView->getCount(), sibr::Vector3f(0.f, 0.f, 0.f));
            _gaussianView->setUVsAndTexture(empty_uvs, empty_dUs, empty_dVs, nullptr);
        }

        ImGui::Separator();
        const auto& slots = _expMapSolver.GetAllSlots();
        if (!slots.empty())
            ImGui::TextColored(ImVec4(0.3f,1.f,0.5f,1.f),
                "Saved Slots: %lu (manage in ExpMap UV window)", slots.size());
        ImGui::Separator();
        const auto& activeTris = _expMapSolver.GetActiveTris();
        if (!activeTris.empty()) {
            ImGui::TextColored(ImVec4(1.f,1.f,0.f,1.f),
                "UV Region: %lu triangles", activeTris.size());
            if (_texPtr)
                ImGui::TextColored(ImVec4(0.5f,1.f,1.f,1.f),
                    "Texture: %s", _expMapSolver.GetTextureLoader().getPath().c_str());
            else
                ImGui::TextColored(ImVec4(1.f,0.5f,0.5f,1.f),
                    "No texture loaded (use File > Load Background)");
            if (!_texGaussians.empty())
                ImGui::TextColored(ImVec4(0.5f,1.f,0.5f,1.f),
                    "Live: %d main-cloud Gaussians", (int)_texGaussians.size());
        } else {
            ImGui::TextColored(ImVec4(0.7f,0.7f,0.7f,1.f),
                "Right-click to select UV region");
        }
        ImGui::End();

        _expMapSolver.RenderUI();

        // If the user picked a new texture in the ExpMap UI, immediately push it
        // to the GPU without requiring another right-click.
        if (_expMapSolver.ConsumeTextureDirty()) {
            _texPtr = _expMapSolver.GetTextureLoader().getTexture();
            if (!_lastAllUVs.empty()) {
                _gaussianView->setUVsAndTexture(_lastAllUVs, _lastAllDUs, _lastAllDVs, _texPtr);
            }
        }
    }

private:
    void performRaycast(const sibr::Input& input) {
        if (!_mesh || !_camHandler) return;

        // --- Build camera ray ---
        const auto& cam = _camHandler->getCamera();
        sibr::Vector2f mp(
            static_cast<float>(input.mousePosition().x()),
            static_cast<float>(input.mousePosition().y())
        );
        float ndcX =  (2.f * mp.x()) / _viewport.finalWidth()  - 1.f;
        float ndcY = 1.f - (2.f * mp.y()) / _viewport.finalHeight();

        sibr::Matrix4f invVP = cam.viewproj().inverse();
        sibr::Vector4f nearPt = invVP * sibr::Vector4f(ndcX, ndcY, -1.f, 1.f);
        sibr::Vector4f farPt  = invVP * sibr::Vector4f(ndcX, ndcY,  1.f, 1.f);
        nearPt /= nearPt.w();
        farPt  /= farPt.w();

        sibr::Vector3f rayOrigin = cam.position();
        sibr::Vector3f rayDir(farPt.x()-nearPt.x(), farPt.y()-nearPt.y(), farPt.z()-nearPt.z());
        rayDir.normalize();

        // --- Moller-Trumbore intersection ---
        const auto& verts     = _mesh->vertices();
        const auto& triangles = _mesh->triangles();
        float minDist = 1e9f;
        int hitTriID = -1;
        sibr::Vector3f hitPos;

        for (size_t t = 0; t < triangles.size(); ++t) {
            const auto& tri = triangles[t];
            sibr::Vector3f e1 = verts[tri[1]] - verts[tri[0]];
            sibr::Vector3f e2 = verts[tri[2]] - verts[tri[0]];
            sibr::Vector3f h  = rayDir.cross(e2);
            float a = e1.dot(h);
            if (a > -1e-6f && a < 1e-6f) continue;

            float f = 1.f / a;
            sibr::Vector3f s = rayOrigin - verts[tri[0]];
            float u = f * s.dot(h);
            if (u < 0.f || u > 1.f) continue;

            sibr::Vector3f q = s.cross(e1);
            float v = f * rayDir.dot(q);
            if (v < 0.f || u + v > 1.f) continue;

            float td = f * e2.dot(q);
            if (td > 1e-6f && td < minDist) {
                minDist = td;
                hitTriID = (int)t;
                hitPos  = rayOrigin + rayDir * td;
            }
        }

        if (hitTriID < 0) {
            std::cout << "[INFO] Raycast missed\n";
            return;
        }

        // --- Update ExpMap solver (computes UV layout) ---
        _expMapSolver.OnRaycastHit(hitPos, _expMapRadius, hitTriID);

        const auto& texLoader = _expMapSolver.GetTextureLoader();
        if (!texLoader.getTexture()) {
            std::cout << "[INFO] No texture.\n";
            return;
        }
        _texPtr = texLoader.getTexture();

        const std::set<int>& activeSet = _expMapSolver.GetActiveTriIndices();
        if (activeSet.empty()) {
            std::cout << "[INFO] No active tris.\n";
            return;
        }

        std::vector<int> validTriIDs(activeSet.begin(), activeSet.end());
        const std::map<int, glm::vec2>& glmUVMap = _expMapSolver.GetDisplayUVs();

        // --- Download main Gaussian cloud data ---
        const std::vector<sibr::Vector3f>& cpuPos = _gaussianView->getCpuPositions();
        const int nGauss = _gaussianView->getCount();

        std::vector<float> cpuRot, cpuScale, cpuOpacity;
        _gaussianView->downloadGaussianData(cpuRot, cpuScale, cpuOpacity);

        // --- Build per-triangle UV + tangent frame cache ---
        struct TriInfo {
            glm::vec3 v0, v1, v2, e1, e2, n, dU, dV;
            float d00, d01, d11;
            glm::vec2 uv0, uv1, uv2;
            bool valid;
        };

        std::vector<TriInfo> triCache;
        triCache.reserve(validTriIDs.size());

        for (int triID : validTriIDs) {
            TriInfo ti;
            ti.valid = false;

            if (triID < 0 || triID >= (int)triangles.size()) {
                triCache.push_back(ti);
                continue;
            }

            const auto& tri = triangles[triID];
            if (!glmUVMap.count(tri[0]) || !glmUVMap.count(tri[1]) || !glmUVMap.count(tri[2])) {
                triCache.push_back(ti);
                continue;
            }

            ti.v0 = {verts[tri[0]].x(), verts[tri[0]].y(), verts[tri[0]].z()};
            ti.v1 = {verts[tri[1]].x(), verts[tri[1]].y(), verts[tri[1]].z()};
            ti.v2 = {verts[tri[2]].x(), verts[tri[2]].y(), verts[tri[2]].z()};
            ti.e1 = ti.v1 - ti.v0;
            ti.e2 = ti.v2 - ti.v0;

            glm::vec3 rn = glm::cross(ti.e1, ti.e2);
            float ln = glm::length(rn);
            if (ln < 1e-9f) {
                triCache.push_back(ti);
                continue;
            }

            ti.n = rn / ln;
            ti.d00 = glm::dot(ti.e1, ti.e1);
            ti.d01 = glm::dot(ti.e1, ti.e2);
            ti.d11 = glm::dot(ti.e2, ti.e2);
            ti.uv0 = glmUVMap.at(tri[0]);
            ti.uv1 = glmUVMap.at(tri[1]);
            ti.uv2 = glmUVMap.at(tri[2]);

            // World-space UV tangent frame
            float du1 = ti.uv1.x - ti.uv0.x;
            float dv1 = ti.uv1.y - ti.uv0.y;
            float du2 = ti.uv2.x - ti.uv0.x;
            float dv2 = ti.uv2.y - ti.uv0.y;
            float det = du1 * dv2 - du2 * dv1;
            if (std::abs(det) < 1e-9f) {
                triCache.push_back(ti);
                continue;
            }

            ti.dU = ( dv2 * ti.e1 - dv1 * ti.e2) / det;
            ti.dV = (-du2 * ti.e1 + du1 * ti.e2) / det;
            ti.valid = true;
            triCache.push_back(ti);
        }

        const float r2 = _expMapRadius * _expMapRadius;
        const glm::vec3 ctr(hitPos.x(), hitPos.y(), hitPos.z());

        _texGaussians.clear();
        _texGaussians.reserve(4096);

        std::vector<sibr::Vector2f> all_uvs(nGauss, sibr::Vector2f(-1.f, -1.f));
        std::vector<sibr::Vector3f> all_dUs(nGauss, sibr::Vector3f(0.f, 0.f, 0.f));
        std::vector<sibr::Vector3f> all_dVs(nGauss, sibr::Vector3f(0.f, 0.f, 0.f));

        // Every Gaussian inside the sphere gets UV from its nearest triangle
        // (no planeDist or margin filter). This ensures all in-sphere Gaussians
        // that can be projected receive a texture UV and are not suppressed.
        // The only Gaussians left suppressed are those with no valid triangle at all.

        for (int k = 0; k < nGauss; ++k) {
            glm::vec3 p(cpuPos[k].x(), cpuPos[k].y(), cpuPos[k].z());
            if (glm::dot(p - ctr, p - ctr) > r2) continue;

            float best = 1e18f;
            glm::vec2 bestUV(0.f, 0.f);
            glm::vec3 bestDU(0.f), bestDV(0.f);
            bool found = false;

            float bestOutDist = 1e18f;
            glm::vec2 bestOutUV(0.f, 0.f);
            glm::vec3 bestOutDU(0.f), bestOutDV(0.f);

            for (const TriInfo& ti : triCache) {
                if (!ti.valid) continue;

                float planeDist = glm::dot(p - ti.v0, ti.n);
                glm::vec3 pv = (p - ti.n * planeDist) - ti.v0;
                float den = ti.d00 * ti.d11 - ti.d01 * ti.d01;
                if (std::abs(den) < 1e-9f) continue;

                float d20 = glm::dot(pv, ti.e1);
                float d21 = glm::dot(pv, ti.e2);
                float bv = (ti.d11 * d20 - ti.d01 * d21) / den;
                float bw = (ti.d00 * d21 - ti.d01 * d20) / den;
                float bu = 1.f - bv - bw;

                const float margin = -0.05f;
                if (bu >= margin && bv >= margin && bw >= margin) {
                    float d2 = planeDist * planeDist;
                    if (d2 < best) {
                        best = d2;
                        bestUV = bu * ti.uv0 + bv * ti.uv1 + bw * ti.uv2;
                        bestDU = ti.dU;
                        bestDV = ti.dV;
                        found = true;
                    }
                } 
                else {
                    float cu = std::max(0.f, bu);
                    float cv = std::max(0.f, bv);
                    float cw = std::max(0.f, bw);
                    float sum = cu + cv + cw;
                    if (sum > 1e-6f) { cu /= sum; cv /= sum; cw /= sum; }

                    glm::vec3 clampedP = cu * ti.v0 + cv * ti.v1 + cw * ti.v2;
                    float d2 = glm::dot(p - clampedP, p - clampedP);

                    if (d2 < bestOutDist) {
                        bestOutDist = d2;
                        bestOutUV = cu * ti.uv0 + cv * ti.uv1 + cw * ti.uv2;
                        bestOutDU = ti.dU;
                        bestOutDV = ti.dV;
                    }
                }
            }
            if (!found && bestOutDist < r2) {
                bestUV = bestOutUV;
                bestDU = bestOutDU;
                bestDV = bestOutDV;
                found = true;
            }

            if (!found) continue;

            all_uvs[k] = {bestUV.x, bestUV.y};
            all_dUs[k] = sibr::Vector3f(bestDU.x, bestDU.y, bestDU.z);
            all_dVs[k] = sibr::Vector3f(bestDV.x, bestDV.y, bestDV.z);

            ProjectedGaussian pg;
            pg.originalIndex = k;
            pg.position      = p;
            pg.uv            = bestUV;
            pg.dU            = bestDU;
            pg.dV            = bestDV;
            pg.originalPos   = p;

            float distToCenter = glm::length(p - ctr);
            float fadeStartRadius = _expMapRadius * 0.8f;
            float alphaMultiplier = 1.0f;

            if (distToCenter > fadeStartRadius) {
                float t = 1.0f - ((distToCenter - fadeStartRadius) / (_expMapRadius - fadeStartRadius));
                alphaMultiplier = t * t * (3.0f - 2.0f * t);
            }

            float opc = std::max(1e-4f, std::min(1.f - 1e-4f, cpuOpacity[k] * alphaMultiplier));
            pg.opacity = std::log(opc / (1.f - opc));

            pg.scale = glm::vec3(
                std::log(std::max(1e-9f, cpuScale[3 * k + 0])),
                std::log(std::max(1e-9f, cpuScale[3 * k + 1])),
                std::log(std::max(1e-9f, cpuScale[3 * k + 2]))
            );

            pg.rotation = glm::vec4(
                cpuRot[4 * k + 0],
                cpuRot[4 * k + 1],
                cpuRot[4 * k + 2],
                cpuRot[4 * k + 3]
            );

            _texGaussians.push_back(pg);
        }

        _liveSSBODirty = true;
        _expMapSolver.SetMainGaussiansForNextSave(_texGaussians);

        // Collect suppressed UV positions for debug visualization.
        // These are in-sphere Gaussians with no valid triangle at all (triCache empty
        // or all degenerate). Show as green dots on UV canvas.
        {
            std::vector<glm::vec2> suppressedUVs;
            suppressedUVs.reserve(64);
            for (int k = 0; k < nGauss; ++k) {
                if (all_uvs[k].x() >= 0.f && all_uvs[k].y() >= 0.f) continue;
                glm::vec3 p(cpuPos[k].x(), cpuPos[k].y(), cpuPos[k].z());
                if (glm::dot(p - ctr, p - ctr) > r2) continue;
                // Try to find nearest UV for visualization only
                float best = 1e18f;
                glm::vec2 bestUV(-1.f, -1.f);
                for (const auto& ti : triCache) {
                    if (!ti.valid) continue;
                    float pd = glm::dot(p - ti.v0, ti.n);
                    glm::vec3 pv = (p - ti.n * pd) - ti.v0;
                    float den = ti.d00 * ti.d11 - ti.d01 * ti.d01;
                    if (std::abs(den) < 1e-9f) continue;
                    float d20 = glm::dot(pv, ti.e1), d21 = glm::dot(pv, ti.e2);
                    float bv = (ti.d11*d20 - ti.d01*d21)/den;
                    float bw = (ti.d00*d21 - ti.d01*d20)/den;
                    float bu = 1.f - bv - bw;
                    bu = std::max(0.f,bu); bv = std::max(0.f,bv); bw = std::max(0.f,bw);
                    float bs = bu+bv+bw; if (bs < 1e-9f) continue;
                    bu/=bs; bv/=bs; bw/=bs;
                    glm::vec3 nr = bu*ti.v0 + bv*ti.v1 + bw*ti.v2;
                    float d2 = glm::dot(p-nr, p-nr);
                    if (d2 < best) { best = d2; bestUV = bu*ti.uv0 + bv*ti.uv1 + bw*ti.uv2; }
                }
                if (bestUV.x >= 0.f && bestUV.x <= 1.f && bestUV.y >= 0.f && bestUV.y <= 1.f)
                    suppressedUVs.push_back(bestUV);
            }
            _expMapSolver.SetSuppressedUVs(suppressedUVs);
            std::cout << "[INFO] Textured=" << _texGaussians.size()
                      << " Suppressed=" << suppressedUVs.size()
                      << " Total=" << nGauss << "\n";
        }

        // Suppress in-sphere Gaussians that have no UV (no valid triangle found).
        _gaussianView->suppressGaussiansInRegion(all_uvs, hitPos, _expMapRadius);

        // Cache UV data so we can re-apply if the user swaps the texture later.
        _lastAllUVs = all_uvs;
        _lastAllDUs = all_dUs;
        _lastAllDVs = all_dVs;

        // Single-pass CUDA render with texture.
        _gaussianView->setUVsAndTexture(all_uvs, all_dUs, all_dVs, _texPtr);
    }

    // ---- members ----
    GaussianView::Ptr               _gaussianView;
    SimplePointRenderer             _geoRenderer;
    SimplePointRenderer             _appRenderer;
    PointCloudRenderer              _pointCloudRenderer;
    MeshWireframeRenderer           _wireframeRenderer;
    GaussianSplatRenderer           _gaussianSplatRenderer;
    MeshDepthStencilRenderer        _meshDepthStencilRenderer;
    ExpMapSolverSIBR                _expMapSolver;

    const sibr::Mesh*                   _mesh;
    sibr::InteractiveCameraHandler::Ptr _camHandler;
    sibr::Viewport                      _viewport;

    sibr::Vector3f _meshColor, _geoColor, _appColor;
    bool   _showMesh     = false;
    bool   _showGeo      = false;
    bool   _showApp      = false;
    float  _pointSize    = 1.0f;
    float  _expMapRadius = 0.5f;

    // Kept for compatibility with existing slot / debug flow.
    std::vector<ProjectedGaussian>  _texGaussians;
    GLuint                          _liveSSBO      = 0;
    bool                            _liveSSBODirty = true;
    sibr::Texture2DRGBA::Ptr        _texPtr;

    // Cached per-Gaussian UV data from the last raycast, so we can re-apply
    // when the user swaps the texture without clicking again.
    std::vector<sibr::Vector2f>     _lastAllUVs;
    std::vector<sibr::Vector3f>     _lastAllDUs;
    std::vector<sibr::Vector3f>     _lastAllDVs;
};

// =============================================================================
// Utilities
// =============================================================================
static std::string findLargestNumberedSubdirectory(const std::string& dirPath) {
    std_fs::path p(dirPath);
    if (!std_fs::exists(p) || !std_fs::is_directory(p)) return "";
    std::regex rx(R"_(iteration_(\d+))_");
    std::string best;
    int bestN = -1;
    for (const auto& entry : std_fs::directory_iterator(p)) {
        if (!std_fs::is_directory(entry)) continue;
        std::string name = entry.path().filename().string();
        std::smatch m;
        if (std::regex_match(name, m, rx)) {
            int n = std::stoi(m[1]);
            if (n > bestN) {
                bestN = n;
                best = name;
            }
        }
    }
    return best;
}

static std::pair<int,int> findArg(const std::string& line, const std::string& name) {
    size_t s = line.find(name, 0);
    s = line.find("=", s) + 1;
    size_t e = line.find_first_of(",)", s);
    return {(int)s, (int)e};
}

static void* User_ReadOpen(ImGuiContext*, ImGuiSettingsHandler*, const char*) {
    return (void*)0x1;
}

static void User_ReadLine(ImGuiContext*, ImGuiSettingsHandler* h, void*, const char* line) {
    int i;
    if (sscanf_s(line, "DontShow=%d", &i) == 1)
        *((bool*)h->UserData) = (i != 0);
}

static void User_WriteAll(ImGuiContext*, ImGuiSettingsHandler* h, ImGuiTextBuffer* buf) {
    buf->reserve(buf->size() + 96);
    buf->appendf("[UserData][UserData]\nDontShow=%d\n\n", *((bool*)h->UserData) ? 1 : 0);
}

// =============================================================================
// main
// =============================================================================
int main(int ac, char** av) {
    CommandLineArgs::parseMainArgs(ac, av);
    GaussianAppArgs myArgs;
    myArgs.displayHelpIfRequired();

    if (!myArgs.modelPath.isInit() && myArgs.modelPathShort.isInit())
        myArgs.modelPath = myArgs.modelPathShort.get();
    if (!myArgs.dataset_path.isInit() && myArgs.pathShort.isInit())
        myArgs.dataset_path = myArgs.pathShort.get();

    sibr::Window window("sibr_3Dgaussian", sibr::Vector2i(50, 50), myArgs);

    bool messageRead = false;
    ImGuiSettingsHandler ini_handler;
    ini_handler.TypeName   = "UserData";
    ini_handler.UserData   = &messageRead;
    ini_handler.TypeHash   = ImHash("UserData", 0);
    ini_handler.ReadOpenFn = User_ReadOpen;
    ini_handler.ReadLineFn = User_ReadLine;
    ini_handler.WriteAllFn = User_WriteAll;
    ImGui::GetCurrentContext()->SettingsHandlers.push_back(ini_handler);
    window.loadSettings();

    std::string cfgLine;
    std::ifstream cfgFile(myArgs.modelPath.get() + "/cfg_args");
    if (!cfgFile.good())
        SIBR_ERR << "Could not find cfg_args at: " << myArgs.modelPath.get();
    std::getline(cfgFile, cfgLine);

    if (!myArgs.dataset_path.isInit()) {
        auto rng = findArg(cfgLine, "source_path");
        myArgs.dataset_path = cfgLine.substr(rng.first + 1, rng.second - rng.first - 2);
    }

    auto rng = findArg(cfgLine, "sh_degree");
    int sh_degree = std::stoi(cfgLine.substr(rng.first, rng.second - rng.first));

    rng = findArg(cfgLine, "white_background");
    bool white_background =
        cfgLine.substr(rng.first, rng.second - rng.first).find("True") != std::string::npos;

    BasicIBRScene::SceneOptions myOpts;
    myOpts.renderTargets = myArgs.loadImages;
    myOpts.mesh    = true;
    myOpts.images  = myArgs.loadImages;
    myOpts.cameras = true;
    myOpts.texture = false;

    BasicIBRScene::Ptr scene;
    try {
        scene.reset(new BasicIBRScene(myArgs, myOpts));
    } catch (...) {
        myArgs.dataset_path = myArgs.modelPath.get();
        scene.reset(new BasicIBRScene(myArgs, myOpts));
    }

    std::string plyBase = myArgs.modelPath.get();
    if (plyBase.back() != '/' && plyBase.back() != '\\')
        plyBase += "/";
    plyBase += "point_cloud";

    std::string iterDir;
    if (!myArgs.iteration.isInit()) {
        iterDir = findLargestNumberedSubdirectory(plyBase);
        std::cout << "Auto-detected iteration: " << iterDir << "\n";
    } else {
        iterDir = "iteration_" + myArgs.iteration.get();
    }

    const std::string plyDir       = plyBase + "/" + iterDir + "/";
    const std::string finalPlyPath = plyDir + "point_cloud.ply";
    const std::string geoPath      = plyDir + "geo_point_cloud.ply";
    const std::string appPath      = plyDir + "app_point_cloud.ply";

    sibr::Mesh::Ptr geoMesh(new sibr::Mesh());
    const sibr::Mesh* meshToRender = nullptr;
    if (std_fs::exists(geoPath) && geoMesh->load(geoPath) && !geoMesh->triangles().empty()) {
        meshToRender = geoMesh.get();
        std::cout << "Using geo mesh: " << geoMesh->triangles().size() << " faces\n";
    } else if (!scene->proxies()->proxy().vertices().empty()) {
        meshToRender = &scene->proxies()->proxy();
        std::cout << "Fallback: proxy mesh\n";
    }

    uint scene_w = scene->cameras()->inputCameras()[0]->w();
    uint scene_h = scene->cameras()->inputCameras()[0]->h();
    float aspect = scene_w * 1.f / scene_h;

    uint rw = myArgs.rendering_size.get()[0];
    uint rh = myArgs.rendering_size.get()[1];
    rw = (rw <= 0) ? std::min(1200U, scene_w) : rw;
    rh = (rh <= 0) ? (uint)(std::min(1200U, scene_w) / aspect) : rh;
    Vector2u usedRes(rw, rh);

    GaussianView::Ptr gaussianView(new GaussianView(
        scene,
        usedRes.x(),
        usedRes.y(),
        finalPlyPath.c_str(),
        &messageRead,
        sh_degree,
        white_background,
        !myArgs.noInterop,
        myArgs.device));

    sibr::InteractiveCameraHandler::Ptr generalCamera(new InteractiveCameraHandler());
    generalCamera->setup(
        scene->cameras()->inputCameras(),
        Viewport(0, 0, (float)usedRes.x(), (float)usedRes.y()),
        nullptr);

    MeshGaussianView::Ptr meshView(new MeshGaussianView(
        gaussianView, meshToRender, geoPath, appPath, generalCamera));

    MultiViewManager multiViewManager(window, false);
    if (myArgs.rendering_mode == 1)
        multiViewManager.renderingMode(IRenderingMode::Ptr(new StereoAnaglyphRdrMode()));

    multiViewManager.addIBRSubView(
        "Point view",
        meshView,
        usedRes,
        ImGuiWindowFlags_ResizeFromAnySide | ImGuiWindowFlags_NoBringToFrontOnFocus);
    multiViewManager.addCameraForView("Point view", generalCamera);

    const auto topView = std::make_shared<sibr::SceneDebugView>(
        scene, generalCamera, myArgs, myArgs.imagesPath.get());
    multiViewManager.addSubView("Top view", topView, usedRes);
    topView->active(false);

    generalCamera->getCameraRecorder().setViewPath(gaussianView, myArgs.dataset_path.get());
    if (myArgs.pathFile.get() != "") {
        generalCamera->getCameraRecorder().loadPath(myArgs.pathFile.get(), usedRes.x(), usedRes.y());
        generalCamera->getCameraRecorder().recordOfflinePath(
            myArgs.outPath,
            multiViewManager.getIBRSubView("Point view"),
            "");
        if (!myArgs.noExit)
            exit(0);
    }

    while (window.isOpened()) {
        sibr::Input::poll();
        window.makeContextCurrent();
        if (sibr::Input::global().key().isPressed(sibr::Key::Escape))
            window.close();
        multiViewManager.onUpdate(sibr::Input::global());
        multiViewManager.onRender(window);
        window.swapBuffer();
    }

    return EXIT_SUCCESS;
}