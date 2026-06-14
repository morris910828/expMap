#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <filesystem>
#include <regex>
#include <array>
#include <cmath>
#include <unordered_map>
#include <opencv2/videoio.hpp>

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
        const sibr::Mesh* mesh,
        const std::string&                  geoPath,
        const std::string&                  appPath,
        sibr::InteractiveCameraHandler::Ptr camHandler
    )   : ViewBase(gaussianView->getResolution().x(), gaussianView->getResolution().y()),
          _gaussianView(gaussianView),
          _mesh(mesh),
          _camHandler(camHandler),
          _viewport(0, 0, (float)gaussianView->getResolution().x(), (float)gaussianView->getResolution().y())
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
                _expMapSolver.RegisterGeoPointCloud(_geoRenderer.getRawData(), _geoRenderer.getGaussianProps());
            if (_appRenderer.isLoaded())
                _expMapSolver.RegisterAppPointCloud(_appRenderer.getRawData(), _appRenderer.getFids(), _appRenderer.getGaussianProps());
            _wireframeRenderer.uploadMesh(_mesh);
        }
        _meshColor = sibr::Vector3f(0.0f, 0.6f, 0.0f);
        _geoColor  = sibr::Vector3f(1.0f, 0.2f, 0.0f);
        _appColor  = sibr::Vector3f(0.0f, 0.4f, 1.0f);
        initGaussianOutlineRenderer();
    }

    ~MeshGaussianView() {
        if (_liveSSBO) glDeleteBuffers(1, &_liveSSBO);
        if (_gaussOutlineVAO)    glDeleteVertexArrays(1, &_gaussOutlineVAO);
        if (_gaussOutlineVBO)    glDeleteBuffers(1, &_gaussOutlineVBO);
        if (_gaussOutlineShader) glDeleteProgram(_gaussOutlineShader);
    }

    void onRenderIBR(sibr::IRenderTarget& dst, const sibr::Camera& eye) override {
        if (!_gaussianView) return;

        if (_recording360) {
            constexpr float kPi = 3.14159265358979323846f;
            float angle    = (2.0f * kPi * _recordFrame) / (float)_recordTotalFrames;
            float elevRad  = _orbitElevation * kPi / 180.0f;
            float horizR   = _orbitRadius * std::cos(elevRad);
            glm::vec3 camPos(
                _orbitCenter.x + horizR * std::sin(angle),
                _orbitCenter.y + _orbitRadius * std::sin(elevRad),
                _orbitCenter.z + horizR * std::cos(angle)
            );
            sibr::Camera orbitCam = eye;
            orbitCam.setLookAt(
                sibr::Vector3f(camPos.x, camPos.y, camPos.z),
                sibr::Vector3f(_orbitCenter.x, _orbitCenter.y, _orbitCenter.z),
                sibr::Vector3f(0.f, 1.f, 0.f)
            );

            _gaussianView->onRenderIBR(dst, orbitCam);
            _viewport = Viewport(0, 0, (float)dst.w(), (float)dst.h());
            dst.bind();
            glViewport(0, 0, dst.w(), dst.h());
            const glm::mat4 mvp = glm::make_mat4(orbitCam.viewproj().data());
            _pointCloudRenderer.render(mvp, _pointSize, _geoRenderer, _geoColor, _showGeo, _appRenderer, _appColor, _showApp);
            if (_showGaussianOutlines) renderGaussianOutlines(orbitCam);
            _wireframeRenderer.render(_mesh, _expMapSolver.GetActiveTriIndices(), mvp, _showMesh);

            // Capture frame and write to video
            int W = dst.w(), H = dst.h();
            if (!_videoWriter.isOpened()) {
                std::string videoPath = _videoDir + "/orbit.mp4";
                _videoWriter.open(videoPath,
                    cv::VideoWriter::fourcc('m','p','4','v'), 30.0, cv::Size(W, H));
                if (!_videoWriter.isOpened())
                    std::cerr << "[Record360] Failed to open VideoWriter: " << videoPath << "\n";
            }
            cv::Mat frame(H, W, CV_8UC3);
            glReadPixels(0, 0, W, H, GL_BGR, GL_UNSIGNED_BYTE, frame.data);
            dst.unbind();
            cv::flip(frame, frame, 0);
            if (_videoWriter.isOpened())
                _videoWriter.write(frame);

            _recordFrame++;
            if (_recordFrame >= _recordTotalFrames) {
                _recording360 = false;
                _videoWriter.release();
                std::cout << "[Record360] Done. Video saved to: "
                          << _videoDir << "/orbit.mp4\n";
            }
        } else {
            _gaussianView->onRenderIBR(dst, eye);
            _viewport = Viewport(0, 0, (float)dst.w(), (float)dst.h());
            dst.bind();
            glViewport(0, 0, dst.w(), dst.h());
            const glm::mat4 mvp = glm::make_mat4(eye.viewproj().data());
            _pointCloudRenderer.render(mvp, _pointSize, _geoRenderer, _geoColor, _showGeo, _appRenderer, _appColor, _showApp);
            if (_showGaussianOutlines) renderGaussianOutlines(eye);
            _wireframeRenderer.render(_mesh, _expMapSolver.GetActiveTriIndices(), mvp, _showMesh);
            dst.unbind();
        }
    }

    void onUpdate(sibr::Input& input) override {
        if (!_gaussianView) return;
        _gaussianView->onUpdate(input);
        if (input.mouseButton().isReleased(sibr::Mouse::Right) &&
            input.key().isActivated(sibr::Key::LeftShift))
            performRaycast(input);
    }

    void onGUI() override {
        if (!_gaussianView) return;
        _gaussianView->onGUI();
        ImGui::Begin("Mesh & Controls");
        ImGui::Checkbox("Show Mesh", &_showMesh);
        ImGui::Checkbox("Show Yellow Wireframe", &_wireframeRenderer._showYellowWireframe);
        ImGui::Checkbox("Show Geo Points", &_showGeo);
        ImGui::Checkbox("Show App Points", &_showApp);
        ImGui::SliderFloat("Point Size", &_pointSize, 1.0f, 3.0f);
        ImGui::SliderFloat("ExpMap Radius", &_expMapRadius, 0.05f, 5.0f);
        ImGui::Separator();
        ImGui::Checkbox("Show Gaussian Outlines", &_showGaussianOutlines);
        ImGui::Separator();
        {
            bool prevShowTex = _showTexture;
            ImGui::Checkbox("Show Texture", &_showTexture);
            if (_showTexture != prevShowTex && !_lastAllUVs.empty()) {
                _gaussianView->restoreOpacities();
                auto blended = computeBlendedPos(_expMapSolver.GetSurfaceBlend());
                _gaussianView->setUVsAndTexture(
                    _lastAllUVs, _lastAllDUs, _lastAllDVs, _lastAllSurfDists,
                    blended,
                    _showTexture ? _texPtr : nullptr
                );
            }
        }
        ImGui::Separator();
        if (!_recording360) {
            ImGui::InputInt("360 Frames##360f", &_recordTotalFrames);
            if (_recordTotalFrames < 1) _recordTotalFrames = 1;
            ImGui::SliderFloat("Orbit Zoom##oz", &_orbitZoom, 0.1f, 10.0f, "%.2f");
            ImGui::SliderFloat("Elevation##elev", &_orbitElevation, -89.0f, 89.0f, "%.1f deg");
            if (ImGui::Button("Record 360\xc2\xb0")) {
                // Compute orbit center from mesh bounding box
                if (_mesh && !_mesh->vertices().empty()) {
                    glm::vec3 mn(1e9f), mx(-1e9f);
                    for (const auto& v : _mesh->vertices()) {
                        glm::vec3 gv(v.x(), v.y(), v.z());
                        mn = glm::min(mn, gv);
                        mx = glm::max(mx, gv);
                    }
                    _orbitCenter = (mn + mx) * 0.5f;
                } else {
                    _orbitCenter = {0.f, 0.f, 0.f};
                }
                const sibr::Camera& cam = _camHandler->getCamera();
                glm::vec3 camPos(cam.position().x(), cam.position().y(), cam.position().z());
                glm::vec2 toXZ(_orbitCenter.x - camPos.x, _orbitCenter.z - camPos.z);
                float baseRadius = glm::length(toXZ);
                if (baseRadius < 1e-4f) baseRadius = 1.0f;
                _orbitRadius = baseRadius * _orbitZoom;
                _videoDir = "D:/SGGaussians/SGGaussians/video";
                std_fs::create_directories(_videoDir);
                _recordFrame = 0;
                _recording360 = true;
                std::cout << "[Record360] Start " << _recordTotalFrames
                          << " frames, center=(" << _orbitCenter.x << ","
                          << _orbitCenter.y << "," << _orbitCenter.z
                          << "), r=" << _orbitRadius << ", elev=" << _orbitElevation << "\n";
            }
        } else {
            ImGui::Text("Recording %d / %d ...", _recordFrame, _recordTotalFrames);
            if (ImGui::Button("Cancel##360cancel")) _recording360 = false;
        }
        ImGui::Separator();
        if (ImGui::Button("Clear ExpMap")) {
            _gaussianView->restoreOpacities();
            _texGaussians.clear();
            _expMapSolver.ClearRaycastState();
            _lastAllOrigPos.clear();
            std::vector<sibr::Vector2f> empty_uvs(_gaussianView->getCount(), sibr::Vector2f(-1.f, -1.f));
            std::vector<sibr::Vector3f> empty_dUs(_gaussianView->getCount(), sibr::Vector3f(0.f, 0.f, 0.f));
            std::vector<sibr::Vector3f> empty_dVs(_gaussianView->getCount(), sibr::Vector3f(0.f, 0.f, 0.f));
            std::vector<float>          empty_sd(_gaussianView->getCount(), 1e9f);
            std::vector<sibr::Vector3f> empty_sp(_gaussianView->getCount(), sibr::Vector3f(-1.f,-1.f,-1.f));
            _gaussianView->setUVsAndTexture(empty_uvs, empty_dUs, empty_dVs, empty_sd, empty_sp, nullptr);
        }
        ImGui::End();
        _expMapSolver.RenderUI();
        // Surface blend slider changed -> update GPU positions with new blend amount
        if (_expMapSolver.ConsumeSurfaceBlendDirty() && !_lastAllUVs.empty() && !_lastAllOrigPos.empty()) {
            auto blended = computeBlendedPos(_expMapSolver.GetSurfaceBlend());
            _gaussianView->setUVsAndTexture(_lastAllUVs, _lastAllDUs, _lastAllDVs, _lastAllSurfDists, blended, _texPtr);
        }
        if (_expMapSolver.ConsumeTextureDirty()) {
            _texPtr = _expMapSolver.GetTextureLoader().getTexture();
            if (!_lastAllUVs.empty()) {
                auto blended = computeBlendedPos(_expMapSolver.GetSurfaceBlend());
                _gaussianView->setUVsAndTexture(_lastAllUVs, _lastAllDUs, _lastAllDVs, _lastAllSurfDists, blended, _texPtr);
            }
        }
    }

private:
    // Compute per-Gaussian surface position interpolated by blend (0=original, 1=surface).
    std::vector<sibr::Vector3f> computeBlendedPos(float blend) const {
        const int n = (int)_lastAllOrigPos.size();
        std::vector<sibr::Vector3f> out(n, sibr::Vector3f(-1.f, -1.f, -1.f));
        for (int i = 0; i < n; ++i) {
            if (i < (int)_lastAllUVs.size() && _lastAllUVs[i].x() >= 0.f)
                out[i] = _lastAllOrigPos[i] + blend * (_lastAllSurfPos[i] - _lastAllOrigPos[i]);
        }
        return out;
    }

    // -------------------------------------------------------------------------
    // Gaussian outline rendering (3-D ellipses in the mesh tangent plane)
    // -------------------------------------------------------------------------
    void initGaussianOutlineRenderer() {
        const std::string vsSrc = R"(
#version 330 core
layout(location = 0) in vec3 aPos;
uniform mat4 uMVP;
void main() { gl_Position = uMVP * vec4(aPos, 1.0); }
)";
        const std::string fsSrc = R"(
#version 330 core
uniform vec4 uColor;
out vec4 fragColor;
void main() { fragColor = uColor; }
)";
        GLuint vs = detail::compileGLShader(vsSrc, GL_VERTEX_SHADER);
        GLuint fs = detail::compileGLShader(fsSrc, GL_FRAGMENT_SHADER);
        _gaussOutlineShader = detail::linkProgram(vs, fs);

        GLint linkOK = 0;
        glGetProgramiv(_gaussOutlineShader, GL_LINK_STATUS, &linkOK);
        if (!linkOK) {
            GLchar log[512]; glGetProgramInfoLog(_gaussOutlineShader, 512, nullptr, log);
            std::cerr << "[GaussianOutline] Shader link error: " << log << "\n";
            glDeleteProgram(_gaussOutlineShader);
            _gaussOutlineShader = 0;
            return;
        }

        glGenVertexArrays(1, &_gaussOutlineVAO);
        glGenBuffers(1, &_gaussOutlineVBO);
        glBindVertexArray(_gaussOutlineVAO);
        glBindBuffer(GL_ARRAY_BUFFER, _gaussOutlineVBO);
        glBufferData(GL_ARRAY_BUFFER, 34 * sizeof(glm::vec3), nullptr, GL_DYNAMIC_DRAW); // pre-allocate
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
        glEnableVertexAttribArray(0);
        glBindVertexArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        std::cout << "[GaussianOutline] Shader initialized OK. Prog=" << _gaussOutlineShader
                  << " VAO=" << _gaussOutlineVAO << " VBO=" << _gaussOutlineVBO << "\n";
    }

    void renderGaussianOutlines(const sibr::Camera& eye) {
        if (!_gaussOutlineShader || !_gaussOutlineVAO) return;

        const auto& geoPoints = _expMapSolver.GetProjectedGeoPoints();
        const auto& appPoints = _expMapSolver.GetProjectedAppPoints();
        if (geoPoints.empty() && appPoints.empty()) return;

        static constexpr float kPi = 3.14159265358979323846f;
        static constexpr int   SEGS = 32;

        // Camera position & forward vector in world space (for depth sorting)
        const glm::vec3 camPos(eye.position().x(), eye.position().y(), eye.position().z());
        // Linear depth along camera forward axis (more accurate than squared distance)
        const sibr::Vector3f eyeDir = eye.dir();
        const glm::vec3 camFwd(eyeDir.x(), eyeDir.y(), eyeDir.z());

        // ---------------------------------------------------------------
        // Compute the exact 1-sigma ellipse ring for one Gaussian.
        // The 3D covariance is projected onto the mesh tangent plane (dU/dV),
        // giving the actual footprint shape, size and orientation on the surface.
        // ---------------------------------------------------------------
        const float outlineBlend = _expMapSolver.GetSurfaceBlend();
        auto blendedCenter = [&](const ProjectedGaussian& pg) -> glm::vec3 {
            return pg.originalPos + outlineBlend * (pg.position - pg.originalPos);
        };

        auto makeRing = [&](const ProjectedGaussian& pg) -> std::vector<glm::vec3> {
            // Quaternion layout: rotation.x=w, .y=x, .z=y, .w=z
            const float qw = pg.rotation.x, qx = pg.rotation.y;
            const float qy = pg.rotation.z, qz = pg.rotation.w;

            // Rotation matrix (GLM column-major): R[i] = world direction of axis i
            const glm::mat3 R(
                1.f-2.f*(qy*qy+qz*qz),  2.f*(qx*qy+qw*qz),   2.f*(qx*qz-qw*qy),
                2.f*(qx*qy-qw*qz),    1.f-2.f*(qx*qx+qz*qz),  2.f*(qy*qz+qw*qx),
                2.f*(qx*qz+qw*qy),    2.f*(qy*qz-qw*qx),   1.f-2.f*(qx*qx+qy*qy)
            );

            const float s0 = std::exp(pg.scale.x);
            const float s1 = std::exp(pg.scale.y);
            const float s2 = std::exp(pg.scale.z);

            // Build orthonormal tangent basis (t1, t2) from mesh dU/dV vectors.
            // If dU is degenerate, fall back to drawing using the two largest 3D axes.
            glm::vec3 t1 = pg.dU;
            const float lenU = glm::length(t1);
            if (lenU < 1e-7f) {
                float sArr[3] = {s0, s1, s2};
                int idx[3] = {0, 1, 2};
                if (sArr[idx[0]] < sArr[idx[1]]) std::swap(idx[0], idx[1]);
                if (sArr[idx[0]] < sArr[idx[2]]) std::swap(idx[0], idx[2]);
                if (sArr[idx[1]] < sArr[idx[2]]) std::swap(idx[1], idx[2]);
                std::vector<glm::vec3> ring(SEGS);
                const glm::vec3 ctr0 = blendedCenter(pg);
                for (int i = 0; i < SEGS; ++i) {
                    const float phi = 2.f * kPi * i / SEGS;
                    ring[i] = ctr0 + sArr[idx[0]]*std::cos(phi)*R[idx[0]]
                                   + sArr[idx[1]]*std::sin(phi)*R[idx[1]];
                }
                return ring;
            }
            t1 /= lenU;

            glm::vec3 t2 = pg.dV - glm::dot(pg.dV, t1) * t1;
            const float lenV = glm::length(t2);
            if (lenV < 1e-7f) {
                glm::vec3 arb = (std::abs(t1.x) < 0.9f) ? glm::vec3(1,0,0) : glm::vec3(0,1,0);
                t2 = glm::normalize(glm::cross(t1, arb));
            } else {
                t2 /= lenV;
            }

            // Project the 3D covariance Σ = R diag(s²) R^T onto the mesh tangent plane.
            // This gives the ellipse footprint that lies flat on the triangle surface.
            const glm::mat3 Rt = glm::transpose(R);
            auto SigmaV = [&](const glm::vec3& v) -> glm::vec3 {
                glm::vec3 u = Rt * v;
                return R * glm::vec3(s0*s0*u.x, s1*s1*u.y, s2*s2*u.z);
            };

            const glm::vec3 St1 = SigmaV(t1), St2 = SigmaV(t2);
            const float a2d = glm::dot(t1, St1);
            const float b2d = glm::dot(t1, St2);
            const float c2d = glm::dot(t2, St2);

            const float disc = std::sqrt(std::max(0.f, 0.25f*(a2d-c2d)*(a2d-c2d) + b2d*b2d));
            const float lam1 = std::max(0.f, 0.5f*(a2d+c2d) + disc);
            const float lam2 = std::max(0.f, 0.5f*(a2d+c2d) - disc);

            const float semiA = std::sqrt(lam1);
            const float semiB = std::sqrt(lam2);

            glm::vec2 ev1;
            if (std::abs(b2d) > 1e-8f) {
                ev1 = glm::normalize(glm::vec2(b2d, lam1 - a2d));
            } else {
                ev1 = (a2d >= c2d) ? glm::vec2(1.f, 0.f) : glm::vec2(0.f, 1.f);
            }
            const glm::vec2 ev2(-ev1.y, ev1.x);

            const glm::vec3 axisA = semiA * (ev1.x * t1 + ev1.y * t2);
            const glm::vec3 axisB = semiB * (ev2.x * t1 + ev2.y * t2);

            std::vector<glm::vec3> ring(SEGS);
            const glm::vec3 ctr1 = blendedCenter(pg);
            for (int i = 0; i < SEGS; ++i) {
                const float phi = 2.f * kPi * i / SEGS;
                ring[i] = ctr1 + std::cos(phi)*axisA + std::sin(phi)*axisB;
            }
            return ring;
        };

        static constexpr float kFillAlpha    = 0.11f;
        static constexpr float kOutlineAlpha = 0.90f;

        auto drawOne = [&](const std::vector<glm::vec3>& ring,
                            const glm::vec3& center,
                            GLint colorLoc,
                            float cr, float cg, float cb) {
            std::vector<glm::vec3> fan;
            fan.reserve(SEGS + 2);
            fan.push_back(center);
            fan.insert(fan.end(), ring.begin(), ring.end());
            fan.push_back(ring[0]);

            glUniform4f(colorLoc, cr, cg, cb, kFillAlpha);
            glBufferData(GL_ARRAY_BUFFER,
                         (GLsizeiptr)(fan.size() * sizeof(glm::vec3)),
                         fan.data(), GL_STREAM_DRAW);
            glDrawArrays(GL_TRIANGLE_FAN, 0, (GLsizei)fan.size());

            glUniform4f(colorLoc, cr, cg, cb, kOutlineAlpha);
            glBufferData(GL_ARRAY_BUFFER,
                         (GLsizeiptr)(ring.size() * sizeof(glm::vec3)),
                         ring.data(), GL_STREAM_DRAW);
            glDrawArrays(GL_LINE_LOOP, 0, (GLsizei)ring.size());
        };

        // ---------------------------------------------------------------
        // Collect all selected Gaussians (geo + app) and sort back-to-front
        // so semi-transparent ellipses blend correctly.
        // ---------------------------------------------------------------
        struct DrawItem {
            const ProjectedGaussian* pg;
            bool isGeo;
            float depth;  // squared distance from camera (for sorting)
        };
        std::vector<DrawItem> items;
        items.reserve(geoPoints.size() + appPoints.size());

        const float surfDistMin = _expMapSolver.GetSurfDistMin();
        const float surfDistMax = _expMapSolver.GetSurfDistMax();
        for (const auto& pg : geoPoints) {
            float d = glm::length(pg.originalPos - pg.position);
            if (d < surfDistMin || d > surfDistMax) continue;
            float depth = glm::dot(blendedCenter(pg) - camPos, camFwd);
            items.push_back({&pg, true, depth});
        }
        for (const auto& pg : appPoints) {
            float d = glm::length(pg.originalPos - pg.position);
            if (d < surfDistMin || d > surfDistMax) continue;
            float depth = glm::dot(blendedCenter(pg) - camPos, camFwd);
            items.push_back({&pg, false, depth});
        }

        // Back-to-front (largest depth first -> drawn first, so near ones on top)
        std::sort(items.begin(), items.end(),
                  [](const DrawItem& a, const DrawItem& b){ return a.depth > b.depth; });

        // ---- setup GL state ----
        // Unbind SSBO slot 0 left by the Gaussian copy-renderer
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, 0);

        glm::mat4 mvp = glm::make_mat4(eye.viewproj().data());

        GLboolean depthWas = GL_FALSE, blendWas = GL_FALSE;
        glGetBooleanv(GL_DEPTH_TEST, &depthWas);
        glGetBooleanv(GL_BLEND,      &blendWas);

        glDisable(GL_DEPTH_TEST);
        glDisable(GL_CULL_FACE);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        glUseProgram(_gaussOutlineShader);
        GLint mvpLoc   = glGetUniformLocation(_gaussOutlineShader, "uMVP");
        GLint colorLoc = glGetUniformLocation(_gaussOutlineShader, "uColor");
        if (mvpLoc == -1 || colorLoc == -1) {
            std::cerr << "[GaussianOutline] Uniform not found: uMVP=" << mvpLoc
                      << " uColor=" << colorLoc << "\n";
        }
        glUniformMatrix4fv(mvpLoc, 1, GL_FALSE, glm::value_ptr(mvp));

        glBindVertexArray(_gaussOutlineVAO);
        glBindBuffer(GL_ARRAY_BUFFER, _gaussOutlineVBO);

        // Draw in depth-sorted order (back -> front)
        int drawnGeo = 0, drawnApp = 0;
        for (const auto& item : items) {
            if (item.isGeo) {
                drawOne(makeRing(*item.pg), blendedCenter(*item.pg), colorLoc, 1.f, 0.2f, 0.2f);
                ++drawnGeo;
            } else {
                drawOne(makeRing(*item.pg), blendedCenter(*item.pg), colorLoc, 0.2f, 0.5f, 1.f);
                ++drawnApp;
            }
        }

        // Log once when count changes
        static size_t lastDrawTotal = 0;
        size_t curDrawTotal = (size_t)(drawnGeo + drawnApp);
        if (curDrawTotal != lastDrawTotal) {
            lastDrawTotal = curDrawTotal;
            std::cout << "[GaussianOutline] drew geo=" << drawnGeo
                      << " app=" << drawnApp << "\n";
        }

        // ---- restore GL state ----
        glBindVertexArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glUseProgram(0);
        if (!blendWas)  glDisable(GL_BLEND);
        if (depthWas)   glEnable(GL_DEPTH_TEST);
    }

    // -------------------------------------------------------------------------

    void performRaycast(const sibr::Input& input) {
        if (!_mesh || !_camHandler) return;
        const auto& cam = _camHandler->getCamera();
        sibr::Vector2f mp((float)input.mousePosition().x(), (float)input.mousePosition().y());
        float ndcX = (2.f * mp.x()) / _viewport.finalWidth() - 1.f;
        float ndcY = 1.f - (2.f * mp.y()) / _viewport.finalHeight();
        sibr::Matrix4f invVP = cam.viewproj().inverse();
        sibr::Vector4f nearPt = invVP * sibr::Vector4f(ndcX, ndcY, -1.f, 1.f);
        sibr::Vector4f farPt  = invVP * sibr::Vector4f(ndcX, ndcY,  1.f, 1.f);
        nearPt /= nearPt.w(); farPt /= farPt.w();
        sibr::Vector3f rayDir(farPt.x()-nearPt.x(), farPt.y()-nearPt.y(), farPt.z()-nearPt.z());
        rayDir.normalize();

        float minDist = 1e9f; int hitTriID = -1; sibr::Vector3f hitPos;
        const auto& verts = _mesh->vertices();
        const auto& triangles = _mesh->triangles();
        for (size_t t = 0; t < triangles.size(); ++t) {
            const auto& tri = triangles[t];
            sibr::Vector3f e1 = verts[tri[1]] - verts[tri[0]], e2 = verts[tri[2]] - verts[tri[0]], h = rayDir.cross(e2);
            float a = e1.dot(h); if (std::abs(a) < 1e-6f) continue;
            float f = 1.f/a; sibr::Vector3f s = cam.position() - verts[tri[0]];
            float u = f * s.dot(h); if (u < 0.f || u > 1.f) continue;
            sibr::Vector3f q = s.cross(e1); float v = f * rayDir.dot(q); if (v < 0.f || u + v > 1.f) continue;
            float td = f * e2.dot(q); if (td > 1e-6f && td < minDist) { minDist = td; hitTriID = (int)t; hitPos = cam.position() + rayDir * td; }
        }
        if (hitTriID < 0) return;

        _expMapSolver.OnRaycastHit(hitPos, _expMapRadius, hitTriID);
        _texPtr = _expMapSolver.GetTextureLoader().getTexture();
        if (!_texPtr) return;

        // Restore GPU positions to original before reading, so cpuPos is always
        // the true pre-snap positions (needed for correct blend origin).
        _gaussianView->restoreOpacities();

        const std::set<int>& activeSet = _expMapSolver.GetActiveTriIndices();
        const std::map<int, glm::vec2>& glmUVMap = _expMapSolver.GetDisplayUVs();
        const int nGauss = _gaussianView->getCount();
        const auto& cpuPos = _gaussianView->getCpuPositions();
        std::vector<float> cpuRot, cpuScale, cpuOpacity;
        _gaussianView->downloadGaussianData(cpuRot, cpuScale, cpuOpacity);

        struct TriInfo { glm::vec3 v0, v1, v2, e1, e2, n, dU, dV; float d00, d01, d11; glm::vec2 uv0, uv1, uv2; bool valid; int vi0, vi1, vi2; };
        std::vector<TriInfo> triCache;
        // Also build a global-triID → cache-index map for O(1) fid lookup
        std::unordered_map<int,int> triIdxToCache;
        triIdxToCache.reserve(activeSet.size());
        {
            int cIdx = 0;
            for (int triID : activeSet) {
                triIdxToCache[triID] = cIdx++;
                TriInfo ti; ti.valid = false; const auto& tri = triangles[triID];
                ti.vi0 = tri[0]; ti.vi1 = tri[1]; ti.vi2 = tri[2];
                if (!glmUVMap.count(tri[0]) || !glmUVMap.count(tri[1]) || !glmUVMap.count(tri[2])) { triCache.push_back(ti); continue; }
                ti.v0 = {verts[tri[0]].x(), verts[tri[0]].y(), verts[tri[0]].z()};
                ti.v1 = {verts[tri[1]].x(), verts[tri[1]].y(), verts[tri[1]].z()};
                ti.v2 = {verts[tri[2]].x(), verts[tri[2]].y(), verts[tri[2]].z()};
                ti.e1 = ti.v1 - ti.v0; ti.e2 = ti.v2 - ti.v0;
                glm::vec3 rn = glm::cross(ti.e1, ti.e2); float ln = glm::length(rn);
                if (ln < 1e-9f) { triCache.push_back(ti); continue; }
                ti.n = rn/ln; ti.d00 = glm::dot(ti.e1, ti.e1); ti.d01 = glm::dot(ti.e1, ti.e2); ti.d11 = glm::dot(ti.e2, ti.e2);
                ti.uv0 = glmUVMap.at(tri[0]); ti.uv1 = glmUVMap.at(tri[1]); ti.uv2 = glmUVMap.at(tri[2]);
                float du1 = ti.uv1.x - ti.uv0.x, dv1 = ti.uv1.y - ti.uv0.y, du2 = ti.uv2.x - ti.uv0.x, dv2 = ti.uv2.y - ti.uv0.y, det = du1*dv2 - du2*dv1;
                if (std::abs(det) < 1e-9f) { triCache.push_back(ti); continue; }
                ti.dU = (dv2*ti.e1 - dv1*ti.e2)/det; ti.dV = (-du2*ti.e1 + du1*ti.e2)/det; ti.valid = true; triCache.push_back(ti);
            }
        }

        // Determine geo/app split: point_cloud.ply = [geo (N_geo)] ++ [app (N_app)]
        // App Gaussians have a pre-assigned face ID; use it directly instead of
        // nearest-triangle search to avoid steeply-inclined triangles stealing them.
        const int nGeo = _geoRenderer.isLoaded()
                         ? (int)(_geoRenderer.getRawData().size() / 3) : nGauss;
        const auto& appFids = _appRenderer.getFids();
        const bool canUseFids = _appRenderer.isLoaded() &&
                                (int)appFids.size() == (nGauss - nGeo);

        const float r2 = _expMapRadius * _expMapRadius;
        const glm::vec3 ctr(hitPos.x(), hitPos.y(), hitPos.z());
        std::vector<sibr::Vector2f> all_uvs(nGauss, sibr::Vector2f(-1.f, -1.f));
        std::vector<sibr::Vector3f> all_dUs(nGauss, sibr::Vector3f(0.f, 0.f, 0.f));
        std::vector<sibr::Vector3f> all_dVs(nGauss, sibr::Vector3f(0.f, 0.f, 0.f));
        std::vector<float>          all_surfDists(nGauss, 1e9f);
        std::vector<sibr::Vector3f> all_surfPos(nGauss, sibr::Vector3f(-1.f, -1.f, -1.f));
        _texGaussians.clear();

        for (int k = 0; k < nGauss; ++k) {
            glm::vec3 p(cpuPos[k].x(), cpuPos[k].y(), cpuPos[k].z());
            if (glm::dot(p - ctr, p - ctr) > r2) continue;

            glm::vec2 bUV(0.f, 0.f);
            glm::vec3 bDU(0.f), bDV(0.f);
            glm::vec3 bSurfPos(0.f);
            float bUVMaxDelta = 0.0f;
            float surfDist = 1e9f;
            bool found = false;

            const int appIdx = k - nGeo;
            const bool isAppWithFid = canUseFids && appIdx >= 0 && appIdx < (int)appFids.size();

            // App Gaussian: try fid-based projection first to avoid inclined-triangle error.
            if (isAppWithFid) {
                int fid = appFids[appIdx];
                auto it = triIdxToCache.find(fid);
                if (it != triIdxToCache.end()) {
                    const TriInfo& ti = triCache[it->second];
                    if (ti.valid) {
                        float planeDist = glm::dot(p - ti.v0, ti.n);
                        glm::vec3 pv = (p - ti.n * planeDist) - ti.v0;
                        float den = ti.d00 * ti.d11 - ti.d01 * ti.d01;
                        if (std::abs(den) >= 1e-9f) {
                            float d20 = glm::dot(pv, ti.e1);
                            float d21 = glm::dot(pv, ti.e2);
                            float bv = (ti.d11 * d20 - ti.d01 * d21) / den;
                            float bw = (ti.d00 * d21 - ti.d01 * d20) / den;
                            float bu = 1.f - bv - bw;
                            bUV      = bu * ti.uv0 + bv * ti.uv1 + bw * ti.uv2;
                            bSurfPos = bu * ti.v0   + bv * ti.v1   + bw * ti.v2;
                            bDU = ti.dU; bDV = ti.dV;
                            glm::vec2 uvMin = glm::min(glm::min(ti.uv0, ti.uv1), ti.uv2);
                            glm::vec2 uvMax = glm::max(glm::max(ti.uv0, ti.uv1), ti.uv2);
                            bUVMaxDelta = glm::length(uvMax - uvMin) * 0.5f;
                            surfDist = std::abs(planeDist);
                            found = true;
                        }
                    }
                }
            }

            // Geo Gaussians, and app Gaussians whose bound triangle is not in the
            // active set: fall back to nearest-triangle (preserves original behaviour).
            if (!found) {
                float best = 1e18f;
                for (const auto& ti : triCache) {
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
                    if (bu >= -0.001f && bv >= -0.001f && bw >= -0.001f) {
                        float d2 = planeDist * planeDist;
                        if (d2 < best) {
                            best = d2;
                            bUV      = bu * ti.uv0 + bv * ti.uv1 + bw * ti.uv2;
                            bSurfPos = bu * ti.v0   + bv * ti.v1   + bw * ti.v2;
                            bDU = ti.dU; bDV = ti.dV;
                            glm::vec2 uvMin = glm::min(glm::min(ti.uv0, ti.uv1), ti.uv2);
                            glm::vec2 uvMax = glm::max(glm::max(ti.uv0, ti.uv1), ti.uv2);
                            bUVMaxDelta = glm::length(uvMax - uvMin) * 0.5f;
                            found = true;
                        }
                    }
                }
                if (found) surfDist = std::sqrt(best);
            }

            if (!found) continue;

            all_uvs[k]       = { bUV.x, bUV.y };
            all_dUs[k]       = { bDU.x, bDU.y, bDU.z };
            all_dVs[k]       = { bDV.x, bDV.y, bDV.z };
            all_surfDists[k] = surfDist;
            all_surfPos[k]   = { bSurfPos.x, bSurfPos.y, bSurfPos.z };

            ProjectedGaussian pg;
            pg.originalIndex = k; pg.position = bSurfPos; pg.originalPos = p;
            pg.uv = bUV; pg.dU = bDU; pg.dV = bDV; pg.uvMaxDelta = bUVMaxDelta;
            float opc = std::max(1e-4f, std::min(1.f - 1e-4f, cpuOpacity[k]));
            pg.opacity = std::log(opc / (1.f - opc));
            pg.scale = { std::log(std::max(1e-9f, cpuScale[3*k])),
                         std::log(std::max(1e-9f, cpuScale[3*k+1])),
                         std::log(std::max(1e-9f, cpuScale[3*k+2])) };
            pg.rotation = { cpuRot[4*k], cpuRot[4*k+1], cpuRot[4*k+2], cpuRot[4*k+3] };
            _texGaussians.push_back(pg);
        }

        _expMapSolver.ClearSuppressedUVs();
        _lastAllUVs = all_uvs; _lastAllDUs = all_dUs; _lastAllDVs = all_dVs;
        _lastAllSurfDists = all_surfDists; _lastAllSurfPos = all_surfPos;
        // Save original positions (cpuPos is already restored to pre-snap state above)
        _lastAllOrigPos.resize(nGauss);
        for (int i = 0; i < nGauss; ++i) _lastAllOrigPos[i] = cpuPos[i];
        _expMapSolver.SetMainGaussiansForNextSave(_texGaussians);
        // Reset blend to 0: Gaussians start at their original positions
        _expMapSolver.ResetSurfaceBlend();
        // Upload UV data; snap target = original positions (blend=0, no movement yet)
        _gaussianView->setUVsAndTexture(all_uvs, all_dUs, all_dVs, all_surfDists, _lastAllOrigPos, _texPtr);
        // record hit position for toggle reuse
        _lastHitPos    = sibr::Vector3f(hitPos.x(), hitPos.y(), hitPos.z());
        _lastExpRadius = _expMapRadius;
        // suppress non-UV gaussians inside texture region to prevent occlusion
        // _gaussianView->suppressGaussiansInRegion(
        //     all_uvs,
        //     _lastHitPos,
        //     _expMapRadius,
        //     all_surfDists
        // );
    }

    GaussianView::Ptr               _gaussianView;
    SimplePointRenderer             _geoRenderer, _appRenderer;
    PointCloudRenderer              _pointCloudRenderer;
    MeshWireframeRenderer           _wireframeRenderer;
    ExpMapSolverSIBR                _expMapSolver;
    const sibr::Mesh* _mesh;
    sibr::InteractiveCameraHandler::Ptr _camHandler;
    sibr::Viewport                  _viewport;
    sibr::Vector3f _meshColor, _geoColor, _appColor;
    bool   _showMesh = false, _showGeo = false, _showApp = false;
    float  _pointSize = 1.0f, _expMapRadius = 0.5f;
    bool   _showGaussianOutlines = false;
    bool   _showTexture = true;
    GLuint _gaussOutlineVAO = 0, _gaussOutlineVBO = 0, _gaussOutlineShader = 0;
    std::vector<ProjectedGaussian>  _texGaussians;
    GLuint _liveSSBO = 0;
    sibr::Texture2DRGBA::Ptr        _texPtr;
    std::vector<sibr::Vector2f>     _lastAllUVs;
    std::vector<sibr::Vector3f>     _lastAllDUs, _lastAllDVs;
    std::vector<sibr::Vector3f>     _lastAllSurfPos;
    std::vector<sibr::Vector3f>     _lastAllOrigPos;
    std::vector<float>              _lastAllSurfDists;
    sibr::Vector3f                  _lastHitPos   = sibr::Vector3f(0,0,0);
    float                           _lastExpRadius = 0.5f;

    // 360° orbit recording
    bool             _recording360      = false;
    int              _recordFrame       = 0;
    int              _recordTotalFrames = 120;
    float            _orbitZoom         = 2.5f;
    float            _orbitElevation    = 0.0f;
    glm::vec3        _orbitCenter       = {0.f, 0.f, 0.f};
    float            _orbitRadius       = 1.0f;
    std::string      _videoDir;
    cv::VideoWriter  _videoWriter;
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