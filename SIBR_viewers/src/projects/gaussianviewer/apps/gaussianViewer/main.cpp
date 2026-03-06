#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <filesystem>
#include <regex>

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

#include "ExpMapSolverSIBR.h"  // also pulls in texture.h

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
        // Load point clouds
        if (std_fs::exists(geoPath)) {
            _geoRenderer.load(geoPath);
            _geoRenderer.loadGaussianProps(geoPath);
        }
        if (std_fs::exists(appPath)) {
            _appRenderer.load(appPath);
            _appRenderer.loadFids(appPath);
            _appRenderer.loadGaussianProps(appPath);
        }

        // Initialise ExpMap solver
        if (_mesh) {
            _expMapSolver.Init(_mesh);
            if (_geoRenderer.isLoaded())
                _expMapSolver.RegisterGeoPointCloud(_geoRenderer.getRawData(),
                                                    _geoRenderer.getGaussianProps());
            if (_appRenderer.isLoaded())
                _expMapSolver.RegisterAppPointCloud(_appRenderer.getRawData(),
                                                    _appRenderer.getFids(),
                                                    _appRenderer.getGaussianProps());

            // NEW: Upload the entire mesh to GPU VRAM for fast rendering
            _wireframeRenderer.uploadMesh(_mesh);
        }

        // Default overlay colours
        _meshColor = sibr::Vector3f(0.0f, 0.6f, 0.0f);
        _geoColor  = sibr::Vector3f(1.0f, 0.2f, 0.0f);
        _appColor  = sibr::Vector3f(0.0f, 0.4f, 1.0f);
    }

    void onRenderIBR(sibr::IRenderTarget& dst, const sibr::Camera& eye) override {
        if (!_gaussianView) return;

        _gaussianView->onRenderIBR(dst, eye);

        _viewport = Viewport(0, 0, (float)dst.w(), (float)dst.h());
        dst.bind();
        glViewport(0, 0, dst.w(), dst.h());

        const glm::mat4 mvp = glm::make_mat4(eye.viewproj().data());

        // 1. Mesh wireframe (black = regular tris, yellow = UV-active tris)
        _wireframeRenderer.render(_mesh, _expMapSolver.GetActiveTriIndices(), mvp, _showMesh);

        // 2. Texture display — texture_gs 風格
        //    每個 Gaussian 以 splat 形式渲染，顏色從 ExpMap UV sample texture
        const glm::mat4 view = glm::make_mat4(eye.view().data());
        const glm::mat4 proj = glm::make_mat4(eye.proj().data());
        const float vpW = (float)dst.w();
        const float vpH = (float)dst.h();

        _gsRenderer.renderAll(
            _expMapSolver.GetAllSlots(),
            view, proj, vpW, vpH
        );
        _gsRenderer.render(
            _expMapSolver.GetProjectedGeoPoints(),
            _expMapSolver.GetProjectedAppPoints(),
            _expMapSolver.GetTextureLoader().getTexture(),
            view, proj, vpW, vpH
        );

        // 3. Geo / App point clouds
        _pointCloudRenderer.render(
            mvp, _pointSize,
            _geoRenderer, _geoColor, _showGeo,
            _appRenderer,  _appColor, _showApp
        );

        // 4. Dijkstra path (legacy immediate-mode GL)
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
        ImGui::Checkbox("Show Mesh",              &_showMesh);
        ImGui::Checkbox("Show UV Wireframe",      &_wireframeRenderer._showYellowWireframe);
        ImGui::Checkbox("Show Geo Points", &_showGeo);
        ImGui::Checkbox("Show App Points", &_showApp);
        _gsRenderer.renderGUI();
        ImGui::SliderFloat("Point Size",    &_pointSize,    1.0f, 3.0f);
        ImGui::SliderFloat("ExpMap Radius", &_expMapRadius, 0.05f, 2.0f);
        if (ImGui::Button("Clear ExpMap")) _expMapSolver.ClearRaycastState();

        ImGui::Separator();
        // Show saved slot count
        const auto& slots = _expMapSolver.GetAllSlots();
        if (!slots.empty()) {
            ImGui::TextColored(ImVec4(0.3f, 1.f, 0.5f, 1.f),
                "Saved Slots: %lu (manage in ExpMap UV window)", slots.size());
        }
        ImGui::Separator();
        const auto& activeTris = _expMapSolver.GetActiveTris();
        if (!activeTris.empty()) {
            ImGui::TextColored(ImVec4(1.f, 1.f, 0.f, 1.f),
                "UV Region: %lu triangles (YELLOW)", activeTris.size());
            if (_expMapSolver.GetTextureLoader().getTexture()) {
                ImGui::TextColored(ImVec4(0.5f, 1.f, 1.f, 1.f),
                    "Texture loaded: %s", _expMapSolver.GetTextureLoader().getPath().c_str());
            } else {
                ImGui::TextColored(ImVec4(1.f, 0.5f, 0.5f, 1.f),
                    "No texture loaded (use File > Load Background)");
            }
        } else {
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.f), "Right-click to select UV region");
        }
        ImGui::End();

        _expMapSolver.RenderUI();
    }

private:
    void performRaycast(const sibr::Input& input) {
        if (!_mesh || !_camHandler) return;

        const auto& cam = _camHandler->getCamera();
        sibr::Vector2f mousePos(
            static_cast<float>(input.mousePosition().x()),
            static_cast<float>(input.mousePosition().y())
        );

        // Screen → NDC
        float ndcX = (2.f * mousePos.x()) / _viewport.finalWidth()  - 1.f;
        float ndcY = 1.f - (2.f * mousePos.y()) / _viewport.finalHeight();

        // NDC → world ray
        sibr::Matrix4f invVP = cam.viewproj().inverse();
        sibr::Vector4f nearPt = invVP * sibr::Vector4f(ndcX, ndcY, -1.f, 1.f);
        sibr::Vector4f farPt  = invVP * sibr::Vector4f(ndcX, ndcY,  1.f, 1.f);
        nearPt /= nearPt.w();
        farPt  /= farPt.w();

        sibr::Vector3f rayOrigin = cam.position();
        sibr::Vector3f rayDir(farPt.x() - nearPt.x(),
                              farPt.y() - nearPt.y(),
                              farPt.z() - nearPt.z());
        rayDir.normalize();

        const auto& verts     = _mesh->vertices();
        const auto& triangles = _mesh->triangles();

        float         minDist  = 1e9f;
        int           hitTriID = -1;
        sibr::Vector3f hitPos;

        // Möller–Trumbore for each triangle
        for (size_t t = 0; t < triangles.size(); ++t) {
            const auto& tri = triangles[t];
            const sibr::Vector3f& v0 = verts[tri[0]];
            const sibr::Vector3f& v1 = verts[tri[1]];
            const sibr::Vector3f& v2 = verts[tri[2]];

            sibr::Vector3f edge1 = v1 - v0;
            sibr::Vector3f edge2 = v2 - v0;
            sibr::Vector3f h     = rayDir.cross(edge2);
            float          a     = edge1.dot(h);

            if (a > -1e-6f && a < 1e-6f) continue; // parallel

            float          f = 1.f / a;
            sibr::Vector3f s = rayOrigin - v0;
            float          u = f * s.dot(h);
            if (u < 0.f || u > 1.f) continue;

            sibr::Vector3f q    = s.cross(edge1);
            float          v    = f * rayDir.dot(q);
            if (v < 0.f || u + v > 1.f) continue;

            float tDist = f * edge2.dot(q);
            if (tDist > 1e-6f && tDist < minDist) {
                minDist  = tDist;
                hitTriID = (int)t;
                hitPos   = rayOrigin + rayDir * tDist;
            }
        }

        if (hitTriID >= 0) {
            std::cout << "[INFO] Raycast hit triangle " << hitTriID
                      << " at distance " << minDist << std::endl;
            _expMapSolver.OnRaycastHit(hitPos, _expMapRadius, hitTriID);
        } else {
            std::cout << "[INFO] Raycast missed all triangles" << std::endl;
        }
    }

    GaussianView::Ptr               _gaussianView;
    SimplePointRenderer             _geoRenderer;
    SimplePointRenderer             _appRenderer;
    PointCloudRenderer              _pointCloudRenderer;
    MeshWireframeRenderer           _wireframeRenderer;
    GaussianSplatRenderer           _gsRenderer;
    ExpMapSolverSIBR                _expMapSolver;

    const sibr::Mesh*                   _mesh;
    sibr::InteractiveCameraHandler::Ptr _camHandler;
    sibr::Viewport                      _viewport;

    sibr::Vector3f _meshColor, _geoColor, _appColor;
    bool   _showMesh    = true;
    bool   _showGeo     = true;
    bool   _showApp     = true;
    float  _pointSize   = 1.0f;
    float  _expMapRadius = 0.5f;
};


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
            if (n > bestN) { bestN = n; best = name; }
        }
    }
    return best;
}

// Return [start, end) of the value for a named argument in a Python-style arg string.
static std::pair<int,int> findArg(const std::string& line, const std::string& name) {
    size_t start = line.find(name, 0);
    start = line.find("=", start) + 1;
    size_t end = line.find_first_of(",)", start);
    return { (int)start, (int)end };
}


static void* User_ReadOpen(ImGuiContext*, ImGuiSettingsHandler*, const char*) { return (void*)0x1; }
static void  User_ReadLine(ImGuiContext*, ImGuiSettingsHandler* h, void*, const char* line) {
    int i;
    if (sscanf_s(line, "DontShow=%d", &i) == 1) { *((bool*)h->UserData) = (i != 0); }
}
static void  User_WriteAll(ImGuiContext*, ImGuiSettingsHandler* h, ImGuiTextBuffer* buf) {
    buf->reserve(buf->size() + 96);
    buf->appendf("[UserData][UserData]\nDontShow=%d\n\n", *((bool*)h->UserData) ? 1 : 0);
}


// ============================================================================
// main
// ============================================================================
int main(int ac, char** av) {
    CommandLineArgs::parseMainArgs(ac, av);
    GaussianAppArgs myArgs;
    myArgs.displayHelpIfRequired();

    if (!myArgs.modelPath.isInit()    && myArgs.modelPathShort.isInit()) myArgs.modelPath    = myArgs.modelPathShort.get();
    if (!myArgs.dataset_path.isInit() && myArgs.pathShort.isInit())      myArgs.dataset_path = myArgs.pathShort.get();

    sibr::Window window("sibr_3Dgaussian", sibr::Vector2i(50, 50), myArgs);

    // Register ImGui settings handler for the "DontShow" flag
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

    // Parse cfg_args
    std::string cfgLine;
    std::ifstream cfgFile(myArgs.modelPath.get() + "/cfg_args");
    if (!cfgFile.good()) SIBR_ERR << "Could not find config file 'cfg_args' at: " << myArgs.modelPath.get();
    std::getline(cfgFile, cfgLine);

    if (!myArgs.dataset_path.isInit()) {
        auto rng = findArg(cfgLine, "source_path");
        myArgs.dataset_path = cfgLine.substr(rng.first + 1, rng.second - rng.first - 2);
    }
    auto rng = findArg(cfgLine, "sh_degree");
    int  sh_degree = std::stoi(cfgLine.substr(rng.first, rng.second - rng.first));
    rng = findArg(cfgLine, "white_background");
    bool white_background = cfgLine.substr(rng.first, rng.second - rng.first).find("True") != std::string::npos;

    // Build scene
    BasicIBRScene::SceneOptions myOpts;
    myOpts.renderTargets = myArgs.loadImages;
    myOpts.mesh    = true;
    myOpts.images  = myArgs.loadImages;
    myOpts.cameras = true;
    myOpts.texture = false;

    BasicIBRScene::Ptr scene;
    try { scene.reset(new BasicIBRScene(myArgs, myOpts)); }
    catch (...) { myArgs.dataset_path = myArgs.modelPath.get(); scene.reset(new BasicIBRScene(myArgs, myOpts)); }

    // Resolve PLY paths
    std::string plyBase = myArgs.modelPath.get();
    if (plyBase.back() != '/' && plyBase.back() != '\\') plyBase += "/";
    plyBase += "point_cloud";

    std::string iterDir;
    if (!myArgs.iteration.isInit()) {
        iterDir = findLargestNumberedSubdirectory(plyBase);
        std::cout << "Auto-detected iteration directory: " << iterDir << std::endl;
    } else {
        iterDir = "iteration_" + myArgs.iteration.get();
    }

    const std::string plyDir       = plyBase + "/" + iterDir + "/";
    const std::string finalPlyPath = plyDir + "point_cloud.ply";
    const std::string geoPath      = plyDir + "geo_point_cloud.ply";
    const std::string appPath      = plyDir + "app_point_cloud.ply";

    // Choose mesh: prefer geo PLY, fall back to scene proxy
    sibr::Mesh::Ptr      geoMesh(new sibr::Mesh());
    const sibr::Mesh*    meshToRender = nullptr;
    if (std_fs::exists(geoPath) && geoMesh->load(geoPath) && !geoMesh->triangles().empty()) {
        meshToRender = geoMesh.get();
        std::cout << "Using geo mesh: " << geoMesh->triangles().size()
                  << " faces, " << geoMesh->vertices().size() << " verts" << std::endl;
    } else if (!scene->proxies()->proxy().vertices().empty()) {
        meshToRender = &scene->proxies()->proxy();
        std::cout << "Fallback: using proxy mesh from scene" << std::endl;
    }

    // Compute rendering resolution
    uint  scene_w  = scene->cameras()->inputCameras()[0]->w();
    uint  scene_h  = scene->cameras()->inputCameras()[0]->h();
    float aspect   = scene_w * 1.f / scene_h;
    uint  rw       = myArgs.rendering_size.get()[0];
    uint  rh       = myArgs.rendering_size.get()[1];
    rw = (rw <= 0) ? std::min(1200U, scene_w) : rw;
    rh = (rh <= 0) ? (uint)(std::min(1200U, scene_w) / aspect) : rh;
    Vector2u usedRes(rw, rh);

    // Create views
    GaussianView::Ptr gaussianView(new GaussianView(
        scene, usedRes.x(), usedRes.y(),
        finalPlyPath.c_str(), &messageRead, sh_degree, white_background,
        !myArgs.noInterop, myArgs.device));

    sibr::InteractiveCameraHandler::Ptr generalCamera(new InteractiveCameraHandler());
    generalCamera->setup(scene->cameras()->inputCameras(),
                         Viewport(0, 0, (float)usedRes.x(), (float)usedRes.y()), nullptr);

    MeshGaussianView::Ptr meshView(new MeshGaussianView(
        gaussianView, meshToRender, geoPath, appPath, generalCamera));

    // Build multi-view layout
    MultiViewManager multiViewManager(window, false);
    if (myArgs.rendering_mode == 1)
        multiViewManager.renderingMode(IRenderingMode::Ptr(new StereoAnaglyphRdrMode()));

    multiViewManager.addIBRSubView("Point view", meshView, usedRes,
        ImGuiWindowFlags_ResizeFromAnySide | ImGuiWindowFlags_NoBringToFrontOnFocus);
    multiViewManager.addCameraForView("Point view", generalCamera);

    const auto topView = std::make_shared<sibr::SceneDebugView>(
        scene, generalCamera, myArgs, myArgs.imagesPath.get());
    multiViewManager.addSubView("Top view", topView, usedRes);
    topView->active(false);

    // Offline path recording
    generalCamera->getCameraRecorder().setViewPath(gaussianView, myArgs.dataset_path.get());
    if (myArgs.pathFile.get() != "") {
        generalCamera->getCameraRecorder().loadPath(myArgs.pathFile.get(), usedRes.x(), usedRes.y());
        generalCamera->getCameraRecorder().recordOfflinePath(
            myArgs.outPath, multiViewManager.getIBRSubView("Point view"), "");
        if (!myArgs.noExit) exit(0);
    }

    // Render loop
    while (window.isOpened()) {
        sibr::Input::poll();
        window.makeContextCurrent();
        if (sibr::Input::global().key().isPressed(sibr::Key::Escape)) window.close();
        multiViewManager.onUpdate(sibr::Input::global());
        multiViewManager.onRender(window);
        window.swapBuffer();
    }

    return EXIT_SUCCESS;
}