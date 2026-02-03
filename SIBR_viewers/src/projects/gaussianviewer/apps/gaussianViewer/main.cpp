/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact sibr@inria.fr and/or George.Drettakis@inria.fr
 */

#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>

#include <core/graphics/Window.hpp>
#include <core/view/MultiViewManager.hpp>
#include <core/system/String.hpp>
#include "projects/gaussianviewer/renderer/GaussianView.hpp" 
#include <core/renderer/DepthRenderer.hpp>
#include <core/raycaster/Raycaster.hpp>
#include <core/view/SceneDebugView.hpp>
#include <algorithm>
#include <boost/filesystem.hpp>
#include <regex>
#include <imgui/imgui_internal.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

// 引入 ExpMap Solver
#include "ExpMapSolverSIBR.h"

namespace fs = boost::filesystem;
using namespace sibr;

const std::string SHADER_DIR = "/mnt/d/SGGaussians/SGGaussians/SIBR_viewers/src/projects/gaussianviewer/apps/gaussianViewer/shaders/";

// Shader Helper Functions
std::string loadShaderFile(const std::string& filename) {
    std::string fullPath = SHADER_DIR + filename;
    std::ifstream file(fullPath);
    if (!file.is_open()) return "";
    std::stringstream buffer; buffer << file.rdbuf(); return buffer.str();
}

GLuint compileShader(const std::string& source, GLenum type) {
    if (source.empty()) return 0;
    GLuint shader = glCreateShader(type); const char* src = source.c_str();
    glShaderSource(shader, 1, &src, nullptr); glCompileShader(shader);
    return shader;
}

// ====================================================================================
// Simple Point Renderer Class
// ====================================================================================
class SimplePointRenderer {
public:
    void load(const std::string& path) {
        sibr::Mesh::Ptr mesh(new sibr::Mesh());
        if (!mesh->load(path)) return;
        _count = mesh->vertices().size(); if (_count == 0) return;
        
        // 儲存原始資料供 ExpMap 使用
        _rawData.clear();
        _rawData.reserve(_count * 3);
        for (const auto& v : mesh->vertices()) { 
            _rawData.push_back(v.x()); 
            _rawData.push_back(v.y()); 
            _rawData.push_back(v.z()); 
        }
        
        glGenVertexArrays(1, &_vao); glGenBuffers(1, &_vbo);
        glBindVertexArray(_vao); glBindBuffer(GL_ARRAY_BUFFER, _vbo);
        glBufferData(GL_ARRAY_BUFFER, _rawData.size() * sizeof(float), _rawData.data(), GL_STATIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0); glBindVertexArray(0); _loaded = true;
    }

    void loadFids(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "ERROR: Could not open PLY file: " << path << std::endl;
            return;
        }

        std::string line;
        long vertex_count = 0;
        int face_id_offset = -1; // in floats
        int property_count = 0; // in floats
        bool is_binary = false;

        // Header parsing
        while (std::getline(file, line)) {
            // Trim potential carriage return
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            
            std::stringstream ss(line);
            std::string token;
            ss >> token;

            if (token == "format") {
                ss >> token;
                if (token == "binary_little_endian") {
                    is_binary = true;
                }
            } else if (token == "element") {
                ss >> token;
                if (token == "vertex") {
                    ss >> vertex_count;
                }
            } else if (token == "property") {
                std::string type, name;
                ss >> type;
                if(type != "float") {
                    std::cerr << "ERROR: Non-float property '" << type << "' not supported by this PLY parser." << std::endl;
                    return;
                }
                ss >> name;
                if (name == "face_id") {
                    face_id_offset = property_count;
                }
                property_count++;

            } else if (token == "end_header") {
                break;
            }
        }

        if (!is_binary) {
            std::cerr << "ERROR: PLY file is not in binary format, which is required." << std::endl;
            return;
        }
        if (vertex_count == 0) {
            std::cerr << "ERROR: Vertex count is 0 in PLY file." << std::endl;
            return;
        }
        if (face_id_offset == -1) {
            std::cerr << "ERROR: 'face_id' property not found in PLY header." << std::endl;
            return;
        }

        // Data parsing
        _fids.clear();
        _fids.reserve(vertex_count);

        // The file stream is now correctly positioned at the start of the binary data.
        size_t vertex_byte_size = property_count * sizeof(float);
        std::vector<char> vertex_buffer(vertex_byte_size);
        float* float_buffer = reinterpret_cast<float*>(vertex_buffer.data());

        for (long i = 0; i < vertex_count; ++i) {
            file.read(vertex_buffer.data(), vertex_byte_size);
            if(!file) {
                std::cerr << "ERROR: Unexpected end of file while reading binary vertex data for vertex " << i << std::endl;
                return;
            }
            _fids.push_back(static_cast<int>(float_buffer[face_id_offset]));
        }
    }

    void draw() { if (!_loaded) return; glBindVertexArray(_vao); glDrawArrays(GL_POINTS, 0, (GLsizei)_count); glBindVertexArray(0); }
    bool isLoaded() const { return _loaded; }
    
    // 取得原始點雲資料介面
    const std::vector<float>& getRawData() const { return _rawData; }
    const std::vector<int>& getFids() const { return _fids; }

private:
    GLuint _vao = 0, _vbo = 0; size_t _count = 0; bool _loaded = false;
    std::vector<float> _rawData;
    std::vector<int> _fids;
};

// ====================================================================================
// MeshGaussianView 
// ====================================================================================
class MeshGaussianView : public sibr::ViewBase {
public:
    typedef std::shared_ptr<MeshGaussianView> Ptr;

    MeshGaussianView(GaussianView::Ptr gaussianView, const sibr::Mesh* mesh, 
                     const std::string& geoPath, const std::string& appPath,
                     sibr::InteractiveCameraHandler::Ptr camHandler)
        : _gaussianView(gaussianView), _mesh(mesh), _camHandler(camHandler), _progID(0) {
        
        // 1. Init Shader
        std::string vertCode = loadShaderFile("mesh.vert");
        std::string fragCode = loadShaderFile("mesh.frag");
        GLuint vs = compileShader(vertCode, GL_VERTEX_SHADER);
        GLuint fs = compileShader(fragCode, GL_FRAGMENT_SHADER);
        if (vs != 0 && fs != 0) {
            _progID = glCreateProgram(); glAttachShader(_progID, vs); glAttachShader(_progID, fs); glLinkProgram(_progID);
            glDeleteShader(vs); glDeleteShader(fs);
            _locMVP = glGetUniformLocation(_progID, "mvp"); _locColor = glGetUniformLocation(_progID, "color");
        }

        // 2. Load Points
        _geoRenderer.load(geoPath);
        _appRenderer.load(appPath);
        _appRenderer.loadFids(appPath);

        // 3. Init ExpMap Solver & 註冊雙點雲
        if (_mesh) {
            _expMapSolver.Init(_mesh);
            
            // 註冊 App 點雲 (青色)
            if (_appRenderer.isLoaded()) {
                std::cout << "[ExpMap] Registering App cloud: " << _appRenderer.getRawData().size() / 3 << " pts." << std::endl;
                _expMapSolver.RegisterAppPointCloud(_appRenderer.getRawData(), _appRenderer.getFids());
            }

            // [新增] 註冊 Geo 點雲 (紅色)
            if (_geoRenderer.isLoaded()) {
                std::cout << "[ExpMap] Registering Geo cloud: " << _geoRenderer.getRawData().size() / 3 << " pts." << std::endl;
                _expMapSolver.RegisterGeoPointCloud(_geoRenderer.getRawData());
            }
        }

        // 顏色設定 (網格設為黑色)
        _meshColor = sibr::Vector3f(0.0f, 0.0f, 0.0f); 
        _geoColor  = sibr::Vector3f(1.0f, 0.0f, 0.0f);
        _appColor  = sibr::Vector3f(0.0f, 0.0f, 1.0f);
    }

    ~MeshGaussianView() { if (_progID != 0) glDeleteProgram(_progID); }

    void onUpdate(Input& input, const Viewport& viewport) override {
        _gaussianView->onUpdate(input);

        // Check for texture apply trigger
        if (_expMapSolver.isApplyTextureTriggered()) {
            _expMapSolver.clearApplyTextureTrigger(); // Reset trigger immediately

            // [FIXED] Changed from sibr::Texture* to sibr::Texture2DRGBA*
            const sibr::Texture2DRGBA* tex = _expMapSolver.getTexture();
            const auto& uvMap_from_solver = _expMapSolver.getUVs();

            if (tex && !uvMap_from_solver.empty()) {
                std::cout << "[MeshGaussianView] Apply texture triggered." << std::endl;

                std::map<sibr::Vector3f, glm::vec2> pos_uv_map;
                for (const auto& pair : uvMap_from_solver) {
                    int point_id_from_solver = pair.first;
                    const glm::vec2& uv = pair.second;

                    // Get the 3D position corresponding to this solver ID
                    glm::vec3 pos_glm = _expMapSolver.getPos(point_id_from_solver);
                    sibr::Vector3f pos_sibr(pos_glm.x, pos_glm.y, pos_glm.z);
                    pos_uv_map[pos_sibr] = uv;
                }

                _gaussianView->applyTexture(*tex, pos_uv_map);
            } else {
                std::cerr << "[MeshGaussianView] ERROR: Texture or UV map is not available for application." << std::endl;
            }
        }

        // 右鍵點擊觸發計算
        if (input.mouseButton().isReleased(sibr::Mouse::Right) && !ImGui::GetIO().WantCaptureMouse) {
            PerformRaycast(input, viewport);
        }
    }

    void onRenderIBR(sibr::IRenderTarget& dst, const sibr::Camera& eye) override {
        _gaussianView->onRenderIBR(dst, eye);
        
        dst.bind();
        glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glEnable(GL_DEPTH_TEST);

        // 1. 畫原始的黑色 Mesh (受 _showMesh 控制)
        if (_progID != 0) {
            glUseProgram(_progID);
            glUniformMatrix4fv(_locMVP, 1, GL_FALSE, eye.viewproj().data());

            if (_mesh && _showMesh && _mesh->vertices().size() > 0) {
                glUniform3fv(_locColor, 1, &_meshColor[0]);
                glLineWidth(1.0f); 
                _mesh->render(true, false, sibr::Mesh::LineRenderMode);
            }
            glPointSize(_pointSize);
            if (_showGeo && _geoRenderer.isLoaded()) { glUniform3fv(_locColor, 1, &_geoColor[0]); _geoRenderer.draw(); }
            if (_showApp && _appRenderer.isLoaded()) { glUniform3fv(_locColor, 1, &_appColor[0]); _appRenderer.draw(); }
            glPointSize(1.0f);
            glUseProgram(0);
        }

        // 2. 畫選取範圍的高亮線框 (黃色) - 獨立繪製，不受 _showMesh 影響
        const auto& activeTris = _expMapSolver.GetActiveTriangles();
        if (_mesh && !activeTris.empty()) {
            glUseProgram(0);
            glMatrixMode(GL_PROJECTION);
            glLoadMatrixf(eye.proj().data());
            glMatrixMode(GL_MODELVIEW);
            glLoadMatrixf(eye.view().data());

            glLineWidth(3.0f); 
            glColor3f(1.0f, 1.0f, 0.0f); // 黃色

            glBegin(GL_LINES);
            const auto& verts = _mesh->vertices();
            for (const auto& t : activeTris) {
                const auto& v0 = verts[t.x()];
                const auto& v1 = verts[t.y()];
                const auto& v2 = verts[t.z()];

                glVertex3f(v0.x(), v0.y(), v0.z()); glVertex3f(v1.x(), v1.y(), v1.z());
                glVertex3f(v1.x(), v1.y(), v1.z()); glVertex3f(v2.x(), v2.y(), v2.z());
                glVertex3f(v2.x(), v2.y(), v2.z()); glVertex3f(v0.x(), v0.y(), v0.z());
            }
            glEnd();
            glLineWidth(1.0f);
        }

        // 3. 畫 3D 空間中的最短路徑 (紅色)
        const auto& dijkstraPath = _expMapSolver.GetDijkstraPath();
        if (!dijkstraPath.empty() && _mesh) {
            glUseProgram(0); 
            glMatrixMode(GL_PROJECTION);
            glLoadMatrixf(eye.proj().data());
            glMatrixMode(GL_MODELVIEW);
            glLoadMatrixf(eye.view().data());

            glLineWidth(5.0f); 
            glColor3f(1.0f, 0.0f, 0.0f); // Red color

            glDisable(GL_DEPTH_TEST); // Always on top
            glBegin(GL_LINE_STRIP); 
            for (int nodeId : dijkstraPath) {
                glm::vec3 pos = _expMapSolver.getPos(nodeId); 
                glVertex3f(pos.x, pos.y, pos.z);
            }
            glEnd();
            glEnable(GL_DEPTH_TEST); 
            glLineWidth(1.0f); 
        }

        dst.unbind();
    }

    void onGUI() override {
        _gaussianView->onGUI();
        
        if (ImGui::Begin("Layers")) {
            ImGui::Checkbox("Mesh", &_showMesh);
            ImGui::Checkbox("Geo Points", &_showGeo);
            ImGui::Checkbox("App Points", &_showApp);
            ImGui::SliderFloat("Pt Size", &_pointSize, 1.0f, 4.0f);
            ImGui::Separator();
            ImGui::Text("ExpMap Tool");
            ImGui::SliderFloat("Radius", &_expMapRadius, 0.01f, 1.2f);
            ImGui::TextColored(ImVec4(1,1,0,1), "Right-click on mesh to flatten!");
        }
        ImGui::End();

        // 呼叫 Solver 的 UI 繪製 (包含 2D UV 視圖)
        _expMapSolver.RenderUI();
    }

private:
    void PerformRaycast(const sibr::Input& input, const sibr::Viewport& viewport) {
        const sibr::Camera& cam = _camHandler->getCamera();
        sibr::Vector2f mousePos = input.mousePosition().cast<float>();

        float x = 2.0f * (mousePos.x() / viewport.finalWidth()) - 1.0f;
        float y = 1.0f - 2.0f * (mousePos.y() / viewport.finalHeight());
        glm::vec4 rayClip(x, y, -1.0f, 1.0f);
        
        glm::mat4 proj = glm::make_mat4(cam.proj().data());
        glm::mat4 view = glm::make_mat4(cam.view().data());
        glm::vec4 rayEye = glm::inverse(proj) * rayClip;
        rayEye = glm::vec4(rayEye.x, rayEye.y, -1.0f, 0.0f);
        glm::vec3 rayDir = glm::normalize(glm::vec3(glm::inverse(view) * rayEye));
        glm::vec3 rayOrigin = glm::make_vec3(cam.position().data());

        if (!_mesh) {
            std::cout << "[DEBUG] PerformRaycast: _mesh is null. Cannot raycast." << std::endl;
            return;
        }
        std::cout << "[DEBUG] PerformRaycast: _mesh is valid. Vertices: " << _mesh->vertices().size() << ", Triangles: " << _mesh->triangles().size() << std::endl;

        float minT = 1e9f;
        bool hit = false;
        int hitTri = -1;
        
        const auto& tris = _mesh->triangles();
        const auto& verts = _mesh->vertices();

        for (int i = 0; i < (int)tris.size(); ++i) {
            auto toVec3 = [](const sibr::Vector3f& v){ return glm::vec3(v.x(), v.y(), v.z()); };
            glm::vec3 v0 = toVec3(verts[tris[i].x()]);
            glm::vec3 v1 = toVec3(verts[tris[i].y()]);
            glm::vec3 v2 = toVec3(verts[tris[i].z()]);

            glm::vec3 e1 = v1 - v0;
            glm::vec3 e2 = v2 - v0;
            glm::vec3 h = glm::cross(rayDir, e2);
            float a = glm::dot(e1, h);
            if (std::abs(a) < 1e-6) continue;
            float f = 1.0f / a;
            glm::vec3 s = rayOrigin - v0;
            float u = f * glm::dot(s, h);
            if (u < 0.0f || u > 1.0f) continue;
            glm::vec3 q = glm::cross(s, e1);
            float v = f * glm::dot(rayDir, q);
            if (v < 0.0f || u + v > 1.0f) continue;
            float t = f * glm::dot(e2, q);

            if (t > 1e-4 && t < minT) {
                minT = t;
                hit = true;
                hitTri = i;
            }
        }

        if (hit) {
            glm::vec3 hitPoint = rayOrigin + rayDir * minT;
            sibr::Vector3f hitNormal(0,1,0);
            if (_mesh->normals().size() > 0) {
                hitNormal = _mesh->normals()[tris[hitTri].x()];
            }
            std::cout << "[DEBUG] PerformRaycast: Hit detected. HitTri: " << hitTri << ", HitPoint: (" << hitPoint.x << ", " << hitPoint.y << ", " << hitPoint.z << ")" << std::endl;
            std::cout << "[DEBUG] PerformRaycast: Mesh has normals: " << (_mesh->normals().size() > 0 ? "Yes" : "No") << std::endl;
            _expMapSolver.Compute(sibr::Vector3f(hitPoint.x, hitPoint.y, hitPoint.z), hitNormal, _expMapRadius);
        } else {
            std::cout << "[DEBUG] PerformRaycast: No hit detected." << std::endl;
        }
    }

    GaussianView::Ptr _gaussianView;
    const sibr::Mesh* _mesh; 
    SimplePointRenderer _geoRenderer;
    SimplePointRenderer _appRenderer;
    
    ExpMapSolverSIBR _expMapSolver;
    sibr::InteractiveCameraHandler::Ptr _camHandler;

    GLuint _progID; GLint _locMVP, _locColor;
    sibr::Vector3f _meshColor, _geoColor, _appColor;
    bool _showMesh = true, _showGeo = true, _showApp = true;
    float _pointSize = 1.0f;
    float _expMapRadius = 0.5f; 
};

// ====================================================================================
// Main Helper Functions
// ====================================================================================
std::string findLargestNumberedSubdirectory(const std::string& directoryPath) {
    fs::path dirPath(directoryPath);
    if (!fs::exists(dirPath) || !fs::is_directory(dirPath)) return "";
    std::regex regexPattern(R"_(iteration_(\d+))_");
    std::string largestSubdirectory; int largestNumber = -1;
    for (const auto& entry : fs::directory_iterator(dirPath)) {
        if (fs::is_directory(entry)) {
            std::string subdirectory = entry.path().filename().string();
            std::smatch match;
            if (std::regex_match(subdirectory, match, regexPattern)) {
                int number = std::stoi(match[1]);
                if (number > largestNumber) { largestNumber = number; largestSubdirectory = subdirectory; }
            }
        }
    }
    return largestSubdirectory;
}

std::pair<int, int> findArg(const std::string& line, const std::string& name) {
    int start = line.find(name, 0); start = line.find("=", start) + 1;
    int end = line.find_first_of(",)", start); return std::make_pair(start, end);
}
static void* User_ReadOpen(ImGuiContext*, ImGuiSettingsHandler*, const char* name) { return (void*)0x1; }
static void User_ReadLine(ImGuiContext*, ImGuiSettingsHandler* handler, void*, const char* line) {
    int i; if (sscanf(line, "DontShow=%d", &i) == 1) if (i) { *((bool*)handler->UserData) = true; return; }
    *((bool*)handler->UserData) = false;
}
static void User_WriteAll(ImGuiContext* imgui_ctx, ImGuiSettingsHandler* handler, ImGuiTextBuffer* buf) {
    buf->reserve(buf->size() + 96); buf->appendf("[UserData][UserData]\nDontShow=%d\n", *((bool*)handler->UserData) ? 1 : 0); buf->appendf("\n");
}

int main(int ac, char** av) 
{
    CommandLineArgs::parseMainArgs(ac, av);
    GaussianAppArgs myArgs;
    myArgs.displayHelpIfRequired();
    if(!myArgs.modelPath.isInit() && myArgs.modelPathShort.isInit()) myArgs.modelPath = myArgs.modelPathShort.get();
    if(!myArgs.dataset_path.isInit() && myArgs.pathShort.isInit()) myArgs.dataset_path = myArgs.pathShort.get();

    int device = myArgs.device;
    sibr::Window window("sibr_3Dgaussian", sibr::Vector2i(50, 50), myArgs);

    bool messageRead = false;
    ImGuiSettingsHandler ini_handler; ini_handler.TypeName = "UserData"; ini_handler.UserData = &messageRead;
    ini_handler.TypeHash = ImHash("UserData", 0, 0); ini_handler.ReadOpenFn = User_ReadOpen;
    ini_handler.ReadLineFn = User_ReadLine; ini_handler.WriteAllFn = User_WriteAll;
    ImGui::GetCurrentContext()->SettingsHandlers.push_back(ini_handler);
    window.loadSettings();

    std::string cfgLine; std::ifstream cfgFile(myArgs.modelPath.get() + "/cfg_args");
    if (!cfgFile.good()) SIBR_ERR << "Could not find config file 'cfg_args'" << std::endl;
    std::getline(cfgFile, cfgLine);
    if (!myArgs.dataset_path.isInit()) {
        auto rng = findArg(cfgLine, "source_path");
        myArgs.dataset_path = cfgLine.substr(rng.first + 1, rng.second - rng.first - 2);
    }
    auto rng = findArg(cfgLine, "sh_degree"); int sh_degree = std::stoi(cfgLine.substr(rng.first, rng.second - rng.first));
    rng = findArg(cfgLine, "white_background"); bool white_background = cfgLine.substr(rng.first, rng.second - rng.first).find("True") != -1;

    BasicIBRScene::SceneOptions myOpts; myOpts.renderTargets = myArgs.loadImages; myOpts.mesh = true; myOpts.images = myArgs.loadImages; myOpts.cameras = true; myOpts.texture = false;
    BasicIBRScene::Ptr scene;
    try { scene.reset(new BasicIBRScene(myArgs, myOpts)); }
    catch (...) { myArgs.dataset_path = myArgs.modelPath.get(); scene.reset(new BasicIBRScene(myArgs, myOpts)); }

    // 修改: 優先嘗試從 geo_point_cloud.ply 載入帶有面資訊的網格
    sibr::Mesh::Ptr geoMeshWithFaces(new sibr::Mesh());
    const sibr::Mesh* meshToRender = nullptr; // 將 meshToRender 初始化為 nullptr

    // 組裝 geo_point_cloud.ply 的完整路徑
    std::string plyfile = myArgs.modelPath.get(); 
    if (plyfile.back() != '/') plyfile += "/"; 
    plyfile += "point_cloud";

    std::string iterDir; 
    if (!myArgs.iteration.isInit()) iterDir = findLargestNumberedSubdirectory(plyfile); 
    else iterDir = "iteration_" + myArgs.iteration.get();
    
    std::string plyDir = plyfile + "/" + iterDir + "/";
    std::string geoPlyPathForMesh = plyDir + "geo_point_cloud.ply";

    if (geoMeshWithFaces->load(geoPlyPathForMesh)) {
        if (geoMeshWithFaces->triangles().size() > 0) {
            std::cout << "[INFO] Successfully loaded geo_point_cloud.ply with " << geoMeshWithFaces->triangles().size() << " faces as mesh." << std::endl;
            meshToRender = geoMeshWithFaces.get();
        } else {
            std::cout << "[INFO] geo_point_cloud.ply loaded, but no faces found. Falling back to mesh.obj/mesh.ply." << std::endl;
        }
    } else {
        std::cout << "[INFO] Could not load geo_point_cloud.ply as mesh. Falling back to mesh.obj/mesh.ply." << std::endl;
    }

    // 如果 geo_point_cloud.ply 沒有成功載入為網格，則執行原有的載入邏輯
    if (!meshToRender) {
        sibr::Mesh::Ptr manualMesh(new sibr::Mesh());
        if (scene->proxies()->proxy().vertices().empty()) {
            std::string objPath = myArgs.dataset_path.get() + "/mesh.obj";
            if (manualMesh->load(objPath)) {
                meshToRender = manualMesh.get();
                std::cout << "[INFO] Loaded mesh.obj as mesh." << std::endl;
            }
            else { 
                std::string plyPath = myArgs.dataset_path.get() + "/mesh.ply"; 
                if (manualMesh->load(plyPath)) {
                    meshToRender = manualMesh.get(); 
                    std::cout << "[INFO] Loaded mesh.ply as mesh." << std::endl;
                } else {
                    std::cout << "[WARNING] No mesh.obj or mesh.ply found. ExpMap functions may not work." << std::endl;
                }
            }
        } else {
            meshToRender = &scene->proxies()->proxy();
            std::cout << "[INFO] Loaded scene proxy as mesh." << std::endl;
        }
    }
    
    // The variables plyfile, iterDir, and plyDir are already defined above.
    // We reuse them here for the remaining point cloud paths.
    std::string finalPlyPath = plyfile + "/" + iterDir + "/point_cloud.ply";
    std::string geoPath = plyDir + "geo_point_cloud.ply";
    std::string appPath = plyDir + "app_point_cloud.ply";

    uint scene_width = scene->cameras()->inputCameras()[0]->w(); uint scene_height = scene->cameras()->inputCameras()[0]->h();
    float scene_aspect_ratio = scene_width * 1.0f / scene_height;
    uint rendering_width = myArgs.rendering_size.get()[0]; uint rendering_height = myArgs.rendering_size.get()[1];
    rendering_width = (rendering_width <= 0) ? std::min(1200U, scene_width) : rendering_width;
    rendering_height = (rendering_height <= 0) ? std::min(1200U, scene_width) / scene_aspect_ratio : rendering_height;
    Vector2u usedResolution(rendering_width, rendering_height);

    GaussianView::Ptr gaussianView(new GaussianView(scene, usedResolution.x(), usedResolution.y(), finalPlyPath.c_str(), &messageRead, sh_degree, white_background, !myArgs.noInterop, device));
    sibr::InteractiveCameraHandler::Ptr generalCamera(new InteractiveCameraHandler());
    generalCamera->setup(scene->cameras()->inputCameras(), Viewport(0, 0, (float)usedResolution.x(), (float)usedResolution.y()), nullptr);

    MeshGaussianView::Ptr meshGaussianView(new MeshGaussianView(gaussianView, meshToRender, geoPath, appPath, generalCamera));

    MultiViewManager multiViewManager(window, false);
    if (myArgs.rendering_mode == 1) multiViewManager.renderingMode(IRenderingMode::Ptr(new StereoAnaglyphRdrMode()));
    
    multiViewManager.addIBRSubView("Point view", meshGaussianView, usedResolution, ImGuiWindowFlags_ResizeFromAnySide | ImGuiWindowFlags_NoBringToFrontOnFocus);
    multiViewManager.addCameraForView("Point view", generalCamera);

    const std::shared_ptr<sibr::SceneDebugView> topView(new sibr::SceneDebugView(scene, generalCamera, myArgs, myArgs.imagesPath.get()));
    multiViewManager.addSubView("Top view", topView, usedResolution); topView->active(false);
    generalCamera->getCameraRecorder().setViewPath(gaussianView, myArgs.dataset_path.get());

    if (myArgs.pathFile.get() !=  "" ) {
        generalCamera->getCameraRecorder().loadPath(myArgs.pathFile.get(), usedResolution.x(), usedResolution.y());
        generalCamera->getCameraRecorder().recordOfflinePath(myArgs.outPath, multiViewManager.getIBRSubView("Point view"), "");
        if( !myArgs.noExit ) exit(0);
    }

    while (window.isOpened()) {
        sibr::Input::poll(); window.makeContextCurrent();
        if (sibr::Input::global().key().isPressed(sibr::Key::Escape)) window.close();
        multiViewManager.onUpdate(sibr::Input::global());
        multiViewManager.onRender(window);
        window.swapBuffer();
    }
    return EXIT_SUCCESS;
}