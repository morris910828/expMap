#pragma once

// Standard library
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <set>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

// SIBR / Third-party
#include <core/graphics/Image.hpp>
#include <core/graphics/Texture.hpp>
#include <core/graphics/Mesh.hpp>
#include <boost/filesystem.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <GL/glew.h>
#include <imgui/imgui.h>

namespace fs = boost::filesystem;

// ============================================================================
// 每個 Gaussian 的物理屬性（從 PLY 讀取，數值為 raw，尚未套 activation）
// ============================================================================
struct GaussianProps {
    float opacity;    // raw，渲染時套 sigmoid
    float scale[3];   // raw，渲染時套 exp
    float rot[4];     // quaternion (w, x, y, z)，已正規化
};

// ============================================================================
// 投影到 UV 空間的 Gaussian 點（ExpMap 結果 + 渲染所需屬性）
// 定義在此處，供 texture.h 內的 GaussianSplatRenderer 使用
// ExpMapSolverSIBR.h 請移除其中的舊版定義，改由此處繼承
// ============================================================================
struct ProjectedGaussian {
    int       originalIndex;  // 對應來源點雲的原始 index
    glm::vec2 uv;             // ExpMap 解算出的 texture UV 座標

    // 渲染所需欄位（由 ProjectAndInsertClouds 填入）
    glm::vec3 position;       // 世界座標原始位置
    float     opacity;        // raw（before sigmoid）
    glm::vec3 scale;          // raw（before exp）
    glm::vec4 rotation;       // quaternion (w, x, y, z)
};

namespace detail {
inline const std::string& meshVertSrc() {
    static const std::string s = R"(
        #version 330 core
        layout(location = 0) in vec3 aPos;
        layout(location = 1) in vec3 aNormal;
        uniform mat4 uMVP;
        out vec3 vNormal;
        void main() {
            gl_Position = uMVP * vec4(aPos, 1.0);
            vNormal = aNormal;
        }
    )";
    return s;
}

inline const std::string& meshFragSrc() {
    static const std::string s = R"(
        #version 330 core
        in vec3 vNormal;
        out vec4 FragColor;
        uniform vec3  uColor;
        uniform bool  uUseLighting;
        void main() {
            if (uUseLighting) {
                vec3 lightDir = normalize(vec3(0.5, 0.5, 1.0));
                float diff = max(dot(normalize(vNormal), lightDir), 0.3);
                FragColor = vec4(uColor * diff, 1.0);
            } else {
                FragColor = vec4(uColor, 1.0);
            }
        }
    )";
    return s;
}

// Compile one shader stage; writes errors to stderr. Returns 0 on failure.
inline GLuint compileGLShader(const std::string& src, GLenum type) {
    if (src.empty()) return 0;
    GLuint sh = glCreateShader(type);
    const char* s = src.c_str();
    glShaderSource(sh, 1, &s, nullptr);
    glCompileShader(sh);
    GLint ok; glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        GLchar log[512]; glGetShaderInfoLog(sh, 512, nullptr, log);
        std::cerr << "[GL] Shader compile error: " << log << "\n";
    }
    return sh;
}

// Link vs + fs into a program; deletes both shader objects.
inline GLuint linkProgram(GLuint vs, GLuint fs) {
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    glDeleteShader(vs);
    glDeleteShader(fs);
    return prog;
}

}


class TextureLoader {
public:
    TextureLoader() = default;

    bool LoadImage(const std::string& path) {
        sibr::ImageRGBA image;
        if (!image.load(path)) return false;
        _texture.reset(new sibr::Texture2DRGBA(image, SIBR_GPU_LINEAR_SAMPLING));
        _path = path;
        return true;
    }

    // Return all image paths under `directory` with a supported extension.
    std::vector<std::string> scanForImages(const std::string& directory) const {
        std::vector<std::string> result;
        fs::path dirPath(directory);
        if (!fs::exists(dirPath) || !fs::is_directory(dirPath)) return result;

        static const std::vector<std::string> kExts = {".png", ".jpg", ".jpeg", ".bmp", ".tga"};
        for (const auto& entry : fs::directory_iterator(dirPath)) {
            if (!fs::is_regular_file(entry.status())) continue;
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (std::find(kExts.begin(), kExts.end(), ext) != kExts.end())
                result.push_back(entry.path().string());
        }
        return result;
    }

    const sibr::Texture2DRGBA::Ptr& getTexture() const { return _texture; }
    const std::string&              getPath()    const { return _path; }

private:
    sibr::Texture2DRGBA::Ptr _texture;
    std::string              _path;
};


class SimplePointRenderer {
public:
    // Upload vertex positions from a PLY / mesh file to the GPU.
    void load(const std::string& path) {
        sibr::Mesh::Ptr mesh(new sibr::Mesh());
        if (!mesh->load(path)) return;
        _count = mesh->vertices().size();
        if (_count == 0) return;

        _rawData.clear();
        _rawData.reserve(_count * 3);
        for (const auto& v : mesh->vertices()) {
            _rawData.push_back(v.x());
            _rawData.push_back(v.y());
            _rawData.push_back(v.z());
        }

        glGenVertexArrays(1, &_vao);
        glGenBuffers(1, &_vbo);
        glBindVertexArray(_vao);
        glBindBuffer(GL_ARRAY_BUFFER, _vbo);
        glBufferData(GL_ARRAY_BUFFER, _rawData.size() * sizeof(float),
                     _rawData.data(), GL_STATIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        glBindVertexArray(0);
        _loaded = true;
    }

    // Read per-vertex `face_id` float property from a binary PLY header.
    void loadFids(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) return;

        std::string line;
        long vertex_count    = 0;
        int  face_id_offset  = -1;
        int  property_count  = 0;
        bool is_binary       = false;

        while (std::getline(file, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            std::stringstream ss(line);
            std::string token; ss >> token;

            if (token == "format") {
                ss >> token;
                if (token == "binary_little_endian") is_binary = true;
            } else if (token == "element") {
                ss >> token;
                if (token == "vertex") ss >> vertex_count;
            } else if (token == "property") {
                std::string type, name; ss >> type;
                if (type != "float") return;
                ss >> name;
                if (name == "face_id") face_id_offset = property_count;
                property_count++;
            } else if (token == "end_header") {
                break;
            }
        }

        if (!is_binary || vertex_count == 0 || face_id_offset == -1) return;

        _fids.clear();
        _fids.reserve(vertex_count);
        size_t stride = property_count * sizeof(float);
        std::vector<char> buf(stride);
        float* fbuf = reinterpret_cast<float*>(buf.data());
        for (long i = 0; i < vertex_count; ++i) {
            file.read(buf.data(), stride);
            if (!file) return;
            _fids.push_back(static_cast<int>(fbuf[face_id_offset]));
        }
    }

    void draw() const {
        if (!_loaded) return;
        glBindVertexArray(_vao);
        glDrawArrays(GL_POINTS, 0, (GLsizei)_count);
        glBindVertexArray(0);
    }

    bool                      isLoaded()   const { return _loaded; }
    const std::vector<float>& getRawData() const { return _rawData; }
    const std::vector<int>&   getFids()    const { return _fids; }

    // 讀取 PLY 裡的 opacity / scale_0~2 / rot_0~3
    // 必須在 load() 之後呼叫，使用同一個 PLY 路徑
    void loadGaussianProps(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) return;

        std::string line;
        long vertex_count = 0;
        bool is_binary    = false;
        std::map<std::string, int> propOffset;
        int prop_count = 0;

        while (std::getline(file, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            std::stringstream ss(line);
            std::string token; ss >> token;
            if (token == "format") {
                ss >> token;
                if (token == "binary_little_endian") is_binary = true;
            } else if (token == "element") {
                ss >> token;
                if (token == "vertex") ss >> vertex_count;
            } else if (token == "property") {
                std::string type, name; ss >> type >> name;
                if (type == "float") { propOffset[name] = prop_count++; }
            } else if (token == "end_header") {
                break;
            }
        }
        if (!is_binary || vertex_count == 0) return;

        static const std::vector<std::string> kRequired =
            {"opacity","scale_0","scale_1","scale_2","rot_0","rot_1","rot_2","rot_3"};
        for (const auto& r : kRequired)
            if (!propOffset.count(r)) { std::cerr << "[loadGaussianProps] missing: " << r << "\n"; return; }

        _gaussianProps.clear();
        _gaussianProps.reserve(vertex_count);
        const size_t stride = prop_count * sizeof(float);
        std::vector<float> buf(prop_count);

        for (long i = 0; i < vertex_count; ++i) {
            file.read(reinterpret_cast<char*>(buf.data()), stride);
            if (!file) { _gaussianProps.clear(); return; }
            GaussianProps p;
            p.opacity  = buf[propOffset.at("opacity")];
            p.scale[0] = buf[propOffset.at("scale_0")];
            p.scale[1] = buf[propOffset.at("scale_1")];
            p.scale[2] = buf[propOffset.at("scale_2")];
            p.rot[0]   = buf[propOffset.at("rot_0")];
            p.rot[1]   = buf[propOffset.at("rot_1")];
            p.rot[2]   = buf[propOffset.at("rot_2")];
            p.rot[3]   = buf[propOffset.at("rot_3")];
            _gaussianProps.push_back(p);
        }
    }

    const std::vector<GaussianProps>& getGaussianProps() const { return _gaussianProps; }

private:
    GLuint _vao    = 0;
    GLuint _vbo    = 0;
    size_t _count  = 0;
    bool   _loaded = false;
    std::vector<float>        _rawData;
    std::vector<int>          _fids;
    std::vector<GaussianProps> _gaussianProps;
};


class PointCloudRenderer {
public:
    PointCloudRenderer()  { init(); }
    ~PointCloudRenderer() { if (_progID) glDeleteProgram(_progID); }

    // Set MVP, configure GL state, and draw both point clouds.
    void render(
        const glm::mat4&           mvp,
        float                      pointSize,
        const SimplePointRenderer& geo,      const sibr::Vector3f& geoColor, bool showGeo,
        const SimplePointRenderer& app,      const sibr::Vector3f& appColor, bool showApp
    ) {
        glUseProgram(_progID);
        glUniformMatrix4fv(_locMVP, 1, GL_FALSE, glm::value_ptr(mvp));
        glUniform1i(_locUseLighting, 0);

        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LEQUAL);
        glEnable(GL_POLYGON_OFFSET_POINT);
        glPolygonOffset(0.0f, -10.0f);
        glPointSize(pointSize);

        if (showGeo && geo.isLoaded()) {
            glUniform3f(_locColor, geoColor.x(), geoColor.y(), geoColor.z());
            geo.draw();
        }
        if (showApp && app.isLoaded()) {
            glUniform3f(_locColor, appColor.x(), appColor.y(), appColor.z());
            app.draw();
        }

        glDisable(GL_POLYGON_OFFSET_POINT);
        glUseProgram(0);
    }

private:
    void init() {
        GLuint vs = detail::compileGLShader(detail::meshVertSrc(), GL_VERTEX_SHADER);
        GLuint fs = detail::compileGLShader(detail::meshFragSrc(), GL_FRAGMENT_SHADER);
        _progID = detail::linkProgram(vs, fs);
        _locMVP         = glGetUniformLocation(_progID, "uMVP");
        _locColor       = glGetUniformLocation(_progID, "uColor");
        _locUseLighting = glGetUniformLocation(_progID, "uUseLighting");
    }

    GLuint _progID         = 0;
    GLint  _locMVP         = -1;
    GLint  _locColor       = -1;
    GLint  _locUseLighting = -1;
};


class MeshWireframeRenderer {
public:
    MeshWireframeRenderer()  { init(); }
    ~MeshWireframeRenderer() { 
        if (_progID) glDeleteProgram(_progID); 
        if (_staticVao) {
            glDeleteBuffers(1, &_staticVbo);
            glDeleteVertexArrays(1, &_staticVao);
        }
    }

    // Upload the entire mesh wireframe to GPU memory (called once)
    void uploadMesh(const sibr::Mesh* mesh) {
        if (!mesh || mesh->vertices().empty()) return;

        const auto& verts   = mesh->vertices();
        const auto& normals = mesh->normals();
        const auto& tris    = mesh->triangles();

        std::vector<float> lines;
        lines.reserve(tris.size() * 6 * 6);
        for (const auto& tri : tris) {
            pushEdges(tri, verts, normals, lines);
        }

        if (_staticVao) {
            glDeleteBuffers(1, &_staticVbo);
            glDeleteVertexArrays(1, &_staticVao);
        }

        glGenVertexArrays(1, &_staticVao);
        glGenBuffers(1, &_staticVbo);
        glBindVertexArray(_staticVao);
        glBindBuffer(GL_ARRAY_BUFFER, _staticVbo);
        glBufferData(GL_ARRAY_BUFFER, lines.size() * sizeof(float), lines.data(), GL_STATIC_DRAW);

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(1);

        glBindVertexArray(0);
        _staticCount = (GLsizei)(lines.size() / 6);
    }

    void render(
        const sibr::Mesh*       mesh,
        const std::set<int>&    activeIndices,
        const glm::mat4&        mvp,
        bool                    showMesh
    ) {
        if (!mesh || mesh->vertices().empty()) return;

        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LEQUAL);
        glUseProgram(_progID);
        glUniformMatrix4fv(_locMVP, 1, GL_FALSE, glm::value_ptr(mvp));
        glUniform1i(_locUseLighting, 0);

        // 1. Draw entire static mesh in black (from GPU VRAM)
        if (showMesh && _staticVao) {
            glLineWidth(1.0f);
            glUniform3f(_locColor, 0.f, 0.f, 0.f);
            glBindVertexArray(_staticVao);
            glDrawArrays(GL_LINES, 0, _staticCount);
            glBindVertexArray(0);
        }

        // 2. Draw yellow lines — UV-active triangles (dynamic overlay)
        if (_showYellowWireframe && !activeIndices.empty()) {
            const auto& verts   = mesh->vertices();
            const auto& normals = mesh->normals();
            const auto& tris    = mesh->triangles();

            std::vector<float> lines;
            lines.reserve(activeIndices.size() * 6 * 6);
            for (int idx : activeIndices)
                if (idx >= 0 && idx < (int)tris.size())
                    pushEdges(tris[idx], verts, normals, lines);
            
            if (!lines.empty()) {
                glLineWidth(2.5f);
                glUniform3f(_locColor, 1.f, 1.f, 0.f);
                uploadAndDraw(lines);
            }
        }

        glUseProgram(0);
    }

    bool _showYellowWireframe = true;  // toggled from GUI

private:
    // Append 3 edges of a triangle (pos + normal, stride = 6 floats) to `out`.
    void pushEdges(
        const sibr::Vector3u&              tri,
        const std::vector<sibr::Vector3f>& verts,
        const std::vector<sibr::Vector3f>& normals,
        std::vector<float>&                out
    ) {
        const sibr::Vector3f kDefaultN(0.f, 1.f, 0.f);
        auto push = [&](unsigned int vid) {
            const auto& v = verts[vid];
            const auto& n = (vid < normals.size()) ? normals[vid] : kDefaultN;
            out.push_back(v.x()); out.push_back(v.y()); out.push_back(v.z());
            out.push_back(n.x()); out.push_back(n.y()); out.push_back(n.z());
        };
        push(tri[0]); push(tri[1]); // edge 0-1
        push(tri[1]); push(tri[2]); // edge 1-2
        push(tri[2]); push(tri[0]); // edge 2-0
    }

    // Upload a temporary VAO/VBO and draw as GL_LINES, then free immediately.
    void uploadAndDraw(const std::vector<float>& data) {
        GLuint vao, vbo;
        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, data.size() * sizeof(float), data.data(), GL_DYNAMIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(1);
        glDrawArrays(GL_LINES, 0, (GLsizei)(data.size() / 6));
        glBindVertexArray(0);
        glDeleteBuffers(1, &vbo);
        glDeleteVertexArrays(1, &vao);
    }

    void init() {
        GLuint vs = detail::compileGLShader(detail::meshVertSrc(), GL_VERTEX_SHADER);
        GLuint fs = detail::compileGLShader(detail::meshFragSrc(), GL_FRAGMENT_SHADER);
        _progID = detail::linkProgram(vs, fs);
        _locMVP         = glGetUniformLocation(_progID, "uMVP");
        _locColor       = glGetUniformLocation(_progID, "uColor");
        _locUseLighting = glGetUniformLocation(_progID, "uUseLighting");
    }

    GLuint _progID         = 0;
    GLint  _locMVP         = -1;
    GLint  _locColor       = -1;
    GLint  _locUseLighting = -1;

    GLuint  _staticVao     = 0;
    GLuint  _staticVbo     = 0;
    GLsizei _staticCount   = 0;
};

struct TextureSlotData {
    std::vector<sibr::Vector3u>      triangles;
    std::map<int, glm::vec2>         uvMap;
    sibr::Texture2DRGBA::Ptr         texture;
    std::string                      texturePath;
    std::string                      name;
    float                            alpha   = 1.f;
    bool                             visible = true;
    // GS 渲染所需：儲存此 slot 的投影 Gaussian 點
    std::vector<ProjectedGaussian>   geoPoints;
    std::vector<ProjectedGaussian>   appPoints;
};

class TextureProjector {
public:
    TextureProjector()  { init(); }
    ~TextureProjector() { if (_progID) glDeleteProgram(_progID); }

    void renderSlot(
        const sibr::Mesh*                  mesh,
        const std::vector<sibr::Vector3u>& triangles,
        const std::map<int, glm::vec2>&    uvMap,
        const sibr::Texture2DRGBA::Ptr&    texture,
        const glm::mat4&                   mvp,
        const glm::vec3&                   camPos,
        float                              alpha
    ) {
        if (!mesh || mesh->vertices().empty()) return;
        if (triangles.empty() || !texture || texture->handle() == 0) return;

        const auto& vertices = mesh->vertices();
        const auto& normals  = mesh->normals();
        const sibr::Vector3f kDefaultN(0.f, 1.f, 0.f);

        // VBO layout: ndcX, ndcY, u, v, nx, ny, nz  (7 floats per vertex)
        std::vector<float> vboData;
        vboData.reserve(triangles.size() * 3 * 7);
        for (const auto& tri : triangles) {
            if (!uvMap.count(tri[0]) || !uvMap.count(tri[1]) || !uvMap.count(tri[2])) continue;

            std::vector<float> triVerts;
            bool allValid = true;
            for (int k = 0; k < 3; ++k) {
                int vid = tri[k];
                const auto& v  = vertices[vid];
                glm::vec4 clip = mvp * glm::vec4(v.x(), v.y(), v.z(), 1.f);
                if (clip.w < 1e-6f) { allValid = false; break; }

                glm::vec2 ndc(clip.x / clip.w, clip.y / clip.w);
                const glm::vec2& uv = uvMap.at(vid);
                const auto& n = (vid < (int)normals.size()) ? normals[vid] : kDefaultN;

                triVerts.push_back(ndc.x);
                triVerts.push_back(ndc.y);
                triVerts.push_back(uv.x);
                triVerts.push_back(1.f - uv.y); // flip Y
                triVerts.push_back(n.x());
                triVerts.push_back(n.y());
                triVerts.push_back(n.z());
            }
            if (allValid)
                for (float f : triVerts) vboData.push_back(f);
        }
        if (vboData.empty()) return;

        GLuint vao, vbo;
        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, vboData.size() * sizeof(float), vboData.data(), GL_DYNAMIC_DRAW);

        constexpr GLsizei stride = 7 * sizeof(float);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride, (void*)0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, (void*)(2 * sizeof(float)));
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride, (void*)(4 * sizeof(float)));
        glEnableVertexAttribArray(2);

        glUseProgram(_progID);
        glUniform1f(_locAlpha, alpha);
        glUniform1i(_locUseLighting, _useLighting ? 1 : 0);

        if (_useLighting) {
            // Light direction derived from camera: camera direction + upward bias
            glm::vec3 camDir  = glm::length(camPos) > 1e-4f ? glm::normalize(camPos) : glm::vec3(0, 0, 1);
            glm::vec3 lightDir = glm::normalize(camDir + glm::vec3(0.f, 0.4f, 0.f));
            glUniform3f(_locLightDir,         lightDir.x, lightDir.y, lightDir.z);
            glUniform1f(_locAmbient,          kAmbient);
            glUniform1f(_locDiffuseStrength,  kDiffuseStrength);
            glUniform1f(_locSpecularStrength, kSpecularStrength);
            glUniform1f(_locShininess,        kShininess);
        }

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture->handle());

        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LEQUAL);
        glDepthMask(GL_TRUE);
        glEnable(GL_CULL_FACE);
        glCullFace(GL_BACK);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glDrawArrays(GL_TRIANGLES, 0, (GLsizei)(vboData.size() / 7));

        glDisable(GL_BLEND);
        glDisable(GL_CULL_FACE);
        glBindTexture(GL_TEXTURE_2D, 0);
        glUseProgram(0);
        glBindVertexArray(0);
        glDeleteBuffers(1, &vbo);
        glDeleteVertexArrays(1, &vao);
    }

    void render(
        const sibr::Mesh*                  mesh,
        const std::vector<sibr::Vector3u>& triangles,
        const std::map<int, glm::vec2>&    uvMap,
        const sibr::Texture2DRGBA::Ptr&    texture,
        const glm::mat4&                   mvp,
        const glm::vec3&                   camPos
    ) {
        if (!_enabled) return;
        renderSlot(mesh, triangles, uvMap, texture, mvp, camPos, _alpha);
    }

    void renderAll(
        const sibr::Mesh*                       mesh,
        const std::vector<TextureSlotData>&     slots,
        const glm::mat4&                        mvp,
        const glm::vec3&                        camPos
    ) {
        if (!_enabled) return;
        for (const auto& slot : slots) {
            if (!slot.visible) continue;
            renderSlot(mesh, slot.triangles, slot.uvMap, slot.texture, mvp, camPos, slot.alpha);
        }
    }

    // ImGui checkbox + alpha slider + lighting toggle.
    void renderGUI() {
        ImGui::Checkbox("Show Texture Projection", &_enabled);
        if (_enabled) {
            ImGui::SameLine();
            ImGui::PushItemWidth(100.f);
            ImGui::SliderFloat("Alpha##proj", &_alpha, 0.f, 1.f);
            ImGui::PopItemWidth();
            ImGui::SameLine();
            ImGui::Checkbox("Lighting##proj", &_useLighting);
        }
    }

    bool  isEnabled() const { return _enabled; }
    void  setEnabled(bool e) { _enabled = e; }
    float getAlpha()  const { return _alpha; }
    void  setAlpha(float a) { _alpha = a; }

private:
    void init() {
        // VBO layout: ndcX, ndcY, u, v, nx, ny, nz  (7 floats, stride = 28 bytes)
        const std::string vertSrc = R"(
            #version 330 core
            layout(location = 0) in vec2 aNDC;
            layout(location = 1) in vec2 aTexCoord;
            layout(location = 2) in vec3 aNormal;
            out vec2 vTexCoord;
            out vec3 vNormal;
            void main() {
                gl_Position = vec4(aNDC, 0.0, 1.0);
                vTexCoord   = aTexCoord;
                vNormal     = aNormal;
            }
        )";
        const std::string fragSrc = R"(
            #version 330 core
            in vec2 vTexCoord;
            in vec3 vNormal;
            out vec4 FragColor;
            uniform sampler2D uTexture;
            uniform float     uAlpha;
            uniform bool      uUseLighting;
            uniform vec3      uLightDir;
            uniform float     uAmbient;
            uniform float     uDiffuseStrength;
            uniform float     uSpecularStrength;
            uniform float     uShininess;
            void main() {
                vec4 texCol = texture(uTexture, vTexCoord);
                vec3 rgb    = texCol.rgb;
                if (uUseLighting) {
                    vec3  N    = normalize(vNormal);
                    vec3  L    = normalize(uLightDir);
                    float diff = max(dot(N, L), 0.0);
                    // Blinn-Phong: view approximated as (0,0,1) in NDC space
                    vec3  H    = normalize(L + vec3(0.0, 0.0, 1.0));
                    float spec = pow(max(dot(N, H), 0.0), uShininess);
                    float light = uAmbient
                                + uDiffuseStrength  * diff
                                + uSpecularStrength * spec;
                    rgb = clamp(rgb * light, 0.0, 1.0);
                }
                FragColor = vec4(rgb, texCol.a * uAlpha);
            }
        )";
        GLuint vs = detail::compileGLShader(vertSrc, GL_VERTEX_SHADER);
        GLuint fs = detail::compileGLShader(fragSrc, GL_FRAGMENT_SHADER);
        _progID = detail::linkProgram(vs, fs);
        _locAlpha            = glGetUniformLocation(_progID, "uAlpha");
        _locUseLighting      = glGetUniformLocation(_progID, "uUseLighting");
        _locLightDir         = glGetUniformLocation(_progID, "uLightDir");
        _locAmbient          = glGetUniformLocation(_progID, "uAmbient");
        _locDiffuseStrength  = glGetUniformLocation(_progID, "uDiffuseStrength");
        _locSpecularStrength = glGetUniformLocation(_progID, "uSpecularStrength");
        _locShininess        = glGetUniformLocation(_progID, "uShininess");
        glUseProgram(_progID);
        glUniform1i(glGetUniformLocation(_progID, "uTexture"), 0);
        glUseProgram(0);
    }

    GLuint _progID              = 0;
    GLint  _locAlpha            = -1;
    GLint  _locUseLighting      = -1;
    GLint  _locLightDir         = -1;
    GLint  _locAmbient          = -1;
    GLint  _locDiffuseStrength  = -1;
    GLint  _locSpecularStrength = -1;
    GLint  _locShininess        = -1;
    float  _alpha               = 1.f;
    bool   _enabled             = true;
    bool   _useLighting         = true;   // lighting ON by default

    // Hard-coded lighting constants
    static constexpr float kAmbient          = 0.35f;
    static constexpr float kDiffuseStrength  = 0.65f;
    static constexpr float kSpecularStrength = 0.15f;
    static constexpr float kShininess        = 32.f;
};

// ============================================================================
// GaussianSplatRenderer
//
// texture_gs 風格渲染：
//   每個 Gaussian 以 instanced quad 渲染成橢圓 splat，
//   顏色來自 ExpMap 解算出的 UV 座標對 texture 做 sample，
//   完全取代 SH 係數取色的標準 3DGS 做法。
//
// 渲染流程：
//   1. CPU 端依相機深度 back-to-front 排序（正確 alpha 合成）
//   2. 建 per-instance VBO：position / scale / rotation / opacity / uv
//   3. Vertex shader：3D covariance → screen-space conic → quad 大小
//   4. Fragment shader：Gaussian falloff alpha × texture2D(uv)
// ============================================================================
class GaussianSplatRenderer {
public:
    GaussianSplatRenderer()  { init(); }
    ~GaussianSplatRenderer() {
        if (_progID)  glDeleteProgram(_progID);
        if (_quadVBO) glDeleteBuffers(1, &_quadVBO);
    }

    // 渲染 live (active) slot
    void render(
        const std::vector<ProjectedGaussian>& geoPoints,
        const std::vector<ProjectedGaussian>& appPoints,
        const sibr::Texture2DRGBA::Ptr&       texture,
        const glm::mat4& view,
        const glm::mat4& proj,
        float vpW, float vpH
    ) {
        if (!_enabled) return;
        renderFlat(geoPoints, appPoints, texture, view, proj, vpW, vpH, _alpha);
    }

    // 渲染所有已儲存的 slots
    void renderAll(
        const std::vector<TextureSlotData>& slots,
        const glm::mat4& view,
        const glm::mat4& proj,
        float vpW, float vpH
    ) {
        if (!_enabled) return;
        for (const auto& slot : slots) {
            if (!slot.visible || !slot.texture) continue;
            renderFlat(slot.geoPoints, slot.appPoints, slot.texture, view, proj, vpW, vpH, slot.alpha);
        }
    }

    void renderGUI() {
        ImGui::Checkbox("Show Texture GS (splats)", &_enabled);
        if (_enabled) {
            ImGui::SameLine();
            ImGui::PushItemWidth(100.f);
            ImGui::SliderFloat("Alpha##gs", &_alpha, 0.f, 1.f);
            ImGui::PopItemWidth();
        }
    }

    bool  isEnabled() const { return _enabled; }
    float getAlpha()  const { return _alpha; }
    void  setAlpha(float a) { _alpha = a; }

private:
    void renderFlat(
        const std::vector<ProjectedGaussian>& geoPoints,
        const std::vector<ProjectedGaussian>& appPoints,
        const sibr::Texture2DRGBA::Ptr&       texture,
        const glm::mat4& view,
        const glm::mat4& proj,
        float vpW, float vpH,
        float alpha
    ) {
        if (!texture || texture->handle() == 0) return;

        // --- 1. 合併兩個點雲並依深度排序 (back-to-front) ---
        std::vector<const ProjectedGaussian*> all;
        all.reserve(geoPoints.size() + appPoints.size());
        for (const auto& p : geoPoints) all.push_back(&p);
        for (const auto& p : appPoints) all.push_back(&p);
        if (all.empty()) return;

        std::sort(all.begin(), all.end(), [&](const ProjectedGaussian* a, const ProjectedGaussian* b) {
            // view 矩陣第三行 = camera -Z 方向，取 view-space z
            float za = view[0][2]*a->position.x + view[1][2]*a->position.y
                     + view[2][2]*a->position.z + view[3][2];
            float zb = view[0][2]*b->position.x + view[1][2]*b->position.y
                     + view[2][2]*b->position.z + view[3][2];
            return za < zb;  // 最遠（最負）先畫
        });

        // --- 2. 建 per-instance buffer (13 floats/instance) ---
        // 佈局：pos(3) | scale(3) | rot(4) | opacity(1) | uv(2)
        constexpr int kStride = 13;
        std::vector<float> inst;
        inst.reserve(all.size() * kStride);
        for (const auto* p : all) {
            inst.push_back(p->position.x);  inst.push_back(p->position.y);  inst.push_back(p->position.z);
            inst.push_back(p->scale.x);     inst.push_back(p->scale.y);     inst.push_back(p->scale.z);
            // rotation: glm::vec4 儲存 (w,x,y,z) → 依序推入
            inst.push_back(p->rotation.x);  inst.push_back(p->rotation.y);
            inst.push_back(p->rotation.z);  inst.push_back(p->rotation.w);
            inst.push_back(p->opacity);
            inst.push_back(p->uv.x);        inst.push_back(p->uv.y);
        }

        // --- 3. 上傳 instance VBO ---
        GLuint instVBO;
        glGenBuffers(1, &instVBO);
        glBindBuffer(GL_ARRAY_BUFFER, instVBO);
        glBufferData(GL_ARRAY_BUFFER, inst.size()*sizeof(float), inst.data(), GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        // --- 4. 建立 VAO（quad + instance）---
        GLuint vao;
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);

        // loc 0: quad 頂點 (static)
        glBindBuffer(GL_ARRAY_BUFFER, _quadVBO);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2*sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);

        // loc 1~5: per-instance
        glBindBuffer(GL_ARRAY_BUFFER, instVBO);
        const GLsizei s = kStride * sizeof(float);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, s, (void*)(0));               glEnableVertexAttribArray(1); glVertexAttribDivisor(1,1);
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, s, (void*)(3*sizeof(float))); glEnableVertexAttribArray(2); glVertexAttribDivisor(2,1);
        glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, s, (void*)(6*sizeof(float))); glEnableVertexAttribArray(3); glVertexAttribDivisor(3,1);
        glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, s, (void*)(10*sizeof(float)));glEnableVertexAttribArray(4); glVertexAttribDivisor(4,1);
        glVertexAttribPointer(5, 2, GL_FLOAT, GL_FALSE, s, (void*)(11*sizeof(float)));glEnableVertexAttribArray(5); glVertexAttribDivisor(5,1);
        glBindVertexArray(0);

        // --- 5. 設定 uniform 並繪製 ---
        float focalX = proj[0][0] * vpW * 0.5f;
        float focalY = proj[1][1] * vpH * 0.5f;

        glUseProgram(_progID);
        glUniformMatrix4fv(_locView,   1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(_locProj,   1, GL_FALSE, glm::value_ptr(proj));
        glUniform1f(_locFocalX,  focalX);
        glUniform1f(_locFocalY,  focalY);
        glUniform1f(_locW,       vpW);
        glUniform1f(_locH,       vpH);
        glUniform1f(_locAlpha,   alpha);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture->handle());

        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LEQUAL);
        glDepthMask(GL_FALSE);          // splat 不寫入 depth buffer
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glDisable(GL_CULL_FACE);

        glBindVertexArray(vao);
        glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, (GLsizei)all.size());
        glBindVertexArray(0);

        // 還原狀態
        glDepthMask(GL_TRUE);
        glDisable(GL_BLEND);
        glBindTexture(GL_TEXTURE_2D, 0);
        glUseProgram(0);

        glDeleteVertexArrays(1, &vao);
        glDeleteBuffers(1, &instVBO);
    }

    void init() {
        // 靜態 unit quad (4 頂點，TRIANGLE_STRIP)
        const float quad[] = { -1.f,-1.f,  1.f,-1.f,  -1.f,1.f,  1.f,1.f };
        glGenBuffers(1, &_quadVBO);
        glBindBuffer(GL_ARRAY_BUFFER, _quadVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        // ---- Vertex Shader ----
        // 從 scale+rotation 計算 3D covariance → 投影到 screen-space conic
        // 決定 quad 大小，輸出 pixel offset (vDelta) 給 fragment 計算 Gaussian falloff
        const std::string vs = R"(
            #version 430 core
            layout(location=0) in vec2  aQuad;    // unit quad corner
            layout(location=1) in vec3  aPos;     // world-space center
            layout(location=2) in vec3  aScale;   // log scale (raw)
            layout(location=3) in vec4  aRot;     // quaternion (w,x,y,z) raw
            layout(location=4) in float aOpacity; // raw (before sigmoid)
            layout(location=5) in vec2  aUV;      // ExpMap UV

            uniform mat4  uView;
            uniform mat4  uProj;
            uniform float uFocalX;
            uniform float uFocalY;
            uniform float uW;
            uniform float uH;

            out vec2  vDelta;    // pixel offset from splat center
            out vec3  vConic;    // inverse 2D covariance (a,b,c)
            out float vOpacity;
            out vec2  vUV;

            void main() {
                // --- view-space position ---
                vec4 posV = uView * vec4(aPos, 1.0);
                float tz = posV.z;
                if (tz >= -0.1) { gl_Position = vec4(0,0,2,1); return; }
                float tx = posV.x, ty = posV.y, tz2 = tz*tz;

                // --- 3D covariance: Sigma = (R*S) * (R*S)^T ---
                vec3  sc = exp(aScale);
                float qw=aRot.x, qx=aRot.y, qy=aRot.z, qz=aRot.w;
                mat3 R = mat3(
                    vec3(1.0-2.0*(qy*qy+qz*qz), 2.0*(qx*qy+qw*qz), 2.0*(qx*qz-qw*qy)),
                    vec3(2.0*(qx*qy-qw*qz), 1.0-2.0*(qx*qx+qz*qz), 2.0*(qy*qz+qw*qx)),
                    vec3(2.0*(qx*qz+qw*qy), 2.0*(qy*qz-qw*qx), 1.0-2.0*(qx*qx+qy*qy))
                );
                mat3 S   = mat3(sc.x,0,0, 0,sc.y,0, 0,0,sc.z);
                mat3 M   = R * S;
                mat3 Sig = M * transpose(M);

                // --- Jacobian (perspective projection) ---
                mat3 J = mat3(
                    vec3(uFocalX/tz,  0.0, 0.0),
                    vec3(0.0, uFocalY/tz,  0.0),
                    vec3(-uFocalX*tx/tz2, -uFocalY*ty/tz2, 0.0)
                );
                mat3 W    = mat3(uView);
                mat3 T    = J * W;
                mat3 cov2 = T * Sig * transpose(T);

                // 2x2 covariance + low-pass filter
                float a = cov2[0][0] + 0.3;
                float b = cov2[0][1];
                float c = cov2[1][1] + 0.3;
                float det = a*c - b*b;
                if (det <= 0.0) { gl_Position = vec4(0,0,2,1); return; }

                // --- conic (inverse 2x2 cov) ---
                vConic = vec3(c/det, -b/det, a/det);

                // --- quad size from larger eigenvalue ---
                float mid  = 0.5*(a+c);
                float disc = sqrt(max(0.1, mid*mid - det));
                float rad  = ceil(3.0 * sqrt(mid + disc));

                // --- screen-space position ---
                vec4  posC      = uProj * posV;
                vec2  centerNDC = posC.xy / posC.w;
                vec2  delta     = aQuad * rad;
                gl_Position = vec4(centerNDC + delta*(2.0/vec2(uW,uH)),
                                   posC.z/posC.w, 1.0);
                vDelta   = delta;
                vUV      = aUV;
                vOpacity = aOpacity;
            }
        )";

        // ---- Fragment Shader ----
        // Gaussian falloff × texture sample = texture_gs 的核心
        const std::string fs = R"(
            #version 330 core
            in vec2  vDelta;
            in vec3  vConic;
            in float vOpacity;
            in vec2  vUV;
            out vec4 FragColor;
            uniform sampler2D uTexture;
            uniform float     uAlpha;

            void main() {
                float dx = vDelta.x, dy = vDelta.y;
                // Gaussian power: -0.5 * x^T * Sigma^{-1} * x
                float power = -0.5*(vConic.x*dx*dx + 2.0*vConic.y*dx*dy + vConic.z*dy*dy);
                if (power > 0.0) discard;

                // sigmoid(opacity) * exp(power) = alpha of this splat
                float alpha = min(0.99, (1.0/(1.0+exp(-vOpacity))) * exp(power));
                if (alpha < 1.0/255.0) discard;

                // texture_gs: colour from UV sample, NOT from SH coefficients
                vec4 c = texture(uTexture, vUV);
                FragColor = vec4(c.rgb, c.a * alpha * uAlpha);
            }
        )";

        GLuint sv = detail::compileGLShader(vs, GL_VERTEX_SHADER);
        GLuint sf = detail::compileGLShader(fs, GL_FRAGMENT_SHADER);
        _progID = detail::linkProgram(sv, sf);

        _locView   = glGetUniformLocation(_progID, "uView");
        _locProj   = glGetUniformLocation(_progID, "uProj");
        _locFocalX = glGetUniformLocation(_progID, "uFocalX");
        _locFocalY = glGetUniformLocation(_progID, "uFocalY");
        _locW      = glGetUniformLocation(_progID, "uW");
        _locH      = glGetUniformLocation(_progID, "uH");
        _locAlpha  = glGetUniformLocation(_progID, "uAlpha");

        glUseProgram(_progID);
        glUniform1i(glGetUniformLocation(_progID, "uTexture"), 0);
        glUseProgram(0);
    }

    GLuint _progID  = 0;
    GLuint _quadVBO = 0;

    GLint  _locView   = -1;
    GLint  _locProj   = -1;
    GLint  _locFocalX = -1;
    GLint  _locFocalY = -1;
    GLint  _locW      = -1;
    GLint  _locH      = -1;
    GLint  _locAlpha  = -1;

    float _alpha   = 1.f;
    bool  _enabled = true;
};