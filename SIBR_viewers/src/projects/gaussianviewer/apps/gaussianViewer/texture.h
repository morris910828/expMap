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

private:
    GLuint _vao    = 0;
    GLuint _vbo    = 0;
    size_t _count  = 0;
    bool   _loaded = false;
    std::vector<float> _rawData;
    std::vector<int>   _fids;
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
    ~MeshWireframeRenderer() { if (_progID) glDeleteProgram(_progID); }

    // activeIndices: set of triangle indices (into mesh->triangles()) that are UV-active.
    // Passing indices directly avoids the previous O(activeTris * totalTris) search.
    void render(
        const sibr::Mesh*       mesh,
        const std::set<int>&    activeIndices,
        const glm::mat4&        mvp,
        bool                    showMesh
    ) {
        if (!mesh || mesh->vertices().empty()) return;

        const auto& verts   = mesh->vertices();
        const auto& normals = mesh->normals();
        const auto& tris    = mesh->triangles();

        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LEQUAL);
        glUseProgram(_progID);
        glUniformMatrix4fv(_locMVP, 1, GL_FALSE, glm::value_ptr(mvp));
        glUniform1i(_locUseLighting, 0);

        // Black lines — non-UV triangles (O(N) single pass)
        if (showMesh) {
            std::vector<float> lines;
            lines.reserve(tris.size() * 6 * 6);
            for (size_t i = 0; i < tris.size(); ++i)
                if (!activeIndices.count((int)i)) pushEdges(tris[i], verts, normals, lines);
            if (!lines.empty()) {
                glLineWidth(1.0f);
                glUniform3f(_locColor, 0.f, 0.f, 0.f);
                uploadAndDraw(lines);
            }
        }

        // Yellow lines — UV-active triangles (O(K), K = active count)
        if (!activeIndices.empty()) {
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
};

struct TextureSlotData {
    std::vector<sibr::Vector3u>   triangles;
    std::map<int, glm::vec2>      uvMap;
    sibr::Texture2DRGBA::Ptr      texture;
    std::string                   texturePath;
    std::string                   name;
    float                         alpha = 1.f;
    bool                          visible = true;
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
        float                              alpha
    ) {
        if (!mesh || mesh->vertices().empty()) return;
        if (triangles.empty() || !texture || texture->handle() == 0) return;

        const auto& vertices = mesh->vertices();

        std::vector<float> vboData;
        vboData.reserve(triangles.size() * 3 * 4);
        for (const auto& tri : triangles) {
            if (!uvMap.count(tri[0]) || !uvMap.count(tri[1]) || !uvMap.count(tri[2])) continue;
            
            // Process all 3 vertices of the triangle
            std::vector<float> triVerts;
            bool allValid = true;
            for (int k = 0; k < 3; ++k) {
                int vid = tri[k];
                const auto& v = vertices[vid];
                glm::vec4 clip = mvp * glm::vec4(v.x(), v.y(), v.z(), 1.f);
                if (clip.w < 1e-6f) {
                    allValid = false;
                    break;
                }
                glm::vec2 ndc(clip.x / clip.w, clip.y / clip.w);
                const glm::vec2& uv = uvMap.at(vid);
                triVerts.push_back(ndc.x);
                triVerts.push_back(ndc.y);
                triVerts.push_back(uv.x);
                triVerts.push_back(1.f - uv.y); // flip Y
            }
            // Only add triangle if all vertices are valid
            if (allValid) {
                for (float v : triVerts) vboData.push_back(v);
            }
        }
        if (vboData.empty()) return;

        GLuint vao, vbo;
        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, vboData.size() * sizeof(float), vboData.data(), GL_DYNAMIC_DRAW);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
        glEnableVertexAttribArray(1);

        glUseProgram(_progID);
        glUniform1f(_locAlpha, alpha);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture->handle());

        // FIXED: Enable depth test to prevent texture from showing through backfaces
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LEQUAL);
        glDepthMask(GL_TRUE);
        
        // Enable back-face culling to avoid rendering on back-facing triangles
        glEnable(GL_CULL_FACE);
        glCullFace(GL_BACK);
        
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glDrawArrays(GL_TRIANGLES, 0, (GLsizei)(vboData.size() / 4));

        // Restore state
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
        const glm::mat4&                   mvp
    ) {
        if (!_enabled) return;
        renderSlot(mesh, triangles, uvMap, texture, mvp, _alpha);
    }

    void renderAll(
        const sibr::Mesh*                       mesh,
        const std::vector<TextureSlotData>&     slots,
        const glm::mat4&                        mvp
    ) {
        if (!_enabled) return;
        for (const auto& slot : slots) {
            if (!slot.visible) continue;
            renderSlot(mesh, slot.triangles, slot.uvMap, slot.texture, mvp, slot.alpha);
        }
    }

    // ImGui checkbox + alpha slider.
    void renderGUI() {
        ImGui::Checkbox("Show Texture Projection", &_enabled);
        if (_enabled) {
            ImGui::SameLine();
            ImGui::PushItemWidth(100.f);
            ImGui::SliderFloat("Alpha##proj", &_alpha, 0.f, 1.f);
            ImGui::PopItemWidth();
        }
    }

    bool  isEnabled() const { return _enabled; }
    void  setEnabled(bool e) { _enabled = e; }
    float getAlpha()  const { return _alpha; }
    void  setAlpha(float a) { _alpha = a; }

private:
    void init() {
        const std::string vertSrc = R"(
            #version 330 core
            layout(location = 0) in vec2 aNDC;
            layout(location = 1) in vec2 aTexCoord;
            out vec2 vTexCoord;
            void main() {
                gl_Position = vec4(aNDC, 0.0, 1.0);
                vTexCoord = aTexCoord;
            }
        )";
        const std::string fragSrc = R"(
            #version 330 core
            in vec2 vTexCoord;
            out vec4 FragColor;
            uniform sampler2D uTexture;
            uniform float uAlpha;
            void main() {
                vec4 col = texture(uTexture, vTexCoord);
                FragColor = vec4(col.rgb, col.a * uAlpha);
            }
        )";
        GLuint vs = detail::compileGLShader(vertSrc, GL_VERTEX_SHADER);
        GLuint fs = detail::compileGLShader(fragSrc, GL_FRAGMENT_SHADER);
        _progID = detail::linkProgram(vs, fs);
        _locAlpha = glGetUniformLocation(_progID, "uAlpha");
        glUseProgram(_progID);
        glUniform1i(glGetUniformLocation(_progID, "uTexture"), 0);
        glUseProgram(0);
    }

    GLuint _progID   = 0;
    GLint  _locAlpha = -1;
    float  _alpha    = 1.f;
    bool   _enabled  = true;   // default ON so saved slots are visible
};