#include "stb_image.h"
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"

#include "Camera.h"
#include "Shader.h"
#include "Model.h"
#include "ExpMap.h"

#include <iostream>
#include <vector>
#include <string>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/ext/matrix_clip_space.hpp>

int SCR_WIDTH = 1280;
int SCR_HEIGHT = 720;

Model* loadedModel = nullptr;

std::vector<std::string> modelList = {
    "../assets/armadillo.obj",
    "../assets/bear.obj",
    "../assets/dancer.obj"
};

int currentModelIndex = 0;
bool useCustomModel = false;
char customModelPath[256] = "";

// Orbit Camera
bool leftMouseDown = false;
float orbitYaw = 0.0f, orbitPitch = 0.0f;
float targetDistance = 2.0f;
glm::vec3 targetPoint(0.0f);

float lastX = 0, lastY = 0;
bool firstMouse = true;

bool showTriangles = false;
bool dragSelecting = false;
bool dragOrbit = false;

// ExpMap selection
std::vector<SelectedTri> g_selected;
std::vector<FlattenTri>  g_flattened;
ExpMapSystem g_expMapSystem;

// Highlight
unsigned int highlightVAO = 0;
unsigned int highlightVBO = 0;

// Patch system
struct PatchVertex {
    glm::vec3 pos;
    glm::vec2 uv;
};

struct Patch {
    std::vector<PatchVertex> vertices;
    std::vector<SelectedTri> selectedTris;
    std::vector<FlattenTri> flattenedUVs;
    ExpMapSystem expSystem;
    int meshIndex = 0;
    
    GLuint vao = 0, vbo = 0;
    GLuint textureID = 0;

    ExpMapTransform transform;
};

std::vector<Patch> g_patches;
int selectedPatchIndex = -1;

// Texture list
std::vector<std::string> textureList = {
    "../assets/t1.png",
    "../assets/t2.jpg",
    "../assets/3.png",
};

int currentTextureIndex = 0;

// Load texture
GLuint LoadTexture(const std::string& path)
{
    int w, h, c;
    stbi_set_flip_vertically_on_load(true);
    unsigned char* data = stbi_load(path.c_str(), &w, &h, &c, 4);

    if (!data)
    {
        std::cout << "Failed to load texture: " << path << std::endl;
        return 0;
    }

    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, data);
    glGenerateMipmap(GL_TEXTURE_2D);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    stbi_image_free(data);
    return tex;
}

// Update patch with transform
void UpdatePatchWithTransform(Patch& P)
{
    if (P.selectedTris.empty() || P.flattenedUVs.empty()) return;

    std::vector<glm::vec3> newPositions;
    std::vector<glm::vec2> newTexCoords;
    
    ApplyExpMapTransform(loadedModel, P.selectedTris, P.flattenedUVs, P.expSystem, 
                        P.transform, newPositions, newTexCoords);

    // Rebuild vertices
    P.vertices.clear();
    for (size_t i = 0; i < newPositions.size(); i++)
    {
        PatchVertex v;
        v.pos = newPositions[i];
        v.uv = newTexCoords[i];
        P.vertices.push_back(v);
    }

    // Update GPU buffer
    glBindBuffer(GL_ARRAY_BUFFER, P.vbo);
    glBufferData(GL_ARRAY_BUFFER,
                P.vertices.size() * sizeof(PatchVertex),
                P.vertices.data(),
                GL_DYNAMIC_DRAW);
}

// Highlight buffer
void InitHighlightBuffers()
{
    glGenVertexArrays(1, &highlightVAO);
    glGenBuffers(1, &highlightVBO);

    glBindVertexArray(highlightVAO);
    glBindBuffer(GL_ARRAY_BUFFER, highlightVBO);
    glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);

    glBindVertexArray(0);
}

void UpdateHighlightBuffer(Model* model)
{
    std::vector<glm::vec3> verts;

    for (auto& t : g_selected)
    {
        Mesh& mesh = model->meshes[t.meshIndex];
        verts.push_back(mesh.vertices[t.i0].Position);
        verts.push_back(mesh.vertices[t.i1].Position);
        verts.push_back(mesh.vertices[t.i2].Position);
    }

    glBindBuffer(GL_ARRAY_BUFFER, highlightVBO);
    glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(glm::vec3), verts.data(), GL_DYNAMIC_DRAW);
}

// Ray casting
struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
};

struct HitInfo {
    bool hit = false;
    glm::vec3 hitPos;
    int meshIndex = -1;
    int triIndex  = -1;
};

Ray ScreenRay(float mx, float my, const glm::mat4& view, const glm::mat4& proj, const glm::vec3& camPos)
{
    float x = (2.0f * mx) / SCR_WIDTH - 1.0f;
    float y = 1.0f - (2.0f * my) / SCR_HEIGHT;

    glm::vec4 clip(x, y, -1, 1);
    glm::vec4 eye = glm::inverse(proj) * clip;
    eye = glm::vec4(eye.x, eye.y, -1, 0);

    glm::vec3 dir = glm::normalize(glm::vec3(glm::inverse(view) * eye));

    return { camPos, dir };
}

bool RayTri(const Ray& r, const glm::vec3& A, const glm::vec3& B, const glm::vec3& C, float& tOut)
{
    const float EPS = 1e-6f;
    glm::vec3 e1 = B - A;
    glm::vec3 e2 = C - A;

    glm::vec3 p = glm::cross(r.direction, e2);
    float det = glm::dot(e1, p);
    if (fabs(det) < EPS) return false;

    float inv = 1.0f / det;

    glm::vec3 s = r.origin - A;
    float u = glm::dot(s, p) * inv;
    if (u < 0 || u > 1) return false;

    glm::vec3 q = glm::cross(s, e1);
    float v = glm::dot(r.direction, q) * inv;
    if (v < 0 || u + v > 1) return false;

    float t = glm::dot(e2, q) * inv;
    if (t > EPS) { tOut = t; return true; }

    return false;
}

bool Raycast(Model* model, const Ray& r, HitInfo& out)
{
    float closest = FLT_MAX;

    for (int m = 0; m < model->meshes.size(); m++)
    {
        Mesh& mesh = model->meshes[m];

        for (int i = 0; i < mesh.indices.size(); i += 3)
        {
            glm::vec3 a = mesh.vertices[mesh.indices[i+0]].Position;
            glm::vec3 b = mesh.vertices[mesh.indices[i+1]].Position;
            glm::vec3 c = mesh.vertices[mesh.indices[i+2]].Position;

            float t = 0;
            if (RayTri(r, a, b, c, t))
            {
                if (t < closest)
                {
                    closest = t;
                    out.hit = true;
                    out.hitPos = r.origin + r.direction * t;
                    out.meshIndex = m;
                    out.triIndex  = i;
                }
            }
        }
    }

    return out.hit;
}

// Mouse callback
void mouse_callback(GLFWwindow*, double x, double y)
{
    if (!leftMouseDown || ImGui::GetIO().WantCaptureMouse)
        return;

    if (dragOrbit)
    {
        if (firstMouse) { lastX = x; lastY = y; firstMouse = false; }

        orbitYaw   += (x - lastX) * 0.3f;
        orbitPitch += (lastY - y) * 0.3f;

        orbitPitch = glm::clamp(orbitPitch, -89.0f, 89.0f);

        lastX = x;
        lastY = y;
    }
    else if (dragSelecting)
    {
        double mx = x, my = y;

        float yawRad = glm::radians(orbitYaw);
        float pitRad = glm::radians(orbitPitch);

        glm::vec3 cam(
            targetDistance * cos(pitRad) * sin(yawRad),
            targetDistance * sin(pitRad),
            targetDistance * cos(pitRad) * cos(yawRad)
        );
        cam += targetPoint;

        glm::mat4 view = glm::lookAt(cam, targetPoint, glm::vec3(0,1,0));
        glm::mat4 proj = glm::perspectiveRH_ZO( glm::radians(45.0f), (float)SCR_WIDTH / SCR_HEIGHT, 0.1f, 100.0f );

        Ray ray = ScreenRay(mx, my, view, proj, cam);

        HitInfo hit;
        if (Raycast(loadedModel, ray, hit))
        {
            Mesh& mesh = loadedModel->meshes[hit.meshIndex];

            unsigned int i0 = mesh.indices[hit.triIndex + 0];
            unsigned int i1 = mesh.indices[hit.triIndex + 1];
            unsigned int i2 = mesh.indices[hit.triIndex + 2];

            g_selected.push_back({hit.meshIndex, i0, i1, i2});
            UpdateHighlightBuffer(loadedModel);

            g_expMapSystem = ComputeExpMap(loadedModel, hit.meshIndex, g_selected, g_flattened);
        }
    }
}

// Mouse button
void mouse_button_callback(GLFWwindow* win, int button, int action, int)
{
    if (button != GLFW_MOUSE_BUTTON_LEFT) return;

    if (action == GLFW_PRESS)
    {
        double mx, my;
        glfwGetCursorPos(win, &mx, &my);

        leftMouseDown = true;
        firstMouse = true;
        dragOrbit = false;
        dragSelecting = false;

        float yawRad = glm::radians(orbitYaw);
        float pitRad = glm::radians(orbitPitch);

        glm::vec3 cam( targetDistance * cos(pitRad) * sin(yawRad), targetDistance * sin(pitRad), targetDistance * cos(pitRad) * cos(yawRad) );
        cam += targetPoint;

        glm::mat4 view = glm::lookAt(cam, targetPoint, glm::vec3(0,1,0));
        glm::mat4 proj = glm::perspectiveRH_ZO( glm::radians(45.0f), (float)SCR_WIDTH / SCR_HEIGHT, 0.1f, 100.0f );

        Ray ray = ScreenRay(mx, my, view, proj, cam);

        HitInfo hit;
        if (Raycast(loadedModel, ray, hit))
            dragSelecting = true;
        else
            dragOrbit = true;
    }
    else if (action == GLFW_RELEASE)
    {
        leftMouseDown = false;
        dragOrbit = false;
        dragSelecting = false;
    }
}

// Scroll
void scroll_callback(GLFWwindow*, double, double dy)
{
    targetDistance = glm::clamp( targetDistance - (float)dy * 0.5f, 1.0f, 50.0f );
}

// MAIN
int main()
{
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* win = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "ExpMap Surface Transform", 0, 0);
    glfwMakeContextCurrent(win);
    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);

    glEnable(GL_DEPTH_TEST);

    glfwSetCursorPosCallback(win, mouse_callback);
    glfwSetMouseButtonCallback(win, mouse_button_callback);
    glfwSetScrollCallback(win, scroll_callback);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(win, true);
    ImGui_ImplOpenGL3_Init("#version 450");
    ImGui::StyleColorsDark();

    Shader modelShader("../src/shaders/model.vert", "../src/shaders/model.frag");
    Shader wireShader ("../src/shaders/wireframe.vert", "../src/shaders/wireframe.frag");
    Shader patchShader("../src/shaders/highlight.vert", "../src/shaders/highlight.frag");

    InitHighlightBuffers();

    loadedModel = new Model(modelList[currentModelIndex]);

    while (!glfwWindowShouldClose(win))
    {
        glfwPollEvents();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // =========================================================
        // UI PANEL
        // =========================================================
        ImGui::Begin("Model Viewer");

        const char* preview = useCustomModel ? customModelPath : modelList[currentModelIndex].c_str();

        if (ImGui::BeginCombo("Model", preview))
        {
            for (int i = 0; i < modelList.size(); i++)
            {
                bool selected = (!useCustomModel && currentModelIndex == i);

                if (ImGui::Selectable(modelList[i].c_str(), selected))
                {
                    useCustomModel = false;
                    currentModelIndex = i;

                    delete loadedModel;
                    loadedModel = new Model(modelList[i]);

                    g_selected.clear();
                    g_flattened.clear();
                    g_patches.clear();
                    selectedPatchIndex = -1;
                }
                if (selected) ImGui::SetItemDefaultFocus();
            }

            ImGui::Separator();
            if (ImGui::Selectable("Custom Path", useCustomModel))
                useCustomModel = true;

            ImGui::EndCombo();
        }

        if (useCustomModel)
        {
            ImGui::InputText("Path", customModelPath, sizeof(customModelPath));
            if (ImGui::Button("Load"))
            {
                delete loadedModel;
                loadedModel = new Model(customModelPath);

                g_selected.clear();
                g_flattened.clear();
                g_patches.clear();
                selectedPatchIndex = -1;
            }
        }

        ImGui::Checkbox("Show Triangles", &showTriangles);

        ImGui::Separator();
        ImGui::Text("Selected Triangles: %d", (int)g_selected.size());
        
        if (ImGui::Button("Clear Selection"))
        {
            g_selected.clear();
            g_flattened.clear();
            UpdateHighlightBuffer(loadedModel);
        }

        ImGui::Separator();
        ImGui::Text("Texture:");

        const char* tPrev = textureList[currentTextureIndex].c_str();
        if (ImGui::BeginCombo("Texture List", tPrev))
        {
            for (int i = 0; i < textureList.size(); i++)
            {
                bool chosen = (currentTextureIndex == i);

                if (ImGui::Selectable(textureList[i].c_str(), chosen))
                    currentTextureIndex = i;

                if (chosen) ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }

        if (ImGui::Button("Apply Texture to Selected"))
        {
            if (!g_selected.empty())
            {
                Patch P;
                P.textureID = LoadTexture(textureList[currentTextureIndex]);
                P.meshIndex = g_selected[0].meshIndex;
                P.expSystem = g_expMapSystem;

                // 初始化變換參數
                P.transform.scale = 1.0f;
                P.transform.rotation = 0.0f;
                P.transform.surfaceOffset = glm::vec2(0.0f);
                std::vector<TriInfo> expandedInfos;
                ExpandExpMap(loadedModel, P.meshIndex, g_selected, P.expSystem, 2.0f, expandedInfos);

                // 將擴展後的資料轉存入 Patch
                P.selectedTris.clear();
                P.flattenedUVs.clear();
                Mesh& mesh = loadedModel->meshes[P.meshIndex];

                for(auto& info : expandedInfos) 
                {
                    int idx0 = mesh.indices[info.triIdx * 3 + 0];
                    int idx1 = mesh.indices[info.triIdx * 3 + 1];
                    int idx2 = mesh.indices[info.triIdx * 3 + 2];
                    
                    P.selectedTris.push_back({
                        P.meshIndex, 
                        (unsigned int)idx0, 
                        (unsigned int)idx1, 
                        (unsigned int)idx2
                    });
                    
                    // 2. 還原 FlattenTri 結構 (用於 UV 計算)
                    FlattenTri ft;
                    ft.a = info.uv0;
                    ft.b = info.uv1;
                    ft.c = info.uv2;
                    ft.triIndex = info.triIdx;
                    P.flattenedUVs.push_back(ft);
                }

                // =====================================================
                // 建立幾何資料 (呼叫新的 ApplyExpMapTransform)
                // =====================================================
                std::vector<glm::vec3> positions;
                std::vector<glm::vec2> texCoords;
                
                // 這裡會根據 P.transform.scale (目前是 1.0) 來決定顯示哪些三角形
                ApplyExpMapTransform(loadedModel, P.selectedTris, P.flattenedUVs, P.expSystem,
                                    P.transform, positions, texCoords);

                P.vertices.clear();
                for (size_t i = 0; i < positions.size(); i++)
                {
                    PatchVertex v;
                    v.pos = positions[i];
                    v.uv = texCoords[i];
                    P.vertices.push_back(v);
                }

                // =====================================================
                // 設定 OpenGL 緩衝區 (VAO / VBO)
                // =====================================================
                glGenVertexArrays(1, &P.vao);
                glGenBuffers(1, &P.vbo);

                glBindVertexArray(P.vao);
                glBindBuffer(GL_ARRAY_BUFFER, P.vbo);

                // 注意：這裡分配的大小是基於目前顯示的頂點數量
                // 如果之後動態縮放導致頂點數變多，UpdatePatchWithTransform 會重新分配
                glBufferData(GL_ARRAY_BUFFER,
                    P.vertices.size() * sizeof(PatchVertex),
                    P.vertices.data(),
                    GL_DYNAMIC_DRAW
                );

                glEnableVertexAttribArray(0);
                glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE,
                                    sizeof(PatchVertex),
                                    (void*)offsetof(PatchVertex, pos));

                glEnableVertexAttribArray(1);
                glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE,
                                    sizeof(PatchVertex),
                                    (void*)offsetof(PatchVertex, uv));

                glBindVertexArray(0);

                // 存入 Patch 列表
                g_patches.push_back(P);

                // 清除目前的預覽選取
                g_selected.clear();
                g_flattened.clear();
                UpdateHighlightBuffer(loadedModel);
            }
        }

        // =====================================================
        // Transform Controls
        // =====================================================
        ImGui::Separator();
        ImGui::Text("Texture Transform (Surface-based)");

        if (ImGui::BeginCombo("Select Patch", 
            selectedPatchIndex >= 0 ? 
            ("Patch " + std::to_string(selectedPatchIndex)).c_str() : "None"))
        {
            for (int i = 0; i < g_patches.size(); i++)
            {
                bool selected = (selectedPatchIndex == i);
                if (ImGui::Selectable(("Patch " + std::to_string(i)).c_str(), selected))
                    selectedPatchIndex = i;
                if (selected) ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }

        if (selectedPatchIndex >= 0 && selectedPatchIndex < g_patches.size())
        {
            Patch& P = g_patches[selectedPatchIndex];
            bool changed = false;

            ImGui::Text("Surface Translation:");
            ImGui::Text("(Move across the surface)");
            changed |= ImGui::SliderFloat("Move U", &P.transform.surfaceOffset.x, -1.0f, 1.0f);
            changed |= ImGui::SliderFloat("Move V", &P.transform.surfaceOffset.y, -1.0f, 1.0f);

            ImGui::Separator();
            ImGui::Text("Rotation (around center):");
            float rotDeg = glm::degrees(P.transform.rotation);
            if (ImGui::SliderFloat("Angle", &rotDeg, -180.0f, 180.0f))
            {
                P.transform.rotation = glm::radians(rotDeg);
                changed = true;
            }

            ImGui::Separator();
            ImGui::Text("Scale (uniform, around center):");
            changed |= ImGui::SliderFloat("Scale", &P.transform.scale, 0.1f, 3.0f);

            if (ImGui::Button("Reset Transform"))
            {
                P.transform = ExpMapTransform();
                changed = true;
            }

            if (changed)
            {
                UpdatePatchWithTransform(P);
            }
        }

        ImGui::End();

        // ================================
        // UV Viewer
        // ================================
        ImGui::Begin("UV Viewer");
        ImGui::Text("Flattened UV (ExpMap)");

        ImVec2 canvasPos  = ImGui::GetCursorScreenPos();
        ImVec2 canvasSize = ImVec2(400, 400);
        ImVec2 canvasEnd  = ImVec2(canvasPos.x + canvasSize.x,
                                    canvasPos.y + canvasSize.y);

        ImDrawList* draw = ImGui::GetWindowDrawList();
        draw->AddRectFilled(canvasPos, canvasEnd, IM_COL32(30,30,30,255));
        draw->AddRect(canvasPos, canvasEnd, IM_COL32(255,255,255,255));

        if (!g_flattened.empty())
        {
            glm::vec2 size = g_expMapSystem.uvMax - g_expMapSystem.uvMin;
            float scale = std::min(canvasSize.x / size.x, canvasSize.y / size.y) * 0.8f;

            float offsetX = canvasPos.x + canvasSize.x * 0.5f - (g_expMapSystem.uvCenter.x - g_expMapSystem.uvMin.x) * scale;
            float offsetY = canvasPos.y + canvasSize.y * 0.5f - (g_expMapSystem.uvCenter.y - g_expMapSystem.uvMin.y) * scale;

            // Draw center point
            ImVec2 center(
                offsetX + (g_expMapSystem.uvCenter.x - g_expMapSystem.uvMin.x) * scale,
                offsetY + (g_expMapSystem.uvCenter.y - g_expMapSystem.uvMin.y) * scale
            );
            draw->AddCircleFilled(center, 5, IM_COL32(255, 0, 0, 255));

            for (auto& ft : g_flattened)
            {
                auto ToScreen = [&](glm::vec2 uv)
                {
                    return ImVec2(
                        offsetX + (uv.x - g_expMapSystem.uvMin.x) * scale,
                        offsetY + (uv.y - g_expMapSystem.uvMin.y) * scale
                    );
                };

                ImVec2 a = ToScreen(ft.a);
                ImVec2 b = ToScreen(ft.b);
                ImVec2 c = ToScreen(ft.c);

                draw->AddLine(a, b, IM_COL32(255,255,255,255), 1.0f);
                draw->AddLine(b, c, IM_COL32(255,255,255,255), 1.0f);
                draw->AddLine(c, a, IM_COL32(255,255,255,255), 1.0f);
            }
        }

        ImGui::InvisibleButton("canvas", canvasSize);
        ImGui::End();

        // =========================================================
        // Rendering
        // =========================================================
        glViewport(0, 0, SCR_WIDTH, SCR_HEIGHT);
        glClearColor(0.15f, 0.15f, 0.17f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        if (loadedModel)
        {
            float yawRad = glm::radians(orbitYaw);
            float pitRad = glm::radians(orbitPitch);

            glm::vec3 cam(
                targetDistance * cos(pitRad) * sin(yawRad),
                targetDistance * sin(pitRad),
                targetDistance * cos(pitRad) * cos(yawRad)
            );
            cam += targetPoint;

            glm::mat4 view = glm::lookAt(cam, targetPoint, glm::vec3(0,1,0));
            glm::mat4 proj = glm::perspectiveRH_ZO(
                glm::radians(45.0f),
                (float)SCR_WIDTH / SCR_HEIGHT,
                0.1f, 100.0f
            );

            modelShader.use();
            modelShader.setMat4("model", glm::mat4(1.0f));
            modelShader.setMat4("view", view);
            modelShader.setMat4("projection", proj);
            modelShader.setVec3("viewPos", cam);
            loadedModel->Draw(modelShader);

            if (showTriangles)
            {
                glEnable(GL_POLYGON_OFFSET_LINE);
                glPolygonOffset(-1.0f, -1.0f);
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

                wireShader.use();
                wireShader.setVec3("lineColor", glm::vec3(1,0,0));
                wireShader.setMat4("model", glm::mat4(1.0f));
                wireShader.setMat4("view", view);
                wireShader.setMat4("projection", proj);

                loadedModel->Draw(wireShader);

                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
                glDisable(GL_POLYGON_OFFSET_LINE);
            }

            // Draw patches (disable depth test to show on top)
            glDisable(GL_DEPTH_TEST);
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            
            // Add small offset to avoid z-fighting
            glEnable(GL_POLYGON_OFFSET_FILL);
            glPolygonOffset(-1.0f, -1.0f);
            
            for (auto& P : g_patches)
            {
                if (P.textureID == 0 || P.vertices.empty())
                    continue;

                patchShader.use();
                patchShader.setMat4("model", glm::mat4(1.0f));
                patchShader.setMat4("view", view);
                patchShader.setMat4("projection", proj);

                patchShader.setInt("useTexture", 1);
                patchShader.setInt("patchTex", 0);

                glActiveTexture(GL_TEXTURE0);
                glBindTexture(GL_TEXTURE_2D, P.textureID);

                glBindVertexArray(P.vao);
                glDrawArrays(GL_TRIANGLES, 0, P.vertices.size());
            }
            
            glDisable(GL_POLYGON_OFFSET_FILL);
            glDisable(GL_BLEND);
            glEnable(GL_DEPTH_TEST);

            // Draw selection highlight
            if (!g_selected.empty())
            {
                glDisable(GL_DEPTH_TEST);

                patchShader.use();
                patchShader.setInt("useTexture", 0);
                patchShader.setVec3("overrideColor", glm::vec3(0.1f, 1.0f, 0.1f));

                patchShader.setMat4("model", glm::mat4(1.0f));
                patchShader.setMat4("view", view);
                patchShader.setMat4("projection", proj);

                glBindVertexArray(highlightVAO);
                glDrawArrays(GL_TRIANGLES, 0, g_selected.size() * 3);

                glEnable(GL_DEPTH_TEST);
            }
        }

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(win);
    }

    delete loadedModel;

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwTerminate();
    return 0;
}