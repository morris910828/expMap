#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"

#include "Camera.h"
#include "Shader.h"
#include "Model.h"
#include "ExpMap.h"    // ⭐ 新增：ExpMap 計算核心

#include <iostream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/ext/matrix_clip_space.hpp>
#include <vector>

// =====================================================
// Global
// =====================================================
int SCR_WIDTH = 1280, SCR_HEIGHT = 720;

Model* loadedModel = nullptr;
std::vector<std::string> modelList = {
    "../assets/armadillo.obj",
    "../assets/bear.obj",
    "../assets/dancer.obj",
};

std::vector<std::string> textureList = {
    "../assets/t1.png",
    "../assets/t2.png",
};
int currentTextureIndex = 0;

int currentModelIndex = 0;
bool useCustomModel = false;
char customModelPath[256] = "";

// Orbit camera
bool leftMouseDown = false;
float orbitYaw = 0.0f, orbitPitch = 0.0f;
float targetDistance = 2.0f;
glm::vec3 targetPoint(0.0f);
float lastX = 0, lastY = 0;
bool firstMouse = true;

bool showTriangles = false;
bool dragSelecting = false;
bool dragOrbit = false;

// Selected triangles (for highlight + UV viewer)
std::vector<SelectedTri> g_selected;

// Flattened triangles from ExpMap
std::vector<FlattenTri> g_flattened;

// Highlight VAO/VBO
unsigned int highlightVAO = 0;
unsigned int highlightVBO = 0;


// =====================================================
// Ray picking structures
// =====================================================
struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
};

struct HitInfo {
    bool hit = false;
    glm::vec3 hitPos;
    int meshIndex = -1;
    int triIndex = -1;
};

// =====================================================
// Ray helpers
// =====================================================
Ray ScreenRay(float mx, float my,
              const glm::mat4& view,
              const glm::mat4& proj,
              const glm::vec3& cam)
{
    float x = (2.0f * mx) / SCR_WIDTH - 1.0f;
    float y = 1.0f - (2.0f * my) / SCR_HEIGHT;

    glm::vec4 clip(x, y, -1.0f, 1.0f);
    glm::vec4 eye = glm::inverse(proj) * clip;
    eye = glm::vec4(eye.x, eye.y, -1.0, 0.0);

    glm::vec3 worldDir = glm::normalize(glm::vec3(glm::inverse(view) * eye));
    return { cam, worldDir };
}

bool RayTri(const Ray& r,
            const glm::vec3& a,
            const glm::vec3& b,
            const glm::vec3& c,
            float& tOut)
{
    const float EPS = 1e-6f;

    glm::vec3 e1 = b - a;
    glm::vec3 e2 = c - a;

    glm::vec3 p = glm::cross(r.direction, e2);
    float det = glm::dot(e1, p);
    if (fabs(det) < EPS) return false;

    float inv = 1.0f / det;

    glm::vec3 s = r.origin - a;
    float u = glm::dot(s, p) * inv;
    if (u < 0 || u > 1) return false;

    glm::vec3 q = glm::cross(s, e1);
    float v = glm::dot(r.direction, q) * inv;
    if (v < 0 || u + v > 1) return false;

    float t = glm::dot(e2, q) * inv;
    if (t > EPS) { tOut = t; return true; }

    return false;
}

bool Raycast(Model* model, const Ray& ray, HitInfo& out)
{
    float closest = FLT_MAX;

    for (int m = 0; m < model->meshes.size(); m++)
    {
        Mesh& mesh = model->meshes[m];

        for (int f = 0; f < mesh.indices.size(); f += 3)
        {
            glm::vec3 v0 = mesh.vertices[mesh.indices[f]].Position;
            glm::vec3 v1 = mesh.vertices[mesh.indices[f + 1]].Position;
            glm::vec3 v2 = mesh.vertices[mesh.indices[f + 2]].Position;

            float t;
            if (RayTri(ray, v0, v1, v2, t))
            {
                if (t < closest)
                {
                    closest = t;
                    out.hit = true;
                    out.hitPos = ray.origin + ray.direction * t;
                    out.meshIndex = m;
                    out.triIndex = f;
                }
            }
        }
    }
    return out.hit;
}


//texture
unsigned int LoadTexture(const std::string& path)
{
    int w, h, ch;
    stbi_set_flip_vertically_on_load(true);
    unsigned char* data = stbi_load(path.c_str(), &w, &h, &ch, 0);

    if (!data) {
        std::cout << "Failed to load texture: " << path << std::endl;
        return 0;
    }

    GLenum format = (ch == 4 ? GL_RGBA : GL_RGB);

    unsigned int tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, format, w, h, 0, format, GL_UNSIGNED_BYTE, data);
    glGenerateMipmap(GL_TEXTURE_2D);

    stbi_image_free(data);
    return tex;
}


// =====================================================
// Highlight system
// =====================================================
void InitHighlightBuffers()
{
    glGenVertexArrays(1, &highlightVAO);
    glGenBuffers(1, &highlightVBO);

    glBindVertexArray(highlightVAO);
    glBindBuffer(GL_ARRAY_BUFFER, highlightVBO);
    glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE,
                          sizeof(glm::vec3), 0);

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
    glBufferData(GL_ARRAY_BUFFER,
                 verts.size() * sizeof(glm::vec3),
                 verts.data(),
                 GL_DYNAMIC_DRAW);
}

void DrawHighlight(Shader& fillShader,
                   const glm::mat4& view,
                   const glm::mat4& proj)
{
    if (g_selected.empty()) return;

    fillShader.use();
    fillShader.setMat4("model", glm::mat4(1.0f));
    fillShader.setMat4("view", view);
    fillShader.setMat4("projection", proj);

    glEnable(GL_POLYGON_OFFSET_FILL);
    glPolygonOffset(-2.0f, -2.0f);
    glBindVertexArray(highlightVAO);
    glDrawArrays(GL_TRIANGLES, 0, g_selected.size() * 3);
    glDisable(GL_POLYGON_OFFSET_FILL);
}


// =====================================================
// Utility: check triangle duplication
// =====================================================
bool IsTriAlreadySelected(int mesh,
                          unsigned int i0,
                          unsigned int i1,
                          unsigned int i2,
                          int& outIndex)
{
    for (int s = 0; s < g_selected.size(); s++)
    {
        auto& t = g_selected[s];
        if (t.meshIndex == mesh &&
            t.i0 == i0 &&
            t.i1 == i1 &&
            t.i2 == i2)
        {
            outIndex = s;
            return true;
        }
    }
    return false;
}


// =====================================================
// Mouse callback
// =====================================================
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
        // Continuous brush selection
        double mx = x, my = y;

        float yawR = glm::radians(orbitYaw);
        float pitR = glm::radians(orbitPitch);

        glm::vec3 cam(
            targetDistance * cos(pitR) * sin(yawR),
            targetDistance * sin(pitR),
            targetDistance * cos(pitR) * cos(yawR)
        );
        cam += targetPoint;

        glm::mat4 view = glm::lookAt(cam, targetPoint,
                                     glm::vec3(0,1,0));
        glm::mat4 proj = glm::perspectiveRH_ZO(
            glm::radians(45.0f),
            (float)SCR_WIDTH / SCR_HEIGHT,
            0.1f, 100.0f);

        Ray ray = ScreenRay(mx, my, view, proj, cam);

        HitInfo hit;
        if (Raycast(loadedModel, ray, hit))
        {
            Mesh& mesh = loadedModel->meshes[hit.meshIndex];

            unsigned int i0 = mesh.indices[hit.triIndex];
            unsigned int i1 = mesh.indices[hit.triIndex + 1];
            unsigned int i2 = mesh.indices[hit.triIndex + 2];

            int exist = -1;

            if (!IsTriAlreadySelected(hit.meshIndex, i0, i1, i2, exist))
            {
                g_selected.push_back({hit.meshIndex, i0, i1, i2});
                UpdateHighlightBuffer(loadedModel);

                // ⭐ ExpMap 重新展開
                ComputeExpMap(loadedModel,
                              hit.meshIndex,
                              g_selected,
                              g_flattened);
            }
        }
    }
}


// =====================================================
// Mouse button (click → select or orbit)
// =====================================================
void mouse_button_callback(GLFWwindow* win, int button, int action, int)
{
    if (button != GLFW_MOUSE_BUTTON_LEFT)
        return;

    if (action == GLFW_PRESS)
    {
        double mx, my;
        glfwGetCursorPos(win, &mx, &my);

        leftMouseDown = true;
        firstMouse = true;
        dragSelecting = false;
        dragOrbit = false;

        if (!loadedModel) return;

        float yawR = glm::radians(orbitYaw);
        float pitR = glm::radians(orbitPitch);

        glm::vec3 cam(
            targetDistance * cos(pitR) * sin(yawR),
            targetDistance * sin(pitR),
            targetDistance * cos(pitR) * cos(yawR)
        );
        cam += targetPoint;

        glm::mat4 view = glm::lookAt(cam, targetPoint,
                                     glm::vec3(0,1,0));
        glm::mat4 proj = glm::perspectiveRH_ZO(
            glm::radians(45.0f),
            (float)SCR_WIDTH / SCR_HEIGHT,
            0.1f,
            100.0f);

        Ray ray = ScreenRay(mx, my, view, proj, cam);

        HitInfo hit;
        if (Raycast(loadedModel, ray, hit))
        {
            dragSelecting = true;
        }
        else
        {
            dragOrbit = true;
        }
    }
    else if (action == GLFW_RELEASE)
    {
        leftMouseDown = false;
        dragSelecting = false;
        dragOrbit = false;
    }
}


// =====================================================
// Scroll zoom
// =====================================================
void scroll_callback(GLFWwindow*, double, double dy)
{
    targetDistance = glm::clamp(
        targetDistance - (float)dy * 0.5f,
        1.0f,
        50.0f
    );
}


// =====================================================
// MAIN
// =====================================================
int main()
{
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,5);
    glfwWindowHint(GLFW_OPENGL_PROFILE,
                   GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* win =
        glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT,
                         "expmap", 0, 0);
    glfwMakeContextCurrent(win);

    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
    glEnable(GL_DEPTH_TEST);

    glfwSetCursorPosCallback(win, mouse_callback);
    glfwSetMouseButtonCallback(win, mouse_button_callback);
    glfwSetScrollCallback(win, scroll_callback);

    // ---------------------------------------------------------
    // ImGui
    // ---------------------------------------------------------
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(win, true);
    ImGui_ImplOpenGL3_Init("#version 450");
    ImGui::StyleColorsDark();

    // ---------------------------------------------------------
    // Shaders
    // ---------------------------------------------------------
    Shader modelShader("../src/shaders/model.vert",
                       "../src/shaders/model.frag");

    Shader wireShader("../src/shaders/wireframe.vert",
                      "../src/shaders/wireframe.frag");

    Shader highlightShader("../src/shaders/highlight.vert",
                           "../src/shaders/highlight.frag");

    InitHighlightBuffers();

    loadedModel = new Model(modelList[currentModelIndex]);

    // =====================================================
    // LOOP
    // =====================================================
    while (!glfwWindowShouldClose(win))
    {
        glfwPollEvents();

        // ---------------- GUI Frame ----------------
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // =====================================================
        // Model Viewer UI
        // =====================================================
        ImGui::Begin("Model Viewer");

        const char* preview =
            useCustomModel ?
            customModelPath :
            modelList[currentModelIndex].c_str();

        if (ImGui::BeginCombo("Model", preview))
        {
            for (int i = 0; i < modelList.size(); i++)
            {
                bool selected =
                    (!useCustomModel &&
                     currentModelIndex == i);

                if (ImGui::Selectable(modelList[i].c_str(),
                                      selected))
                {
                    useCustomModel = false;
                    currentModelIndex = i;

                    delete loadedModel;
                    loadedModel = new Model(modelList[i]);

                    g_selected.clear();
                    g_flattened.clear();
                    UpdateHighlightBuffer(loadedModel);
                }
                if (selected) ImGui::SetItemDefaultFocus();
            }

            ImGui::Separator();
            if (ImGui::Selectable("Custom Path",
                                  useCustomModel))
                useCustomModel = true;

            ImGui::EndCombo();
        }

        if (useCustomModel)
        {
            ImGui::InputText("Path", customModelPath,
                             sizeof(customModelPath));
            if (ImGui::Button("Load"))
            {
                delete loadedModel;
                loadedModel = new Model(customModelPath);

                g_selected.clear();
                g_flattened.clear();
                UpdateHighlightBuffer(loadedModel);
            }
        }

        ImGui::Checkbox("Show Triangles", &showTriangles);

        // ============================
        // Texture Selector UI
        // ============================
        ImGui::Separator();
        ImGui::Text("Texture");

        const char* texPreview = textureList[currentTextureIndex].c_str();

        if (ImGui::BeginCombo("Select Texture", texPreview))
        {
            for (int i = 0; i < textureList.size(); i++)
            {
                bool selected = (currentTextureIndex == i);

                if (ImGui::Selectable(textureList[i].c_str(), selected))
                {
                    currentTextureIndex = i;

                    // 只是載入，不會自動套用
                    unsigned int texID = LoadTexture(textureList[i]);
                    std::cout << "Loaded texture: " << textureList[i]
                            << " (ID=" << texID << ")" << std::endl;
                }

                if (selected)
                    ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }

        ImGui::End();



        // =====================================================
        // ExpMap Flatten Viewer
        // =====================================================
        ImGui::Begin("ExpMap Flattened Patch");

        static std::vector<FlattenTri> flat;
        ComputeExpMap(loadedModel, 0, g_selected, flat);  // meshIndex=0，你可改成 hit.meshIndex

        // 畫布大小
        ImVec2 canvasSize = ImGui::GetContentRegionAvail();
        ImVec2 origin = ImGui::GetCursorScreenPos();
        ImDrawList* draw = ImGui::GetWindowDrawList();

        // ------------------------------------------------------------
        // Compute bounding box of all flattened triangles
        // ------------------------------------------------------------
        if (!flat.empty())
        {
            float minX = FLT_MAX, minY = FLT_MAX;
            float maxX = -FLT_MAX, maxY = -FLT_MAX;

            for (auto& t : flat)
            {
                minX = std::min({minX, t.a.x, t.b.x, t.c.x});
                minY = std::min({minY, t.a.y, t.b.y, t.c.y});
                maxX = std::max({maxX, t.a.x, t.b.x, t.c.x});
                maxY = std::max({maxY, t.a.y, t.b.y, t.c.y});
            }

            float w = maxX - minX;
            float h = maxY - minY;

            // scale 讓 patch 填滿畫布
            float scale = 0.9f * std::min(canvasSize.x / w, canvasSize.y / h);

            // ------------------------------------------------------------
            // 畫所有三角形（自動置中 & 放大）
            // ------------------------------------------------------------
            auto ToScreen = [&](glm::vec2 v)
            {
                float x = (v.x - minX) * scale + origin.x + (canvasSize.x - w * scale) * 0.5f;
                float y = (v.y - minY) * scale + origin.y + (canvasSize.y - h * scale) * 0.5f;
                // 注意：ImGui Y 座標向下，因此不需要 invert
                return ImVec2(x, y);
            };

            ImU32 col = IM_COL32(255, 255, 0, 255);

            for (auto& t : flat)
            {
                ImVec2 p0 = ToScreen(t.a);
                ImVec2 p1 = ToScreen(t.b);
                ImVec2 p2 = ToScreen(t.c);

                draw->AddTriangle(p0, p1, p2, col, 2.5f);
            }
        }

        ImGui::Dummy(canvasSize); // 占位，讓視窗正確呈現
        ImGui::End();



        // =====================================================
        // Render Scene
        // =====================================================
        glViewport(0,0,SCR_WIDTH,SCR_HEIGHT);
        glClearColor(0.15f, 0.15f, 0.17f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT |
                GL_DEPTH_BUFFER_BIT);

        if (loadedModel)
        {
            float yawR = glm::radians(orbitYaw);
            float pitR = glm::radians(orbitPitch);

            glm::vec3 cam(
                targetDistance *
                    cos(pitR) * sin(yawR),
                targetDistance *
                    sin(pitR),
                targetDistance *
                    cos(pitR) * cos(yawR)
            );
            cam += targetPoint;

            glm::mat4 view = glm::lookAt( cam, targetPoint, glm::vec3(0,1,0) );

            glm::mat4 proj = glm::perspectiveRH_ZO( glm::radians(45.0f), (float)SCR_WIDTH / SCR_HEIGHT, 0.1f, 100.0f );

            // Main model
            modelShader.use();
            modelShader.setMat4("model", glm::mat4(1.0f));
            modelShader.setMat4("view", view);
            modelShader.setMat4("projection", proj);
            loadedModel->Draw(modelShader);

            // Wireframe
            if (showTriangles)
            {
                glEnable(GL_POLYGON_OFFSET_LINE);
                glPolygonOffset(-1.0f, -1.0f);
                glPolygonMode(GL_FRONT_AND_BACK,
                              GL_LINE);

                wireShader.use();
                wireShader.setVec3("lineColor", glm::vec3(1,0,0));
                wireShader.setMat4("model", glm::mat4(1.0f));
                wireShader.setMat4("view", view);
                wireShader.setMat4("projection", proj);

                loadedModel->Draw(wireShader);

                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
                glDisable(GL_POLYGON_OFFSET_LINE);
            }

            // highlight selected triangles
            DrawHighlight(highlightShader,
                          view, proj);
        }

        // End Draw ImGui
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(
            ImGui::GetDrawData()
        );
        glfwSwapBuffers(win);
    }

    delete loadedModel;

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwTerminate();
    return 0;
}
