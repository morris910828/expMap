#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"

#include "Camera.h"
#include "Shader.h"
#include "Model.h"

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

// =====================================================
// Ray picking
// =====================================================
struct Ray { glm::vec3 origin, direction; };

struct HitInfo {
    bool hit = false;
    glm::vec3 hitPos;
    int meshIndex = -1;
    int triIndex = -1;    // f (0-based)
};

// UV Viewer
struct UVTriangle {
    glm::vec2 uv0, uv1, uv2;
};
std::vector<UVTriangle> g_uvTriangles;

// Selected triangles
struct SelectedTri {
    int meshIndex;
    unsigned int i0, i1, i2;
};
std::vector<SelectedTri> g_selected;

// Highlight
unsigned int highlightVAO = 0;
unsigned int highlightVBO = 0;

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
            glm::vec3 v0 = mesh.vertices[ mesh.indices[f] ].Position;
            glm::vec3 v1 = mesh.vertices[ mesh.indices[f+1] ].Position;
            glm::vec3 v2 = mesh.vertices[ mesh.indices[f+2] ].Position;

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
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), 0);

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

void DrawHighlight(Shader& fillShader, Shader& lineShader, 
                   const glm::mat4& view, const glm::mat4& proj)
{
    if (g_selected.empty()) return;

    // -------------------------------
    // 1. 填滿 highlight (綠色)
    // -------------------------------
    fillShader.use();
    fillShader.setMat4("model", glm::mat4(1.0f));
    fillShader.setMat4("view", view);
    fillShader.setMat4("projection", proj);

    glEnable(GL_POLYGON_OFFSET_FILL);
    glPolygonOffset(-2.0f, -2.0f);
    glBindVertexArray(highlightVAO);
    glDrawArrays(GL_TRIANGLES, 0, (GLsizei)g_selected.size() * 3);
    glDisable(GL_POLYGON_OFFSET_FILL);

    // -------------------------------
    // 2. outline 使用「紅色 shader」
    // -------------------------------
    lineShader.use();
    lineShader.setMat4("model", glm::mat4(1.0f));
    lineShader.setMat4("view", view);
    lineShader.setMat4("projection", proj);
    lineShader.setVec3("lineColor", glm::vec3(1,0,0));

    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glLineWidth(1.0f);
    glDepthFunc(GL_ALWAYS);

    glBindVertexArray(highlightVAO);
    glDrawArrays(GL_TRIANGLES, 0, (GLsizei)g_selected.size() * 3);

    // restore
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glDepthFunc(GL_LESS);
}



// =====================================================
// Utilities
// =====================================================
void mouse_callback(GLFWwindow*, double x, double y)
{
    if (!leftMouseDown || ImGui::GetIO().WantCaptureMouse) return;

    if (firstMouse) { lastX = x; lastY = y; firstMouse = false; }

    orbitYaw += (x - lastX) * 0.3f;
    orbitPitch += (lastY - y) * 0.3f;

    orbitPitch = glm::clamp(orbitPitch, -89.0f, 89.0f);

    lastX = x; lastY = y;
}

bool IsTriAlreadySelected(int mesh, unsigned int i0, unsigned int i1, unsigned int i2, int& outIndex)
{
    for (int s = 0; s < g_selected.size(); s++)
    {
        auto& t = g_selected[s];
        if (t.meshIndex == mesh &&
            t.i0 == i0 && t.i1 == i1 && t.i2 == i2)
        {
            outIndex = s;
            return true;
        }
    }
    return false;
}

void mouse_button_callback(GLFWwindow* win, int button, int action, int)
{
    if (button != GLFW_MOUSE_BUTTON_LEFT) return;

    if (action == GLFW_PRESS)
    {
        leftMouseDown = true;
        firstMouse = true;

        if (!loadedModel) return;

        double mx, my;
        glfwGetCursorPos(win, &mx, &my);

        float yawR = glm::radians(orbitYaw);
        float pitR = glm::radians(orbitPitch);

        glm::vec3 cam(
            targetDistance * cos(pitR) * sin(yawR),
            targetDistance * sin(pitR),
            targetDistance * cos(pitR) * cos(yawR)
        );
        cam += targetPoint;

        glm::mat4 view = glm::lookAt(cam, targetPoint, glm::vec3(0,1,0));
        glm::mat4 proj = glm::perspectiveRH_ZO(glm::radians(45.0f),
                                               (float)SCR_WIDTH/SCR_HEIGHT,
                                               0.1f, 100.0f);

        Ray ray = ScreenRay(mx, my, view, proj, cam);

        HitInfo hit;
        if (Raycast(loadedModel, ray, hit))
        {
            Mesh& mesh = loadedModel->meshes[hit.meshIndex];

            unsigned int i0 = mesh.indices[ hit.triIndex ];
            unsigned int i1 = mesh.indices[ hit.triIndex + 1 ];
            unsigned int i2 = mesh.indices[ hit.triIndex + 2 ];

            int removeIndex = -1;

            if (IsTriAlreadySelected(hit.meshIndex, i0, i1, i2, removeIndex))
            {
                g_selected.erase(g_selected.begin() + removeIndex);
                g_uvTriangles.erase(g_uvTriangles.begin() + removeIndex);
            }
            else
            {
                g_selected.push_back({ hit.meshIndex, i0, i1, i2 });

                UVTriangle uv;
                uv.uv0 = mesh.vertices[i0].TexCoords;
                uv.uv1 = mesh.vertices[i1].TexCoords;
                uv.uv2 = mesh.vertices[i2].TexCoords;
                g_uvTriangles.push_back(uv);
            }

            UpdateHighlightBuffer(loadedModel);
        }
    }
    else if (action == GLFW_RELEASE)
    {
        leftMouseDown = false;
    }
}

void scroll_callback(GLFWwindow*, double, double dy)
{
    targetDistance = glm::clamp(targetDistance - (float)dy * 0.5f, 1.0f, 50.0f);
}

// =====================================================
// MAIN
// =====================================================
int main()
{
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* win = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "expmap", 0, 0);
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
    Shader modelShader("../src/shaders/model.vert", "../src/shaders/model.frag");
    Shader wireShader ("../src/shaders/wireframe.vert","../src/shaders/wireframe.frag");
    Shader highlightShader("../src/shaders/highlight.vert","../src/shaders/highlight.frag");

    InitHighlightBuffers();

    loadedModel = new Model(modelList[currentModelIndex]);

    // ---------------------------------------------------------
    // Loop
    // ---------------------------------------------------------
    while (!glfwWindowShouldClose(win))
    {
        glfwPollEvents();

        // ---------------- GUI frame ----------------
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // =====================================================
        // Model Viewer UI
        // =====================================================
        ImGui::Begin("Model Viewer");

        const char* preview =
            useCustomModel ? customModelPath : modelList[currentModelIndex].c_str();

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
                    g_uvTriangles.clear();
                    UpdateHighlightBuffer(loadedModel);
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
                g_uvTriangles.clear();
                UpdateHighlightBuffer(loadedModel);
            }
        }

        ImGui::Checkbox("Show Triangles", &showTriangles);
        ImGui::End();


        // =====================================================
        // Selected Triangle UVs
        // =====================================================
        ImGui::Begin("Selected Triangle UVs");

        ImGui::Text("Selected: %d", (int)g_uvTriangles.size());
        ImGui::Separator();

        float scale = 200.0f;

        for (int idx = 0; idx < g_uvTriangles.size(); idx++)
        {
            const UVTriangle& tri = g_uvTriangles[idx];

            ImGui::Text("Triangle %d", idx);
            ImGui::Text("UV0: (%.3f, %.3f)", tri.uv0.x, tri.uv0.y);
            ImGui::Text("UV1: (%.3f, %.3f)", tri.uv1.x, tri.uv1.y);
            ImGui::Text("UV2: (%.3f, %.3f)", tri.uv2.x, tri.uv2.y);

            ImVec2 wp = ImGui::GetCursorScreenPos();
            ImDrawList* draw = ImGui::GetWindowDrawList();

            auto toScreen = [&](glm::vec2 uv)
            {
                return ImVec2(
                    wp.x + uv.x * scale,
                    wp.y + (1 - uv.y) * scale
                );
            };

            ImVec2 p0 = toScreen(tri.uv0);
            ImVec2 p1 = toScreen(tri.uv1);
            ImVec2 p2 = toScreen(tri.uv2);

            draw->AddLine(p0, p1, IM_COL32_WHITE, 2.0f);
            draw->AddLine(p1, p2, IM_COL32_WHITE, 2.0f);
            draw->AddLine(p2, p0, IM_COL32_WHITE, 2.0f);

            ImGui::Dummy(ImVec2(scale, scale));
            ImGui::Separator();
        }

        ImGui::End();


        // =====================================================
        // Render Scene
        // =====================================================
        glViewport(0,0,SCR_WIDTH,SCR_HEIGHT);
        glClearColor(0.15f,0.15f,0.17f,1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        if (loadedModel)
        {
            float yawR = glm::radians(orbitYaw);
            float pitR = glm::radians(orbitPitch);

            glm::vec3 cam(
                targetDistance * cos(pitR) * sin(yawR),
                targetDistance * sin(pitR),
                targetDistance * cos(pitR) * cos(yawR)
            );
            cam += targetPoint;

            glm::mat4 view = glm::lookAt(cam, targetPoint, glm::vec3(0,1,0));
            glm::mat4 proj = glm::perspectiveRH_ZO(glm::radians(45.0f),
                                                   (float)SCR_WIDTH/SCR_HEIGHT,
                                                   0.1f, 100.0f);

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
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

                wireShader.use();
                wireShader.setMat4("model", glm::mat4(1.0f));
                wireShader.setMat4("view", view);
                wireShader.setMat4("projection", proj);
                wireShader.setVec3("lineColor", glm::vec3(1,0,0));

                loadedModel->Draw(wireShader);

                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
                glDisable(GL_POLYGON_OFFSET_LINE);
            }

            // Highlight triangles
            DrawHighlight(highlightShader, wireShader, view, proj);

        }

        // End ImGui
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
