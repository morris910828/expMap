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

// ============================
// Global State
// ============================
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

// orbit camera
bool leftMouseDown = false;
float orbitYaw = 0.0f, orbitPitch = 0.0f;
float targetDistance = 2.0f;
glm::vec3 targetPoint(0.0f);
float lastX = 0, lastY = 0;
bool firstMouse = true;

// debug picking
bool hasDebugPoint = false;
glm::vec3 debugPoint;

// wireframe flag
bool showTriangles = false;

// debug point renderer
unsigned int debugVAO = 0, debugVBO = 0;
Shader* debugShader = nullptr;


// ============================
// Utility: Build a Ray
// ============================
struct Ray { glm::vec3 origin, direction; };

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

// ------------------------------------------------
// Ray-triangle
// ------------------------------------------------
bool RayTri(const Ray& r,
            const glm::vec3& a,
            const glm::vec3& b,
            const glm::vec3& c,
            float& tOut)
{
    const float EPS = 1e-6f;
    glm::vec3 e1 = b - a, e2 = c - a;
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

// ------------------------------------------------
// Raycast Model
// ------------------------------------------------
bool Raycast(Model* model, const Ray& ray, glm::vec3& hitPos)
{
    float closest = FLT_MAX;
    bool hit = false;

    for (auto& mesh : model->meshes)
    {
        for (int i = 0; i < mesh.indices.size(); i += 3)
        {
            auto& v0 = mesh.vertices[ mesh.indices[i] ].Position;
            auto& v1 = mesh.vertices[ mesh.indices[i+1] ].Position;
            auto& v2 = mesh.vertices[ mesh.indices[i+2] ].Position;

            float t;
            if (RayTri(ray, v0, v1, v2, t) && t < closest)
            {
                closest = t;
                hit = true;
            }
        }
    }

    if (hit) hitPos = ray.origin + ray.direction * closest;
    return hit;
}

// ============================
// Debug Point Setup
// ============================
void InitDebugRenderer()
{
    if (debugVAO) return;

    glGenVertexArrays(1, &debugVAO);
    glGenBuffers(1, &debugVBO);

    glBindVertexArray(debugVAO);
    glBindBuffer(GL_ARRAY_BUFFER, debugVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3), nullptr, GL_DYNAMIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), 0);

    glBindVertexArray(0);
}

void DrawDebugPoint(glm::vec3 p,
                    const glm::mat4& view,
                    const glm::mat4& proj)
{
    if (!debugShader) return;

    glBindBuffer(GL_ARRAY_BUFFER, debugVBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(glm::vec3), &p);

    debugShader->use();
    debugShader->setMat4("model", glm::mat4(1.0f));
    debugShader->setMat4("view", view);
    debugShader->setMat4("projection", proj);

    glPointSize(15);
    glBindVertexArray(debugVAO);
    glDrawArrays(GL_POINTS, 0, 1);
    glBindVertexArray(0);
}


// ============================
// Mouse Input
// ============================
void mouse_callback(GLFWwindow*, double x, double y)
{
    if (!leftMouseDown || ImGui::GetIO().WantCaptureMouse) return;

    if (firstMouse) { lastX = x; lastY = y; firstMouse = false; }

    orbitYaw += (x - lastX) * 0.3f;
    orbitPitch += (lastY - y) * 0.3f;

    orbitPitch = glm::clamp(orbitPitch, -89.0f, 89.0f);

    lastX = x; lastY = y;
}

void mouse_button_callback(GLFWwindow* win, int button, int action, int)
{
    if (button != GLFW_MOUSE_BUTTON_LEFT) return;

    if (action == GLFW_PRESS)
    {
        leftMouseDown = true; firstMouse = true;

        if (loadedModel)
        {
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
                                                   (float)SCR_WIDTH / SCR_HEIGHT,
                                                   0.1f, 100.0f);

            Ray ray = ScreenRay(mx, my, view, proj, cam);

            glm::vec3 hit;
            if (Raycast(loadedModel, ray, hit))
            {
                hasDebugPoint = true;
                debugPoint = hit;
            }
        }
    }
    else if (action == GLFW_RELEASE)
        leftMouseDown = false;
}

void scroll_callback(GLFWwindow*, double, double dy)
{
    targetDistance = glm::clamp(targetDistance - (float)dy * 0.5f, 1.0f, 50.0f);
}


// ============================
// Main
// ============================
int main()
{
    // ---------------------------------------------------------
    // GLFW + GL init
    // ---------------------------------------------------------
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* win = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "ExpMap", 0, 0);
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
    Shader shader("../src/shaders/model.vert", "../src/shaders/model.frag");
    Shader wireframeShader("../src/shaders/wireframe.vert", "../src/shaders/wireframe.frag");
    debugShader = new Shader("../src/shaders/debugPoint.vert","../src/shaders/debugPoint.frag");
    InitDebugRenderer();

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

        ImGui::Begin("Model Viewer");

        const char* preview =
            useCustomModel ? customModelPath : modelList[currentModelIndex].c_str();

        if (ImGui::BeginCombo("Model", preview))
        {
            for (int i=0;i<modelList.size();i++)
            {
                bool selected = (!useCustomModel && currentModelIndex == i);
                if (ImGui::Selectable(modelList[i].c_str(), selected))
                {
                    useCustomModel = false;
                    currentModelIndex = i;

                    delete loadedModel;
                    loadedModel = new Model(modelList[i]);
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
            }
        }

        ImGui::Checkbox("Show Triangles", &showTriangles);
        ImGui::End();

        // ---------------- Render ----------------
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

            // normal shaded model
            shader.use();
            shader.setMat4("model", glm::mat4(1.0f));
            shader.setMat4("view", view);
            shader.setMat4("projection", proj);
            loadedModel->Draw(shader);

            // debug point
            if (hasDebugPoint)
                DrawDebugPoint(debugPoint, view, proj);

            // ---------------------------------------------------------
            // PASS 2: Wireframe Overlay
            // ---------------------------------------------------------
            if (showTriangles)
            {
                glEnable(GL_POLYGON_OFFSET_LINE);
                glPolygonOffset(-4.0f, -4.0f);

                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

                wireframeShader.use();
                wireframeShader.setMat4("model", glm::mat4(1.0f));
                wireframeShader.setMat4("view", view);
                wireframeShader.setMat4("projection", proj);

                wireframeShader.setVec3("lineColor", glm::vec3(1.0f, 0.0f, 0.0f));
                wireframeShader.setFloat("intensity", 1.0f);

                loadedModel->Draw(wireframeShader);

                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
                glDisable(GL_POLYGON_OFFSET_LINE);
            }
        }

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(win);
    }

    delete loadedModel;
    delete debugShader;

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwTerminate();
    return 0;
}
