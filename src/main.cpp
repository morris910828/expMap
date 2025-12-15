#include "stb_image.h"
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"

#include "Shader.h"
#include "Model.h"
#include "ExpMap.h"

#include "HighlightRenderer.h"
#include "InteractionSystem.h"

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

// Model list
std::vector<std::string> modelList = {
    "../assets/armadillo.obj",
    "../assets/bear.obj",
    "../assets/dancer.obj"
};

int currentModelIndex = 0;
bool useCustomModel = false;
char customModelPath[256] = "";

// Mouse input states
bool leftMouseDown = false;
bool dragSelecting = false;
bool dragOrbit = false;
bool firstMouse = true;
float lastX = 0, lastY = 0;

bool showTriangles = false;

// ExpMap selection
std::vector<SelectedTri> g_selected;
std::vector<FlattenTri>  g_flattened;
ExpMapSystem g_expMapSystem;

// Patch system --------------------------------------
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
    "../assets/3.png"
};

int currentTextureIndex = 0;

// Systems
HighlightRenderer highlightRenderer;
InteractionSystem interaction;


// ======================================================
// Load Texture
// ======================================================
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


// ======================================================
// Update Patch Transform
// ======================================================
void UpdatePatchWithTransform(Patch& P)
{
    if (P.selectedTris.empty() || P.flattenedUVs.empty()) return;

    std::vector<glm::vec3> newPositions;
    std::vector<glm::vec2> newTexCoords;

    ApplyExpMapTransform(
        loadedModel, P.selectedTris,
        P.flattenedUVs, P.expSystem,
        P.transform, newPositions, newTexCoords
    );

    P.vertices.clear();
    for (size_t i = 0; i < newPositions.size(); i++)
    {
        PatchVertex v;
        v.pos = newPositions[i];
        v.uv = newTexCoords[i];
        P.vertices.push_back(v);
    }

    glBindBuffer(GL_ARRAY_BUFFER, P.vbo);
    glBufferData(
        GL_ARRAY_BUFFER,
        P.vertices.size() * sizeof(PatchVertex),
        P.vertices.data(),
        GL_DYNAMIC_DRAW
    );
}


// ======================================================
// Mouse Move Callback
// ======================================================
void mouse_callback(GLFWwindow*, double x, double y)
{
    if (!leftMouseDown || ImGui::GetIO().WantCaptureMouse)
        return;

    if (dragOrbit)
    {
        if (firstMouse) { lastX = x; lastY = y; firstMouse = false; }

        float dx = (float)(x - lastX);
        float dy = (float)(lastY - y);

        interaction.Orbit(dx * 0.3f, dy * 0.3f);

        lastX = x;
        lastY = y;
    }
    else if (dragSelecting)
    {
        Ray ray = interaction.GenerateRay(
            (float)x, (float)y,
            SCR_WIDTH, SCR_HEIGHT
        );

        HitInfo hit;
        if (interaction.Raycast(loadedModel, ray, hit))
        {
            Mesh& mesh = loadedModel->meshes[hit.meshIndex];

            unsigned int i0 = mesh.indices[hit.triIndex + 0];
            unsigned int i1 = mesh.indices[hit.triIndex + 1];
            unsigned int i2 = mesh.indices[hit.triIndex + 2];

            g_selected.push_back({ hit.meshIndex, i0, i1, i2 });
            highlightRenderer.Update(loadedModel, g_selected);

            g_expMapSystem = ComputeExpMap(
                loadedModel,
                hit.meshIndex,
                g_selected,
                g_flattened
            );
        }
    }
}


// ======================================================
// Mouse Button Callback
// ======================================================
void mouse_button_callback(GLFWwindow* win, int button, int action, int)
{
    if (button != GLFW_MOUSE_BUTTON_LEFT)
        return;

    if (action == GLFW_PRESS)
    {
        leftMouseDown = true;
        firstMouse = true;

        double mx, my;
        glfwGetCursorPos(win, &mx, &my);

        Ray ray = interaction.GenerateRay(
            (float)mx, (float)my,
            SCR_WIDTH, SCR_HEIGHT
        );

        HitInfo hit;
        if (interaction.Raycast(loadedModel, ray, hit))
            dragSelecting = true;
        else
            dragOrbit = true;
    }
    else if (action == GLFW_RELEASE)
    {
        leftMouseDown = false;
        dragSelecting = false;
        dragOrbit = false;
    }
}


// ======================================================
// Scroll Callback
// ======================================================
void scroll_callback(GLFWwindow*, double, double dy)
{
    interaction.Zoom((float)dy * 0.5f);
}


// ======================================================
// MAIN
// ======================================================
int main()
{
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* win = glfwCreateWindow(
        SCR_WIDTH, SCR_HEIGHT,
        "ExpMap Surface Transform",
        nullptr, nullptr
    );
    glfwMakeContextCurrent(win);

    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);

    glfwSetCursorPosCallback(win, mouse_callback);
    glfwSetMouseButtonCallback(win, mouse_button_callback);
    glfwSetScrollCallback(win, scroll_callback);

    // -------------------
    // ImGui
    // -------------------
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(win, true);
    ImGui_ImplOpenGL3_Init("#version 450");
    ImGui::StyleColorsDark();

    Shader modelShader("../src/shaders/model.vert",
                       "../src/shaders/model.frag");
    Shader wireShader("../src/shaders/wireframe.vert",
                      "../src/shaders/wireframe.frag");
    Shader patchShader("../src/shaders/highlight.vert",
                      "../src/shaders/highlight.frag");

    highlightRenderer.Init();

    loadedModel = new Model(modelList[currentModelIndex]);

    // Set camera target
    interaction.SetTarget(glm::vec3(0.0f));

    // -------------------
    // Render Loop
    // -------------------
    while (!glfwWindowShouldClose(win))
    {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
                // ============================
        // UI PANEL
        // ============================
        ImGui::Begin("Model Viewer");

        const char* preview = useCustomModel ?
                              customModelPath :
                              modelList[currentModelIndex].c_str();

        if (ImGui::BeginCombo("Model", preview))
        {
            for (int i = 0; i < modelList.size(); i++)
            {
                bool selected = (!useCustomModel &&
                                  currentModelIndex == i);

                if (ImGui::Selectable(modelList[i].c_str(), selected))
                {
                    useCustomModel = false;
                    currentModelIndex = i;

                    delete loadedModel;
                    loadedModel = new Model(modelList[i]);

                    g_selected.clear();
                    g_flattened.clear();
                    g_patches.clear();

                    highlightRenderer.Update(loadedModel, g_selected);
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

                highlightRenderer.Update(loadedModel, g_selected);
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
            highlightRenderer.Update(loadedModel, g_selected);
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

                P.transform.scale = 1.0f;
                P.transform.rotation = 0.0f;
                P.transform.surfaceOffset = glm::vec2(0.0f);

                std::vector<TriInfo> expandedInfos;
                ExpandExpMap(loadedModel, P.meshIndex,
                             g_selected, P.expSystem,
                             5.0f, expandedInfos);

                Mesh& mesh = loadedModel->meshes[P.meshIndex];

                for (auto& info : expandedInfos)
                {
                    int idx0 = mesh.indices[info.triIdx * 3 + 0];
                    int idx1 = mesh.indices[info.triIdx * 3 + 1];
                    int idx2 = mesh.indices[info.triIdx * 3 + 2];

                    P.selectedTris.push_back({
                        P.meshIndex, (unsigned int)idx0,
                        (unsigned int)idx1, (unsigned int)idx2
                    });

                    FlattenTri ft;
                    ft.a = info.uv0;
                    ft.b = info.uv1;
                    ft.c = info.uv2;
                    ft.triIndex = info.triIdx;
                    P.flattenedUVs.push_back(ft);
                }

                std::vector<glm::vec3> positions;
                std::vector<glm::vec2> texCoords;

                ApplyExpMapTransform(loadedModel, P.selectedTris,
                                     P.flattenedUVs, P.expSystem,
                                     P.transform, positions, texCoords);

                P.vertices.clear();
                for (size_t i = 0; i < positions.size(); i++)
                {
                    PatchVertex v;
                    v.pos = positions[i];
                    v.uv = texCoords[i];
                    P.vertices.push_back(v);
                }

                glGenVertexArrays(1, &P.vao);
                glGenBuffers(1, &P.vbo);

                glBindVertexArray(P.vao);
                glBindBuffer(GL_ARRAY_BUFFER, P.vbo);

                glBufferData(GL_ARRAY_BUFFER,
                    P.vertices.size() * sizeof(PatchVertex),
                    P.vertices.data(), GL_DYNAMIC_DRAW);

                glEnableVertexAttribArray(0);
                glVertexAttribPointer(
                    0, 3, GL_FLOAT, GL_FALSE,
                    sizeof(PatchVertex),
                    (void*)offsetof(PatchVertex, pos)
                );

                glEnableVertexAttribArray(1);
                glVertexAttribPointer(
                    1, 2, GL_FLOAT, GL_FALSE,
                    sizeof(PatchVertex),
                    (void*)offsetof(PatchVertex, uv)
                );

                glBindVertexArray(0);

                g_patches.push_back(P);

                g_selected.clear();
                g_flattened.clear();

                highlightRenderer.Update(loadedModel, g_selected);
            }
        }

        // ========== Transform Controls ================
        ImGui::Separator();
        ImGui::Text("Texture Transform (Surface-based)");

        if (ImGui::BeginCombo(
            "Select Patch",
            selectedPatchIndex >= 0 ?
                ("Patch " + std::to_string(selectedPatchIndex)).c_str() :
                "None"
        ))
        {
            for (int i = 0; i < g_patches.size(); i++)
            {
                bool selected = (selectedPatchIndex == i);
                if (ImGui::Selectable(("Patch " + std::to_string(i)).c_str(), selected))
                    selectedPatchIndex = i;

                if (selected)
                    ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }

        if (selectedPatchIndex >= 0 &&
            selectedPatchIndex < g_patches.size())
        {
            Patch& P = g_patches[selectedPatchIndex];
            bool changed = false;

            ImGui::Text("Surface Translation:");
            changed |= ImGui::SliderFloat("Move U", &P.transform.surfaceOffset.x, -1.0f, 1.0f);
            changed |= ImGui::SliderFloat("Move V", &P.transform.surfaceOffset.y, -1.0f, 1.0f);

            ImGui::Separator();
            ImGui::Text("Rotation:");
            float rotDeg = glm::degrees(P.transform.rotation);
            if (ImGui::SliderFloat("Angle", &rotDeg, -180.0f, 180.0f))
            {
                P.transform.rotation = glm::radians(rotDeg);
                changed = true;
            }

            ImGui::Separator();
            ImGui::Text("Scale:");
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

        ImGui::End(); // End main panel


        // ========== UV Viewer ==================================
        ImGui::Begin("UV Viewer");
        ImGui::Text("Flattened UV (ExpMap)");

        ImVec2 canvasPos = ImGui::GetCursorScreenPos();
        ImVec2 canvasSize(400, 400);
        ImVec2 canvasEnd(canvasPos.x + canvasSize.x, canvasPos.y + canvasSize.y);

        ImDrawList* draw = ImGui::GetWindowDrawList();
        draw->AddRectFilled(canvasPos, canvasEnd, IM_COL32(30,30,30,255));
        draw->AddRect(canvasPos, canvasEnd, IM_COL32(255,255,255,255));

        if (!g_flattened.empty())
        {
            glm::vec2 size = g_expMapSystem.uvMax - g_expMapSystem.uvMin;
            float scale = std::min(canvasSize.x / size.x,
                                   canvasSize.y / size.y) * 0.8f;

            float offsetX = canvasPos.x + canvasSize.x * 0.5f -
                (g_expMapSystem.uvCenter.x - g_expMapSystem.uvMin.x) * scale;
            float offsetY = canvasPos.y + canvasSize.y * 0.5f -
                (g_expMapSystem.uvCenter.y - g_expMapSystem.uvMin.y) * scale;

            ImVec2 center(
                offsetX + (g_expMapSystem.uvCenter.x - g_expMapSystem.uvMin.x) * scale,
                offsetY + (g_expMapSystem.uvCenter.y - g_expMapSystem.uvMin.y) * scale
            );

            draw->AddCircleFilled(center, 5, IM_COL32(255,0,0,255));

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

        ImGui::InvisibleButton("uv_canvas", canvasSize);
        ImGui::End();


        // ==================================================
        // Rendering
        // ==================================================
        glViewport(0, 0, SCR_WIDTH, SCR_HEIGHT);
        glClearColor(0.15f, 0.15f, 0.17f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        if (loadedModel)
        {
            glm::mat4 view = interaction.GetViewMatrix();
            glm::mat4 proj = glm::perspectiveRH_ZO(
                glm::radians(45.0f),
                (float)SCR_WIDTH / SCR_HEIGHT,
                0.1f, 100.0f
            );

            glm::vec3 camPos = interaction.GetCameraPos();

            // ----------- Draw model -------------
            modelShader.use();
            modelShader.setMat4("model", glm::mat4(1.0f));
            modelShader.setMat4("view", view);
            modelShader.setMat4("projection", proj);
            modelShader.setVec3("viewPos", camPos);
            loadedModel->Draw(modelShader);

            // ----------- Draw wireframe ----------
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

            // ----------- Draw patches ------------
            glDisable(GL_DEPTH_TEST);
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

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

            // ------------- Draw highlight (selection) --------------
            highlightRenderer.Draw(patchShader, view, proj);
        }

        // ==============================
        // Render ImGui
        // ==============================
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        // Swap buffers
        glfwSwapBuffers(win);
    }

    // Cleanup
    delete loadedModel;

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwTerminate();
    return 0;
}
