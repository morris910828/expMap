#include <glad/glad.h>
#include <GLFW/glfw3.h>

// [關鍵] 引入檔案系統 (需 C++17 標準)
#include <filesystem>
namespace fs = std::filesystem;

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <iostream>
#include <vector>
#include <string>
#include <algorithm> // for std::transform

// 專案標頭檔
#include "Shader.h"
#include "Model.h"
#include "InteractionSystem.h"
#include "SelectionSystem.h"   
#include "HighlightRenderer.h" 
#include "DecalSystem.h"       

// --- 全域變數 ---
const unsigned int SCR_WIDTH = 1400;
const unsigned int SCR_HEIGHT = 900;

InteractionSystem interaction;
SelectionSystem selectionSystem;
HighlightRenderer* highlightRenderer = nullptr; 
DecalSystem* decalSystem = nullptr;             
Model* loadedModel = nullptr;

// 模型列表 (自動掃描)
std::vector<std::string> modelList;
int currentModelIndex = 0;
// 自定義路徑緩衝區
char customModelPath[256] = "../assets/";

// --- 貼圖資源 ---
struct TextureAsset {
    std::string name;
    std::string path;
    unsigned int id;
    int width;  
    int height; 
};
std::vector<TextureAsset> textureList;
int currentTextureIndex = 0;

// --- 編輯狀態 ---
std::vector<int> currentSelectionIndices;   
std::vector<FlatVertex> currentExpMapUVs;   
float brushRadius = 0.05f;
int lastHitTriangle = -1;
int editingDecalIndex = -1; 

// [參數]
float textureTiling = 0.45f; 
float textureRotation = 0.0f;

// 滑鼠狀態
bool isCameraRotating = false;
bool isPatchMoving = false;
float lastX = 0, lastY = 0;
glm::vec3 lastHitPos(0.0f);

// -----------------------------------------------------------
// 輔助函式：讀取貼圖
// -----------------------------------------------------------
unsigned int loadTextureFromFile(const char* path, int* outWidth, int* outHeight) {
    unsigned int textureID;
    glGenTextures(1, &textureID);

    int width, height, nrComponents;
    stbi_set_flip_vertically_on_load(true);
    unsigned char *data = stbi_load(path, &width, &height, &nrComponents, 0);
    if (data) {
        GLenum format = (nrComponents == 4) ? GL_RGBA : GL_RGB;
        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);
        
        // Decal 設定：Clamp 防止邊緣重複
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
        float borderColor[] = { 0.0f, 0.0f, 0.0f, 0.0f };
        glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        
        stbi_image_free(data);
        if (outWidth) *outWidth = width;
        if (outHeight) *outHeight = height;
    } else {
        std::cout << "Failed to load texture: " << path << std::endl;
        stbi_image_free(data);
    }
    return textureID;
}

void InitTextures() {
    // 這裡嘗試多個路徑前綴，增加讀取成功率
    std::vector<std::string> prefixes = { "../assets/", "assets/", "../../assets/" };
    
    struct TexDef { std::string name; std::string file; };
    std::vector<TexDef> defs = {
        { "Checkerboard", "checkerboard.png" },
        { "Grass", "grass.png" },
        { "Skin", "skin.png" },
        { "Sky", "sky.jpg" },
        { "Wave", "wave.png" }
    };

    for (auto& def : defs) {
        bool loaded = false;
        for(auto& prefix : prefixes) {
            std::string fullPath = prefix + def.file;
            if(fs::exists(fullPath)) {
                int w, h;
                unsigned int id = loadTextureFromFile(fullPath.c_str(), &w, &h);
                textureList.push_back({ def.name, fullPath, id, w, h });
                loaded = true;
                break; 
            }
        }
        if(!loaded) std::cout << "[Warning] Texture not found: " << def.file << std::endl;
    }
}

// -----------------------------------------------------------
// 邏輯：建立 OverlayVertex (含長寬比修正 + 法線)
// -----------------------------------------------------------
std::vector<OverlayVertex> CreateOverlayData(const Model* model, const std::vector<FlatVertex>& flatRes, float uvScale, float normalizeRadius, float rotationDeg, int texWidth, int texHeight) {
    std::vector<OverlayVertex> result;
    const auto& meshVerts = model->meshes[0].vertices;
    
    if (normalizeRadius < 0.0001f) normalizeRadius = 1.0f;

    float rad = glm::radians(rotationDeg);
    float cosR = cos(rad);
    float sinR = sin(rad);

    float aspectRatio = (float)texWidth / (float)texHeight;
    float scaleX = 1.0f;
    float scaleY = 1.0f;

    if (aspectRatio > 1.0f) scaleY = aspectRatio; 
    else scaleX = 1.0f / aspectRatio;

    for (const auto& v : flatRes) {
        OverlayVertex ov;
        ov.position = meshVerts[v.originalIndex].Position;
        
        // [關鍵] 拷貝法線
        ov.normal = meshVerts[v.originalIndex].Normal; 

        glm::vec2 rawUV = (v.uv / normalizeRadius) * uvScale;
        rawUV.x *= scaleX;
        rawUV.y *= scaleY;

        glm::vec2 rotatedUV;
        rotatedUV.x = rawUV.x * cosR - rawUV.y * sinR;
        rotatedUV.y = rawUV.x * sinR + rawUV.y * cosR;

        ov.uv = rotatedUV + glm::vec2(0.5f, 0.5f);
        result.push_back(ov);
    }
    return result;
}

// -----------------------------------------------------------
// 輔助：載入模型
// -----------------------------------------------------------
void LoadNewModel(const std::string& path) {
    if (loadedModel) {
        delete loadedModel;
        loadedModel = nullptr;
    }

    std::cout << "[System] Loading model: " << path << std::endl;
    try {
        loadedModel = new Model(path);
    } catch (...) {
        std::cout << "[Error] Failed to load model. Check format or path." << std::endl;
        return;
    }

    // 清空舊 Decals
    for(auto& d : decalSystem->decals) d.Destroy();
    decalSystem->decals.clear();

    currentSelectionIndices.clear();
    currentExpMapUVs.clear();
    lastHitTriangle = -1;
    editingDecalIndex = -1;
    
    brushRadius = 0.05f;
    textureRotation = 0.0f;

    if(highlightRenderer) 
        highlightRenderer->UpdateTextured(loadedModel, {}, 1.0f, 1.0f, 0.0f);

    if (loadedModel && !loadedModel->meshes.empty()) {
        selectionSystem.BuildTopology(loadedModel->meshes[0].vertices, loadedModel->meshes[0].indices);
    }
}

// -----------------------------------------------------------
// Callbacks
// -----------------------------------------------------------
glm::vec3 GetRayFromMouse(double mx, double my) {
    glm::mat4 proj = glm::perspective(glm::radians(45.0f), (float)SCR_WIDTH / SCR_HEIGHT, 0.1f, 100.0f);
    glm::mat4 view = interaction.GetViewMatrix();
    float x = (2.0f * (float)mx) / SCR_WIDTH - 1.0f;
    float y = 1.0f - (2.0f * (float)my) / SCR_HEIGHT;
    glm::vec4 ray_clip = glm::vec4(x, y, -1.0, 1.0);
    glm::vec4 ray_eye = glm::inverse(proj) * ray_clip;
    ray_eye = glm::vec4(ray_eye.x, ray_eye.y, -1.0, 0.0);
    return glm::normalize(glm::vec3(glm::inverse(view) * ray_eye));
}

void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
    if (ImGui::GetIO().WantCaptureMouse) return;
    if (isCameraRotating) {
        float xoffset = (float)xpos - lastX;
        float yoffset = lastY - (float)ypos; 
        interaction.Orbit(xoffset * 0.2f, yoffset * 0.2f);
    }
    lastX = (float)xpos;
    lastY = (float)ypos;
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    if (ImGui::GetIO().WantCaptureMouse) return;
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            double x, y;
            glfwGetCursorPos(window, &x, &y);
            lastX = (float)x;
            lastY = (float)y;
            if (loadedModel && !loadedModel->meshes.empty()) {
                int hit = selectionSystem.PickTriangle(
                    interaction.GetCameraPos(), 
                    GetRayFromMouse(x, y), 
                    loadedModel->meshes[0].vertices, 
                    loadedModel->meshes[0].indices,
                    lastHitPos 
                );
                if (hit != -1) { isPatchMoving = true; isCameraRotating = false; }
                else { isPatchMoving = false; isCameraRotating = true; }
            }
        } else if (action == GLFW_RELEASE) {
            isCameraRotating = false; isPatchMoving = false;
        }
    }
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    if (ImGui::GetIO().WantCaptureMouse) return;
    interaction.Zoom((float)yoffset * 0.8f);
}

// -----------------------------------------------------------
// Main
// -----------------------------------------------------------
int main() {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "ExpMap Decal Editor", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetScrollCallback(window, scroll_callback);
    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);

    // =======================================================
    // [除錯] 顯示當前工作目錄
    // =======================================================
    std::cout << "[Debug] Current Working Directory: " << fs::current_path() << std::endl;

    // [智慧搜尋] 嘗試在多個路徑尋找 assets 資料夾
    modelList.clear();
    std::vector<std::string> searchPaths = { 
        "../assets/", 
        "assets/", 
        "../../assets/" 
    };

    bool foundAnyAssets = false;
    for (const auto& path : searchPaths) {
        if (fs::exists(path)) {
            std::cout << "[Info] Scanning assets in: " << path << std::endl;
            
            for (const auto& entry : fs::directory_iterator(path)) {
                std::string ext = entry.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

                if (ext == ".obj" || ext == ".ply") {
                    // 使用 generic_string 確保路徑格式統一 (使用 /)
                    modelList.push_back(entry.path().generic_string());
                    std::cout << "  - Found: " << entry.path().filename() << std::endl;
                }
            }
            if(!modelList.empty()) {
                foundAnyAssets = true;
                break; // 找到資料夾且裡面有東西，就停止搜尋
            }
        }
    }

    if (!foundAnyAssets) {
        std::cout << "[Warning] No 3D models found in standard asset paths!" << std::endl;
        std::cout << "          Please put .obj or .ply files in 'assets/' folder." << std::endl;
        // 塞一個假路徑防止崩潰
        modelList.push_back("assets/armadillo.obj");
    }

    // Systems Setup
    highlightRenderer = new HighlightRenderer();
    decalSystem = new DecalSystem();
    InitTextures();

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");
    ImGui::StyleColorsDark();

    Shader modelShader("../src/shaders/model.vert", "../src/shaders/model.frag");

    // 初次載入
    if(!modelList.empty()) {
        LoadNewModel(modelList[0]);
    }
    interaction.SetTarget(glm::vec3(0, 0, 0));

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // --- 1. 選取邏輯 ---
        if (isPatchMoving && !ImGui::GetIO().WantCaptureMouse) {
            double mx, my; glfwGetCursorPos(window, &mx, &my);
            
            int hitTri = selectionSystem.PickTriangle(
                interaction.GetCameraPos(), 
                GetRayFromMouse(mx, my), 
                loadedModel->meshes[0].vertices, 
                loadedModel->meshes[0].indices, 
                lastHitPos
            );

            if (hitTri != -1) {
                lastHitTriangle = hitTri;
                currentSelectionIndices = selectionSystem.GetPatchByRadius(hitTri, brushRadius);
                if (selectionSystem.topology.isValid) {
                    currentExpMapUVs = selectionSystem.FlattenPatch(
                        currentSelectionIndices, 
                        loadedModel->meshes[0].vertices, 
                        loadedModel->meshes[0].indices, 
                        lastHitPos 
                    );
                    
                    if (highlightRenderer) 
                        highlightRenderer->UpdateTextured(loadedModel, currentExpMapUVs, textureTiling, brushRadius, textureRotation);
                }
            }
        }

        // --- 2. 繪圖 ---
        glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glm::mat4 proj = glm::perspective(glm::radians(45.0f), (float)SCR_WIDTH / SCR_HEIGHT, 0.01f, 100.0f);
        glm::mat4 view = interaction.GetViewMatrix();
        glm::mat4 model = glm::mat4(1.0f);

        if (loadedModel) {
            // A. 畫模型
            modelShader.use();
            modelShader.setMat4("projection", proj);
            modelShader.setMat4("view", view);
            modelShader.setMat4("model", model);
            modelShader.setVec3("viewPos", interaction.GetCameraPos());
            loadedModel->Draw(modelShader);

            // B. 畫所有貼圖 (Decals) - 記得切換 Shader
            glEnable(GL_POLYGON_OFFSET_FILL);
            glPolygonOffset(-20.0f, -20.0f);
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            glDepthFunc(GL_LEQUAL);

            if (decalSystem) {
                Shader* decalShader = highlightRenderer->GetShader();
                if (decalShader) {
                    decalShader->use();
                    decalShader->setMat4("view", view);
                    decalShader->setMat4("projection", proj);
                    decalShader->setMat4("model", model);
                    decalShader->setVec3("viewPos", interaction.GetCameraPos());
                    decalShader->setInt("texture1", 0);
                }
                decalSystem->Draw(view, proj, model, editingDecalIndex);
            }

            // C. 畫目前的選取預覽
            if (highlightRenderer && !currentSelectionIndices.empty()) {
                unsigned int previewTexID = textureList[currentTextureIndex].id;
                highlightRenderer->Draw(view, proj, model, previewTexID, interaction.GetCameraPos()); 
            }

            glDisable(GL_POLYGON_OFFSET_FILL);
            glDepthFunc(GL_LESS);
            glDisable(GL_BLEND);
        }

        // --- 3. UI ---
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::Begin("Decal Editor");

        // 3.0 模型選擇
        if (ImGui::CollapsingHeader("Model Selection", ImGuiTreeNodeFlags_DefaultOpen)) {
            // 顯示目前列表
            if (ImGui::BeginCombo("List", (modelList.empty() ? "None" : modelList[currentModelIndex].c_str()))) {
                for (int i = 0; i < modelList.size(); i++) {
                    if (ImGui::Selectable(modelList[i].c_str(), currentModelIndex == i)) {
                        currentModelIndex = i;
                        LoadNewModel(modelList[currentModelIndex]);
                    }
                }
                ImGui::EndCombo();
            }
            ImGui::Text("Or load from path (Absolute Path):");
            ImGui::InputText("Path", customModelPath, sizeof(customModelPath));
            ImGui::SameLine();
            if (ImGui::Button("Load")) {
                modelList.push_back(std::string(customModelPath));
                currentModelIndex = (int)modelList.size() - 1;
                LoadNewModel(modelList[currentModelIndex]);
            }
        }
        ImGui::Separator();

        // 3.1 貼圖選單
        if (!textureList.empty()) {
            if (ImGui::BeginCombo("Texture", textureList[currentTextureIndex].name.c_str())) {
                for (int i = 0; i < textureList.size(); i++) {
                    if (ImGui::Selectable(textureList[i].name.c_str(), currentTextureIndex == i)) {
                        currentTextureIndex = i;
                        if(!currentExpMapUVs.empty()) 
                            highlightRenderer->UpdateTextured(loadedModel, currentExpMapUVs, textureTiling, brushRadius, textureRotation);
                    }
                }
                ImGui::EndCombo();
            }
        }

        // 3.2 參數
        bool paramChanged = false;
        if (ImGui::SliderFloat("Rotation", &textureRotation, 0.0f, 360.0f)) paramChanged = true;
        
        if (paramChanged && !currentExpMapUVs.empty()) {
            highlightRenderer->UpdateTextured(loadedModel, currentExpMapUVs, textureTiling, brushRadius, textureRotation);
        }

        if (ImGui::SliderFloat("Radius", &brushRadius, 0.01f, 3.0f)) {
            if (lastHitTriangle != -1) {
                currentSelectionIndices = selectionSystem.GetPatchByRadius(lastHitTriangle, brushRadius);
                if (selectionSystem.topology.isValid) {
                    currentExpMapUVs = selectionSystem.FlattenPatch(currentSelectionIndices, loadedModel->meshes[0].vertices, loadedModel->meshes[0].indices, lastHitPos);
                    highlightRenderer->UpdateTextured(loadedModel, currentExpMapUVs, textureTiling, brushRadius, textureRotation);
                }
            }
        }

        // 3.4 按鈕
        if (editingDecalIndex == -1) {
            if (ImGui::Button("Apply Texture (Create)")) {
                if (!currentExpMapUVs.empty()) {
                    int w = textureList[currentTextureIndex].width;
                    int h = textureList[currentTextureIndex].height;
                    std::vector<OverlayVertex> verts = CreateOverlayData(loadedModel, currentExpMapUVs, textureTiling, brushRadius, textureRotation, w, h); 
                    
                    std::string name = "Decal " + std::to_string(decalSystem->decals.size() + 1);
                    Decal newDecal(name, textureList[currentTextureIndex].id, verts, currentSelectionIndices, lastHitTriangle, brushRadius, textureTiling, textureRotation);
                    decalSystem->AddDecal(newDecal);
                    
                    currentSelectionIndices.clear();
                    currentExpMapUVs.clear();
                    lastHitTriangle = -1;
                    highlightRenderer->UpdateTextured(loadedModel, {}, 1.0f, 1.0f, 0.0f); 
                }
            }
        } else {
            if (ImGui::Button("Update Decal")) {
                if (!currentExpMapUVs.empty()) {
                    Decal& d = decalSystem->decals[editingDecalIndex];
                    d.Destroy();
                    
                    int w = textureList[currentTextureIndex].width;
                    int h = textureList[currentTextureIndex].height;
                    std::vector<OverlayVertex> verts = CreateOverlayData(loadedModel, currentExpMapUVs, textureTiling, brushRadius, textureRotation, w, h);

                    d = Decal(d.name, textureList[currentTextureIndex].id, verts, currentSelectionIndices, lastHitTriangle, brushRadius, textureTiling, textureRotation);
                    
                    editingDecalIndex = -1;
                    currentSelectionIndices.clear();
                    currentExpMapUVs.clear();
                    lastHitTriangle = -1;
                    highlightRenderer->UpdateTextured(loadedModel, {}, 1.0f, 1.0f, 0.0f);
                }
            }
            ImGui::SameLine();
            if (ImGui::Button("Cancel Edit")) {
                editingDecalIndex = -1;
                currentSelectionIndices.clear();
                currentExpMapUVs.clear();
                lastHitTriangle = -1;
                highlightRenderer->UpdateTextured(loadedModel, {}, 1.0f, 1.0f, 0.0f);
            }
        }

        ImGui::Separator();
        ImGui::Text("Decal List (Click to Edit):");
        for (int i = 0; i < decalSystem->decals.size(); i++) {
            bool isSelected = (editingDecalIndex == i);
            if (ImGui::Selectable(decalSystem->decals[i].name.c_str(), isSelected)) {
                editingDecalIndex = i;
                Decal& d = decalSystem->decals[i];
                
                lastHitTriangle = d.centerTriangle;
                brushRadius = d.radius;
                textureTiling = d.textureTiling; 
                textureRotation = d.rotation; 
                
                for(int t=0; t<textureList.size(); t++) {
                    if(textureList[t].id == d.textureID) {
                        currentTextureIndex = t;
                        break;
                    }
                }
                
                currentSelectionIndices = selectionSystem.GetPatchByRadius(lastHitTriangle, brushRadius);
                if (selectionSystem.topology.isValid) {
                    glm::vec3 centerPos = selectionSystem.topology.centroids[lastHitTriangle];
                    lastHitPos = centerPos; 
                    currentExpMapUVs = selectionSystem.FlattenPatch(currentSelectionIndices, loadedModel->meshes[0].vertices, loadedModel->meshes[0].indices, lastHitPos);
                    highlightRenderer->UpdateTextured(loadedModel, currentExpMapUVs, textureTiling, brushRadius, textureRotation);
                }
            }
        }
        ImGui::End();

        // UV Viewer
        ImGui::Begin("UV Viewer (Flattened)");
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        ImVec2 p = ImGui::GetCursorScreenPos();
        float canvas_sz = 300.0f;
        draw_list->AddRectFilled(p, ImVec2(p.x + canvas_sz, p.y + canvas_sz), IM_COL32(50, 50, 50, 255));
        draw_list->AddRect(p, ImVec2(p.x + canvas_sz, p.y + canvas_sz), IM_COL32(255, 255, 255, 255));

        if (!currentExpMapUVs.empty()) {
            glm::vec2 minUV(10000.0f), maxUV(-10000.0f);
            for (const auto& v : currentExpMapUVs) {
                minUV.x = std::min(minUV.x, v.uv.x);
                minUV.y = std::min(minUV.y, v.uv.y);
                maxUV.x = std::max(maxUV.x, v.uv.x);
                maxUV.y = std::max(maxUV.y, v.uv.y);
            }
            glm::vec2 size = maxUV - minUV;
            float availableSize = canvas_sz - 40.0f;
            float scaleX = (size.x > 0.0001f) ? (availableSize / size.x) : 1.0f;
            float scaleY = (size.y > 0.0001f) ? (availableSize / size.y) : 1.0f;
            float scale = (scaleX < scaleY) ? scaleX : scaleY;
            glm::vec2 midUV = (minUV + maxUV) * 0.5f;
            ImVec2 canvasCenter(p.x + canvas_sz * 0.5f, p.y + canvas_sz * 0.5f);

            auto ToScreen = [&](glm::vec2 uv) -> ImVec2 {
                float x = (uv.x - midUV.x) * scale + canvasCenter.x;
                float y = (uv.y - midUV.y) * scale + canvasCenter.y;
                return ImVec2(x, y);
            };

            for (size_t i = 0; i < currentExpMapUVs.size(); i += 3) {
                if (i + 2 >= currentExpMapUVs.size()) break;
                ImVec2 p1 = ToScreen(currentExpMapUVs[i].uv);
                ImVec2 p2 = ToScreen(currentExpMapUVs[i+1].uv);
                ImVec2 p3 = ToScreen(currentExpMapUVs[i+2].uv);
                draw_list->AddTriangle(p1, p2, p3, IM_COL32(255, 255, 0, 255), 2.0f);
            }
            ImVec2 centerScreen = ToScreen(glm::vec2(0, 0));
            draw_list->AddCircleFilled(centerScreen, 4.0f, IM_COL32(255, 0, 0, 255));
        }
        ImGui::InvisibleButton("canvas", ImVec2(canvas_sz, canvas_sz));
        ImGui::End();
        
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }
    
    // Cleanup
    delete loadedModel;
    delete highlightRenderer;
    delete decalSystem;
    
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwTerminate();

    return 0;
}