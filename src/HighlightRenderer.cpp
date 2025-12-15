#include "HighlightRenderer.h"
#include <glad/glad.h>

// ------------------------------------------------------------
// Constructor / Destructor
// ------------------------------------------------------------
HighlightRenderer::HighlightRenderer() {}
HighlightRenderer::~HighlightRenderer()
{
    if (vbo) glDeleteBuffers(1, &vbo);
    if (vao) glDeleteVertexArrays(1, &vao);
}

// ------------------------------------------------------------
// Init
// ------------------------------------------------------------
void HighlightRenderer::Init()
{
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE,
                          sizeof(glm::vec3), (void*)0);

    glBindVertexArray(0);
}

// ------------------------------------------------------------
// Update highlight buffer
// ------------------------------------------------------------
void HighlightRenderer::Update(const Model* model,
                               const std::vector<SelectedTri>& selected)
{
    std::vector<glm::vec3> verts;
    verts.reserve(selected.size() * 3);

    for (auto& t : selected)
    {
        const Mesh& mesh = model->meshes[t.meshIndex];
        verts.push_back(mesh.vertices[t.i0].Position);
        verts.push_back(mesh.vertices[t.i1].Position);
        verts.push_back(mesh.vertices[t.i2].Position);
    }

    vertexCount = (int)verts.size();

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER,
                 verts.size() * sizeof(glm::vec3),
                 verts.data(),
                 GL_DYNAMIC_DRAW);
}

// ------------------------------------------------------------
// Draw
// ------------------------------------------------------------
void HighlightRenderer::Draw(
        Shader& shader,
        const glm::mat4& view,
        const glm::mat4& projection)
{
    if (vertexCount == 0) return;

    glDisable(GL_DEPTH_TEST);

    shader.use();
    shader.setInt("useTexture", 0);
    shader.setVec3("overrideColor", glm::vec3(0.1f, 1.0f, 0.1f));
    shader.setMat4("model", glm::mat4(1.0f));
    shader.setMat4("view", view);
    shader.setMat4("projection", projection);

    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, vertexCount);

    glEnable(GL_DEPTH_TEST);
}
