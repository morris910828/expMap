#pragma once
#include <vector>
#include <glm/glm.hpp>
#include "Model.h"
#include "Shader.h"

// ------------------------------------------------------------
// 專門負責 highlight 三角形的繪製
// ------------------------------------------------------------
class HighlightRenderer
{
public:
    HighlightRenderer();
    ~HighlightRenderer();

    void Init();
    void Update(const Model* model, 
                const std::vector<SelectedTri>& selected);

    void Draw(Shader& shader, 
              const glm::mat4& view,
              const glm::mat4& projection);

private:
    GLuint vao = 0;
    GLuint vbo = 0;
    int vertexCount = 0;
};
