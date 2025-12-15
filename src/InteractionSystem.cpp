#include "InteractionSystem.h"
#include <glm/gtc/matrix_inverse.hpp>
#include <cfloat>

InteractionSystem::InteractionSystem()
{
    yaw = 0.0f;
    pitch = 0.0f;
    distance = 2.0f;
    target = glm::vec3(0.0f);
}

// ========================================
// CAMERA
// ========================================
void InteractionSystem::SetTarget(glm::vec3 t)
{
    target = t;
}

void InteractionSystem::Orbit(float deltaYaw, float deltaPitch)
{
    yaw   += deltaYaw;
    pitch += deltaPitch;
    pitch = glm::clamp(pitch, -89.0f, 89.0f);
}

void InteractionSystem::Zoom(float delta)
{
    distance -= delta;
    distance = glm::clamp(distance, 1.0f, 50.0f);
}

glm::vec3 InteractionSystem::GetCameraPos() const
{
    float yawRad = glm::radians(yaw);
    float pitRad = glm::radians(pitch);

    glm::vec3 cam(
        distance * cos(pitRad) * sin(yawRad),
        distance * sin(pitRad),
        distance * cos(pitRad) * cos(yawRad)
    );
    return cam + target;
}

glm::mat4 InteractionSystem::GetViewMatrix() const
{
    return glm::lookAt(GetCameraPos(), target, glm::vec3(0,1,0));
}

glm::mat4 InteractionSystem::GetProjectionMatrix(float aspect) const
{
    return glm::perspectiveRH_ZO(glm::radians(45.0f),
                                 aspect,
                                 0.1f, 100.0f);
}

// ========================================
// RAYCAST
// ========================================
Ray InteractionSystem::GenerateRay(float mx, float my,
                                   int screenW, int screenH) const
{
    float x = (2.0f * mx) / screenW - 1.0f;
    float y = 1.0f - (2.0f * my) / screenH;

    glm::mat4 proj = GetProjectionMatrix((float)screenW / screenH);
    glm::mat4 view = GetViewMatrix();

    glm::vec4 clip(x, y, -1, 1);

    glm::vec4 eye = glm::inverse(proj) * clip;
    eye = glm::vec4(eye.x, eye.y, -1, 0);

    glm::vec3 dir = glm::normalize(glm::vec3(glm::inverse(view) * eye));

    return { GetCameraPos(), dir };
}

bool InteractionSystem::RayTri(const Ray& r,
                               const glm::vec3& A,
                               const glm::vec3& B,
                               const glm::vec3& C,
                               float& tOut) const
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

bool InteractionSystem::Raycast(Model* model,
                                const Ray& r,
                                HitInfo& out)
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
                    out.hit       = true;
                    out.hitPos    = r.origin + r.direction * t;
                    out.meshIndex = m;
                    out.triIndex  = i;
                }
            }
        }
    }

    return out.hit;
}
