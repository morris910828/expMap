#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>

#include "Model.h"

// -----------------------------------------
// Ray Struct
// -----------------------------------------
struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
};

struct HitInfo
{
    bool hit = false;
    glm::vec3 hitPos;
    int meshIndex = -1;
    int triIndex  = -1;
};

// -----------------------------------------
// InteractionSystem
// - Orbit Camera
// - Raycasting
// -----------------------------------------
class InteractionSystem
{
public:
    InteractionSystem();

    // =========== CAMERA ===========
    void SetTarget(glm::vec3 t);
    void Orbit(float deltaYaw, float deltaPitch);
    void Zoom(float delta);
    glm::vec3 GetCameraPos() const;

    float GetYaw() const { return yaw; }
    float GetPitch() const { return pitch; }
    float GetDistance() const { return distance; }

    glm::mat4 GetViewMatrix() const;
    glm::mat4 GetProjectionMatrix(float aspect) const;

    // =========== RAYCAST ===========
    Ray GenerateRay(float mouseX, float mouseY,
                    int screenW, int screenH) const;

    bool Raycast(Model* model, const Ray& r, HitInfo& out);

private:
    float yaw, pitch;
    float distance;
    glm::vec3 target;

    bool RayTri(const Ray& r,
                const glm::vec3& A,
                const glm::vec3& B,
                const glm::vec3& C,
                float& tOut) const;
};
