#ifndef MODEL_H
#define MODEL_H

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <string>
#include <vector>
#include <iostream>

#include "Mesh.h"
#include "stb_image.h"

struct SelectedTri {
    int meshIndex;
    unsigned int i0, i1, i2;
};

class Model
{
public:

    std::vector<Mesh> meshes;
    std::string directory;

    Model(const std::string &path)
    {
        loadModel(path);
    }

    void Draw(Shader &shader)
    {
        for (unsigned int i = 0; i < meshes.size(); i++) {
            meshes[i].Draw(shader);
        }
    }

private:

    // -------------------------------
    // 讀取模型
    // -------------------------------
    void loadModel(std::string const &path)
    {
        Assimp::Importer importer;

        const aiScene *scene = importer.ReadFile(
            path,
            aiProcess_Triangulate |
            aiProcess_FlipUVs |
            aiProcess_GenNormals |
            aiProcess_JoinIdenticalVertices
        );

        if (!scene || !scene->mRootNode || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE)
        {
            std::cout << "ASSIMP ERROR: " << importer.GetErrorString() << std::endl;
            return;
        }

        // 記錄目錄路徑
        directory = path.substr(0, path.find_last_of('/'));

        processNode(scene->mRootNode, scene);
    }

    // -------------------------------
    // 遞迴處理節點
    // -------------------------------
    void processNode(aiNode *node, const aiScene *scene)
    {
        // 處理 Node 中的所有 Mesh
        for (unsigned int i = 0; i < node->mNumMeshes; i++)
        {
            aiMesh *mesh = scene->mMeshes[node->mMeshes[i]];
            meshes.push_back(processMesh(mesh, scene));

            // [修改] 請務必取消這一行的註解！
            meshes.back().BuildAdjacency(); 
        }

        // 處理子節點
        for (unsigned int i = 0; i < node->mNumChildren; i++)
        {
            processNode(node->mChildren[i], scene);
        }
    }

    // -------------------------------
    // 處理 mesh：讀取頂點 + 面
    // -------------------------------
    Mesh processMesh(aiMesh *mesh, const aiScene *scene)
    {
        std::vector<Vertex> vertices;
        std::vector<unsigned int> indices;

        // ---- 讀取頂點 ----
        for (unsigned int i = 0; i < mesh->mNumVertices; i++)
        {
            Vertex vertex;

            // position
            vertex.Position = glm::vec3(
                mesh->mVertices[i].x,
                mesh->mVertices[i].y,
                mesh->mVertices[i].z
            );

            // normal
            if (mesh->HasNormals())
            {
                vertex.Normal = glm::vec3(
                    mesh->mNormals[i].x,
                    mesh->mNormals[i].y,
                    mesh->mNormals[i].z
                );
            }
            else
                vertex.Normal = glm::vec3(0.0f);

            // texcoords
            if (mesh->mTextureCoords[0])
            {
                vertex.TexCoords = glm::vec2(
                    mesh->mTextureCoords[0][i].x,
                    mesh->mTextureCoords[0][i].y
                );
            }
            else
                vertex.TexCoords = glm::vec2(0.0f);

            vertices.push_back(vertex);
        }

        // ---- 讀取面（三角形）----
        for (unsigned int i = 0; i < mesh->mNumFaces; i++)
        {
            aiFace face = mesh->mFaces[i];
            for (unsigned int j = 0; j < face.mNumIndices; j++)
                indices.push_back(face.mIndices[j]);
        }

        // 建立 Mesh 並返回
        return Mesh(vertices, indices);
    }
};

#endif
