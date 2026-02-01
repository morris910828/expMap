#pragma once

#include <string>
#include <vector>
#include <memory>
#include <core/graphics/Image.hpp>
#include <core/graphics/Texture.hpp>
#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;

class TextureLoader {
public:
    TextureLoader() = default;

    bool LoadImage(const std::string& path) {
        sibr::ImageRGBA image;
        if (image.load(path)) {
            _texture.reset(new sibr::Texture2DRGBA(image, SIBR_GPU_LINEAR_SAMPLING));
            _path = path;
            return true;
        }
        return false;
    }

    std::vector<std::string> scanForImages(const std::string& directory) {
        std::vector<std::string> imagePaths;
        fs::path dirPath(directory);
        if (!fs::exists(dirPath) || !fs::is_directory(dirPath)) {
            return imagePaths;
        }

        const std::vector<std::string> supported_extensions = { ".png", ".jpg", ".jpeg", ".bmp", ".tga" };

        for (const auto& entry : fs::directory_iterator(dirPath)) {
            if (fs::is_regular_file(entry.status())) {
                fs::path entryPath = entry.path();
                std::string extension = entryPath.extension().string();
                std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
                for (const auto& sup_ext : supported_extensions) {
                    if (extension == sup_ext) {
                        imagePaths.push_back(entryPath.string());
                        break;
                    }
                }
            }
        }
        return imagePaths;
    }

    const sibr::Texture2DRGBA::Ptr& getTexture() const {
        return _texture;
    }

    const std::string& getPath() const {
        return _path;
    }

private:
    sibr::Texture2DRGBA::Ptr _texture;
    std::string _path;
};
