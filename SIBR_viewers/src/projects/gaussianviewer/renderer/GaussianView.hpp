/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact sibr@inria.fr and/or George.Drettakis@inria.fr
 */
#pragma once

# include "Config.hpp"
# include <core/renderer/RenderMaskHolder.hpp>
# include <core/scene/BasicIBRScene.hpp>
# include <core/system/SimpleTimer.hpp>
# include <core/system/Config.hpp>
# include <core/graphics/Mesh.hpp>
# include <core/view/ViewBase.hpp>
# include <core/renderer/CopyRenderer.hpp>
# include <core/renderer/PointBasedRenderer.hpp>
# include <memory>
# include <core/graphics/Texture.hpp>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <functional>
#include <unordered_map>
# include "GaussianSurfaceRenderer.hpp"

namespace CudaRasterizer
{
	class Rasterizer;
}

namespace sibr { 

	class BufferCopyRenderer;
	class BufferCopyRenderer2;

	/**
	 * \class GaussianView
	 * \brief Renders 3D Gaussian splats. Does NOT draw overlay points;
	 *        overlay point rendering is handled by MeshGaussianView in main.cpp.
	 */
	class SIBR_EXP_ULR_EXPORT GaussianView : public sibr::ViewBase
	{
		SIBR_CLASS_PTR(GaussianView);

	public:

		GaussianView(const sibr::BasicIBRScene::Ptr& ibrScene, uint render_w, uint render_h,
		             const char* file, bool* message_read, int sh_degree,
		             bool white_bg = false, bool useInterop = true, int device = 0);

		void setScene(const sibr::BasicIBRScene::Ptr & newScene);

		void onRenderIBR(sibr::IRenderTarget& dst, const sibr::Camera& eye) override;
		void onUpdate(Input& input) override;
		void onGUI() override;

		/** Update Gaussian colors (SH DC) for the given original PLY indices.
		 *  Uses inverse mapping to handle Morton-sorted buffer and SH domain conversion.
		 */
		void updateGaussianColorsBulk(const std::vector<int>& originalIndices, const std::vector<sibr::Vector3f>& colors);

		/** Update Gaussian colours by matching world-space positions.
		 *  This is the correct method when the caller has geo/app sub-cloud points
		 *  whose indices do NOT directly map to point_cloud.ply indices.
		 *  Builds a position->sortedIndex hash at construction time; call once per bake.
		 *  Higher-order SH bands are zeroed for baked Gaussians (pure diffuse texture).
		 */
		void updateGaussianColorsByWorldPos(
			const std::vector<sibr::Vector3f>& worldPositions,
			const std::vector<sibr::Vector3f>& colors);

		const std::shared_ptr<sibr::BasicIBRScene> & getScene() const { return _scene; }

		virtual ~GaussianView() override;

		bool* _dontshow;

	protected:

		std::string currMode = "Splats";
		bool _antialiasing = false;
		bool _cropping = false;
		sibr::Vector3f _boxmin, _boxmax, _scenemin, _scenemax;
		char _buff[512] = "cropped.ply";

		bool _fastCulling = true;
		int _device = 0;
		int _sh_degree = 3;

		int count;
		float* pos_cuda;
		float* rot_cuda;
		float* scale_cuda;
		float* opacity_cuda;
		float* shs_cuda;
		int* rect_cuda;

		// Inverse mapping: original PLY index -> sorted GPU index
		std::vector<int> _plyToSorted;

		// Position -> sorted GPU index (built at load time for world-pos baking)
		// Key = hash of raw float bits of (x, y, z).
		std::unordered_map<uint64_t, int> _posToSorted;

		GLuint imageBuffer;
		cudaGraphicsResource_t imageBufferCuda;

		size_t allocdGeom = 0, allocdBinning = 0, allocdImg = 0;
		void* geomPtr = nullptr, * binningPtr = nullptr, * imgPtr = nullptr;
		std::function<char* (size_t N)> geomBufferFunc, binningBufferFunc, imgBufferFunc;

		float* view_cuda;
		float* proj_cuda;
		float* cam_pos_cuda;
		float* background_cuda;

		float _scalingModifier = 1.0f;
		GaussianData* gData;

		bool _interop_failed = false;
		std::vector<char> fallback_bytes;
		float* fallbackBufferCuda = nullptr;
		bool accepted = false;

		std::shared_ptr<sibr::BasicIBRScene> _scene;
		PointBasedRenderer::Ptr _pointbasedrenderer;
		BufferCopyRenderer* _copyRenderer;
		GaussianSurfaceRenderer* _gaussianRenderer;
	};

} /*namespace sibr*/