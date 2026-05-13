# Texture using expMap on gaussian splatting model
After building sibr viewer run these code :
  cd D:\SGGaussians\SGGaussians\SIBR_viewers
  cmake -B build
  cmake --build build --config Release --target install

  cd install\bin
  .\SIBR_gaussianViewer_app -m D:\SGGaussians\SGGaussians\output\armadillo
