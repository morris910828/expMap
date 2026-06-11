# Physics-Based Gaussian Splatting Applications via Hybrid Structures

## Set-up

### Conda Environment
```
conda env create --file environment.yml
conda activate physics-based-gaussian
```

The ACAP modules should be compiled:

```
cd ACAP
unzip pyACAPv1.zip
chmod +x install_OpenMesh.sh
install_OpenMesh.sh
python setup.py install
```

## Get Starting
### Example

There is an example of final directory:
```
<physics-based-gaussian>
|---data
|   |---<chair>
|   |   |---transforms_train.json
|   |   |---transforms_test.json
|   |   |---transforms_val.json
|   |   |---mesh.obj
|   |   |---train
|   |   |   |---cam000.png
|   |   |   |---cam001.png
|   |   |   |---...
|   |   |---test
|   |   |   |---cam000.png
|   |   |   |---cam001.png
|   |   |   |---...
|   |   |---val
|   |   |   |---cam000.png
|   |   |   |---cam001.png
|   |   |   |---...
|   |   |---...
|---deform
|   |---<chair>
|   |   |---test_1.obj
|   |   |---...
|---train.py
|---metrics.py
|---...
```


### Dataset

The dataset used in this study is stored in 
```
/ResearchData/115_ZhiTeng/data
```
Download file **`dataset.zip`**

- Downloaded dataset should have the following format:
```
<dataset>
|---data
|   |---<model-name>
|   |   |---transforms_train.json
|   |   |---transforms_test.json
|   |   |---transforms_val.json
|   |   |---mesh.obj
|   |   |---train
|   |   |   |---cam000.png
|   |   |   |---cam001.png
|   |   |   |---...
|   |   |---test
|   |   |   |---cam000.png
|   |   |   |---cam001.png
|   |   |   |---...
|   |   |---val
|   |   |   |---cam000.png
|   |   |   |---cam001.png
|   |   |   |---...
|   |   |---...
|---deform
|   |---<model-name>
|   |   |---test_1.obj
|   |   |---...
```

To generate same dataset as above, use code stored in
```
/ResearchData/115_ZhiTeng/data/dataset_generate.zip
```

After extracting the zip file, please follow the instructions provided in readme.txt.


### Training
```
python train.py -s <path to dataset> -m <path to trained model>
```

### Evaluation
```
python train.py -s <path to dataset> -m <path to trained model> --eval
python render.py -m <path to trained model> 
python metrics.py -m <path to trained model> 
```

### Deformation
```
python deformation.py -m <path to trained model> --deform_path <path to deform mesh>
```

## Unreal Plugins
To use unreal plugins for Physics-Based Gaussian Splatting

Download file **`NTUST_GS_Plugin.zip`** store in 
```
/ResearchData/115_ZhiTeng/程式碼
```

After extracting the zip file, please follow the instructions provided in word file.