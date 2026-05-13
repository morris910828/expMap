python trainMesh_SG.py -s .\data\armadillo -m .\output\armadillo --eval -w
python trainMesh_SG.py -s .\data\bunny -m .\output\bunny --eval -w
python trainMesh_SG.py -s .\data\dragon -m .\output\dragon --eval -w
python trainMesh_SG.py -s .\data\lucy -m .\output\lucy --eval -w
python trainMesh_SG.py -s .\data\max -m .\output\max --eval -w

python render.py -m .\output\armadillo 
python render.py -m .\output\bunny
python render.py -m .\output\dragon
python render.py -m .\output\lucy
python render.py -m .\output\max

python metrics.py -m .\output\armadillo 
python metrics.py -m .\output\bunny
python metrics.py -m .\output\dragon
python metrics.py -m .\output\lucy
python metrics.py -m .\output\max