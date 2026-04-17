# CS4100 Final: Word Game

## Clustering Set Up

Located in `notebooks\`

Set up the python environment:

1. `python -m venv venv`
2. `venv\Scripts\activate`
3. `pip install -r requirements.txt`
4. `jupyter notebook`
5. Open `clustering.ipynb` at localhost:8888 and run all cells to generate the clusters
6. Access clusters with the `clusters` dictionary (cluster_id: [words in cluster])


To run training, run python train.py
To run testing, run python main.py
The struction is generally as follows, 
-the main directly contained the main files and envrionment
-the notebooks contains the clustering code
-scripts contains the clues giving interfacing the cluster and the Q-learning
-data contains our wordsets and pkls for clustering.
