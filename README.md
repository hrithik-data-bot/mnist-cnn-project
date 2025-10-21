# mnist-cnn-project

## ⚙️ Setup Instructions

1. **Clone the repository**
```bash
git clone git@github.com:hrithik-data-bot/mnist-cnn-project.git
cd mnist-cnn-project
```
2. **Creating the environment**
```bash
conda env create --file conda.yaml
```
3. **Activating the environment**
```bash
conda activate mnist-cnn-gpu
```
4. **Deactivating the environment**
```bash
conda deactivate
```
5. **Adding ipykernel**
```bash
conda activate mnist-cnn-gpu
python -m ipykernel install --user --name=mnist-cnn-gpu
```