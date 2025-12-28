You can use [Google Colab in VSCode](https://www.freecodecamp.org/news/how-to-use-google-colab-with-vs-code/), or follow the steps below to run locally in your machine:
```
pip install uv
uv venv --python 3.12
.venv\Scripts\activate
uv pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124 --index-url https://pypi.org/simple
uv run python -c "import torch; print(torch.cuda.is_available())"
```

Execute jupyter notebook: 
	- `uv run jupyter lab`
	- `uv run jupyter notebook`
	- `uv run jupyter lab --ServerApp.iopub_msg_rate_limit=0 --ServerApp.rate_limit_window=0`
	- `uv run jupyter notebook --ServerApp.iopub_msg_rate_limit=0 --ServerApp.rate_limit_window=0`

https://github.com/guilherme-alves-silve/machine-learning-python/blob/master/rf_lr/deepqlearning/deep_q_learning_rflr_gym.ipynb
https://github.com/FareedKhan-dev/all-rl-algorithms/blob/master/13_dqn.ipynb
https://github.com/Curt-Park/rainbow-is-all-you-need/blob/master/01.dqn.ipynb
