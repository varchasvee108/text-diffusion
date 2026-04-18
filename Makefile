.PHONY: train inference

train:
	python -m scripts.train

inference:
	python -m scripts.infer
