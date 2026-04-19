.PHONY: train inference

train:
	python -m scripts.train

infer:
	python -m scripts.infer
