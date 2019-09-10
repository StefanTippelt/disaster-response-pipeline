SHELL := /usr/bin/env bash

DATA := data
MODEL := models

.PHONY: deps
deps:
	pip install --upgrade pip setuptools
	pip install -r requirements.txt

.PHONY: process_data
process_data:
	python ${DATA}/process_data.py ${DATA}/disaster_messages.csv ${DATA}/disaster_categories.csv ${DATA}/disaster.db

.PHONY: train_model
train_model:
	python ${MODEL}/train_classifier.py ${DATA}/disaster.db ${MODEL}/classifier.pkl

.PHONY: run
run:
	python app/run.py
