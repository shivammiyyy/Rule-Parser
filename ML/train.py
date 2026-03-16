"""
CLI entry-point: python -m ML.train

Trains all models and saves them to the models/ directory.
Use ML.trainer.train_all() to train programmatically.
"""

from .trainer import train_all

if __name__ == "__main__":
    train_all()