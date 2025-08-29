from mealpy.swarm_based import MBO
import numpy as np
from src.train import trainer


def objective_function(solution):
    lr, batch_size, epochs = solution
    trainer.args.learning_rate = float(lr)
    trainer.args.per_device_train_batch_size = int(batch_size)
    trainer.args.num_train_epochs = int(epochs)

    trainer.train()
    metrics = trainer.evaluate()
    return metrics["eval_loss"]


problem = {
    "fit_func": objective_function,
    "lb": [1e-6, 8, 2],
    "ub": [5e-5, 32, 5],
    "minmax": "min",
}

if __name__ == "__main__":
    optimizer = MBO.BaseMBO(problem, epoch=5, pop_size=4)
    best_params, best_fitness = optimizer.solve()
    print("Best Params:", best_params)
