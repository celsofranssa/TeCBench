import os
import hydra
from omegaconf import OmegaConf
from source.helper.EvalHelper import EvalHelper
from source.helper.FitHelper import FitHelper
from source.helper.PredictHelper import PredictHelper
from source.helper.SeparabilityHelper import SeparabilityHelper
from source.helper.TSNEHelper import TSNEHelper


def fit(params):
    fit_helper = FitHelper(params)
    fit_helper.perform_fit()


def predict(params):
    predict_helper = PredictHelper(params)
    predict_helper.perform_predict()


def eval(params):
    eval_helper = EvalHelper(params)
    eval_helper.perform_eval()


def z_shot_cls(params):
    raise NotImplementedError("Not yet implemented.")

def tsne(params):
    tsne_helper = TSNEHelper(params)
    tsne_helper.perform_tsne()


def separability(params):
    separability_helper = SeparabilityHelper(params)
    separability_helper.perform_eval()

@hydra.main(config_path="settings/", config_name="settings.yaml")
def perform_tasks(params):
    os.chdir(hydra.utils.get_original_cwd())
    OmegaConf.resolve(params)
    if "fit" in params.tasks:
        fit(params)

    if "predict" in params.tasks:
        predict(params)

    if "eval" in params.tasks:
        eval(params)

    if "z-shot-cls" in params.tasks:
        z_shot_cls(params)

    if "tsne" in params.tasks:
        tsne(params)

    if "separability" in params.tasks:
        separability(params)


if __name__ == '__main__':
    perform_tasks()