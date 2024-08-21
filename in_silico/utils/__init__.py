# Created by David Coggan on 2023 07 10
from .model_contrasts import (model_base, model_dirs, model_contrasts,
                              model_dirs_gen)
from .plot_filters import plot_filters
from .behavioral_benchmark import (make_svc_dataset,
                                   make_pca_dataset,
                                   train_svc,
                                   get_responses,
                                   analyse_performance,
                                   plot_performance,
                                   evaluate_reconstructions)
from .behavioral_benchmark import miscellaneous_plots as behavioral_plots
from .behavioral_compare_models import compare_models as compare_models_behav
from .fMRI_benchmark import (
    get_model_responses, RSA_fMRI, generate_reconstructions)
from .fMRI_benchmark import compare_models as compare_models_fMRI
from .tile_sample_inputs import tile_sample_inputs
from .pixel_attribution import pixel_attribution, evaluate_salience
from .pixel_attribution import compare_models as compare_models_pixel
from .pixel_attribution import make_plots as plot_pixel_attribution
#from .BrainScore_benchmark import measure_scores
#from .get_BrainScore import get_score
#from .BrainScore_benchmark import compare_models as compare_models_brainscore