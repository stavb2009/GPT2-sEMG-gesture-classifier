from pydantic_settings import BaseSettings


class Env(BaseSettings):
    DATA_LOCATION: str = ''
    SAMPLING_RATIO: int = 1
    BATCH_SIZE : int = 4
    NUM_LAYERS : int = -9 # last 9 layers, for example
    lr : int = 5e-5
    NUM_EPOCHS : int = 20
    NUM_EPOCHS_FOR_VALIDATION : int = 5
    MODEL_NAME : str = 'gpt2'
    PROBLEM_TYPE: srt = 'single_label_classification'
    LEGAL_VALUES = list(range(8))
    NUM_LABELS : int = len(LEGAL_VALUES)
    PADDING_SIDE : str = 'right'
    MODEL_SAVING_DIR : str = ''
    VAL_DICT_SAVE_NAME : str = 'all_results_dict.pkl'
    RESULTS_DICT_SAVE_NAME : str = 'all_results_dict_final.pkl'
    CONFUSION_MATRIX_NAME :str = 'final_result_confusion'
    ACC_GRAPH_NAME :str = 'final_result_accuracy'
    MODEL_SAVING_NAME : str = 'fine_tuned_gpt2_model'


ENV = Env()
