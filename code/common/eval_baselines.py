import os, sys, subprocess, json

TEST = len(sys.argv) > 1 and sys.argv[1] == 'test'
DEBUG = len(sys.argv) > 1 and sys.argv[1] == 'debug'
if DEBUG: TEST = True 

TASKS = {
    "glue": ["cola", "sst2", "mrpc", "qqp", "mnli", "mnli-mm", "qnli", "rte",
             "boolq", "multirc", "wsc"],
    "blimp": ["anaphor_agreement", "argument_structure", "binding", "control_raising",
              "determiner_noun_agreement", "ellipsis", "filler_gap", "irregular_forms",
              "island_effects", "npi_licensing", "quantifiers", "subject_verb_agreement"],
    "supplement": ["hypernym", "qa_congruence_easy", "qa_congruence_tricky",
                   "subject_aux_inversion", "turn_taking"],
    "msgs": ["main_verb_control", "control_raising_control", "syntactic_category_control",
             "relative_position_control", "lexical_content_the_control",
             "main_verb_lexical_content_the", "main_verb_relative_token_position",
             "control_raising_lexical_content_the", "control_raising_relative_token_position",
             "syntactic_category_lexical_content_the", "syntactic_category_relative_position"]
}

def evaluate(model_path): 
    ''' run babylm pipeline, log training to a file '''

    with open(os.path.join(model_path, 'eval.log'), 'wb') as f:

        process = subprocess.Popen(f'CUDA_VISIBLE_DEVICES=0 ./evaluate.sh {model_path}', 
                stdout=subprocess.PIPE, shell=True)

        for c in iter(lambda: process.stdout.read(1), b''):
            sys.stdout.buffer.write(c)
            f.write(c)

        process.wait() # wait until the entire eval is done


def eval_and_aggregate(model):
    ''' run evaluation, find all scores in the model dir, and aggregate them according 
        to how the BLiMP/GLUE/Super(GLUE)/MSGS papers describe. '''

    model_path = os.path.join(os.path.abspath(MODEL_DIR), model)

    # TODO: uncomment
    # evaluate(model_path)
    blimp, supplement, glue, msgs = get_scores(model_path)

    blimp_sub_avg = sum(task['eval_accuracy'] for task in blimp.values()) / len(blimp)
    supplement_avg = sum(task['eval_accuracy'] for task in supplement.values()) / len(supplement)
    blimp_avg = sum(task['eval_accuracy'] for task in [*blimp.values(), *supplement.values()]) / (len(blimp) + len(supplement))

    def glue_metric(task: str, results: dict[str, float]):
        ''' returns the appropriate glue score for aggregating results '''

        if task in ['sst2', 'mnli', 'mnli-mm', 'qnli', 'rte', 'boolq', 'wsc']:
            return results['eval_accuracy']
        elif task in ['cola']: return results['eval_mcc']
        elif task in ['mrpc', 'qqp']: 
            return (results['eval_accuracy'] + results['eval_f1']) / 2 

        elif task in ['multirc']:
            # TODO: THIS SHOULD NOT BE MCC, BUT EXACT MATCH; SEE NOTES
            return (results['eval_f1'] + results['eval_mcc']) / 2

    glue_metrics = {task: glue_metric(task, results) for task, results in glue.items()}
    glue_avg = sum(glue_metrics.values()) / len(glue_metrics)

    print(f'''
        \033[1mMULTIRC NOT IMPLEMENTED PROPERLY; SEE NOTES\033[0m
        \033[1mMSGS FINETUNED & INFERENCED, BUT NOT COMPUTED CORRECTLY AND OMMITTED\033[0m

        \033[1mFinal scores for {model} \033[0m
        BLiMP: {blimp_avg*100:.1f}%  \t ({blimp_sub_avg:.1f} base, {supplement_avg:.1f} supplement)
        GLUE : {glue_avg*100:.1f}    \t (multiplied by 100)
        ''')


def get_scores(model_path) -> tuple[dict, dict, dict[str, dict], dict[str, dict]]:
    ''' get the scores from the model's directory and create dictionaries 
        for each benchmark in the evaluation '''

    # get ZEROSHOT scores 
    blimp, supplement = {}, {} 
    for task in os.listdir(os.path.join(model_path, 'zeroshot')):
        with open(os.path.join(model_path, 'zeroshot', task, 'eval_results.json'), 'r') as f:

            score = json.load(f)
            if task in TASKS['blimp']: blimp[task] = score
            elif task in TASKS['supplement']: supplement[task] = score
            else: raise ValueError(f"Invalid task: {task}!")

    # get FINETUNED scores 
    glue, msgs = {}, {} 
    for task in os.listdir(os.path.join(model_path, 'finetune')):
        with open(os.path.join(model_path, 'finetune', task, 'all_results.json'), 'r') as f:

            score = json.load(f)
            if task in TASKS['glue']: glue[task] = score
            elif task in TASKS['msgs']: msgs[task] = score
            else: raise ValueError(f"Invalid task: {task}!")
    
    # print these babies
    for name, benchmark in [('BLiMP', blimp), ('Supp.', supplement), ('GLUE', glue), ('MSGS', msgs)]: 

        print(f'\n\033[1m{name:>50s}  \tAcc.\t F1 \t MCC\033[0m')
        for task, score in sorted(benchmark.items()):

            acc = f'{score["eval_accuracy"]*100:.2f}%' if 'eval_accuracy' in score else '-'
            f1 = f'{score["eval_f1"]:.2f}' if 'eval_f1' in score else '-'
            mcc = f'{score["eval_mcc"]:.2f}' if 'eval_mcc' in score else '-'

            print(f'{task:>50s}: \t{acc}\t {f1}\t {mcc}')

    return blimp, supplement, glue, msgs


if __name__ == '__main__':

    # check if the current conda env is named 'babylm'
    if not os.environ['CONDA_DEFAULT_ENV'] == 'babylm':
        print('\033[1m WARNING: You are not in an environment named babylm \033[0m')

    MODEL_DIR = 'models/baseline'

    for model in os.listdir(MODEL_DIR):
        eval_and_aggregate(model)
        break