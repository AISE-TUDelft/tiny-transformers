import os, sys, subprocess, json, torch, wandb, pandas as pd 
from tqdm.contrib.concurrent import process_map

TEST = len(sys.argv) > 1 and sys.argv[1] == 'test'
DEBUG = len(sys.argv) > 1 and sys.argv[1] == 'debug'
if DEBUG: TEST = True 

MODEL_DIR       = 'models/baseline'
NO_INFERENCE    = False 
N_CUDA_DEVICES  = torch.cuda.device_count()

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

def evaluate(model_path, cuda_index=0): 
    ''' run babylm pipeline, log training to a file '''

    # open file in append mode 
    with open(os.path.join(model_path, 'eval.log'), 'ab') as f:

        cuda_index = cuda_index % N_CUDA_DEVICES
        process = subprocess.Popen(f'CUDA_VISIBLE_DEVICES={cuda_index} ./evaluate.sh {model_path}', 
                stdout=subprocess.PIPE, shell=True)

        for c in iter(lambda: process.stdout.read(1), b''):
            sys.stdout.buffer.write(c)
            f.write(c)

        process.wait() # wait until the entire eval is done


def eval_and_aggregate(kwargs) -> dict:
    ''' run evaluation, find all scores in the model dir, and aggregate them according 
        to how the BLiMP/GLUE/Super(GLUE)/MSGS papers describe. 
        Then, return all (relevant) scores as a big dic'''

    model, index, no_train = kwargs['model'], kwargs['index'], kwargs['no_train']
    model_path = os.path.join(os.path.abspath(MODEL_DIR), model)

    print(f'\033[1mEvaluating {model:40} on GPU {index % N_CUDA_DEVICES} \033[0m \t{"(no finetuning)" if no_train else ""}')

    # if the model_path contains finetune and zerosho, set no_train to true 
    if all(os.path.exists(os.path.join(model_path, x)) for x in ['finetune', 'zeroshot']):
        print(f'\t{model} has already been trained; skipping training')
        no_train = True

    # Skip inference if the user indicates so
    if not no_train: 
        # Check if the model actually exists in the directory
        if not os.path.exists(os.path.join(model_path, 'config.json')):
            print(f'\tconfig.json not found in {model_path}; skipping evaluation')
            return {'model': model}

        evaluate(model_path, cuda_index=index)

    # Early exit if either the zeroshot or finetune scores are missing
    if not os.path.exists(os.path.join(model_path, 'zeroshot')) \
        or len(os.listdir(os.path.join(model_path, 'zeroshot'))) != len(TASKS['blimp']) + len(TASKS['supplement']):
        print(f'\t{model} did not evaluate on all BLiMP tasks; skipping aggregation')
        return {'model': model}

    if not os.path.exists(os.path.join(model_path, 'finetune')) \
        or len(os.listdir(os.path.join(model_path, 'finetune'))) != len(TASKS['glue']) + len(TASKS['msgs']):
        print(f'\t{model} did not evaluate on all GLUE tasks; skipping aggregation')
        return {'model': model}

    # Get scores for every benchmark's subtasks
    blimp, supplement, glue, msgs = get_scores(model_path)

    # Aggregate the scores according to the BLiMP/GLUE papers
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

    return {
        'model': model, 
        # Aggregated (mostly averaged) score
        'blimp_avg': blimp_avg,
        'glue_avg': glue_avg,

        # Score per component; not sure if it's normal to report these separate or whether it
        # was just a thing for the BabyLM challenge.
        'base_avg': blimp_sub_avg,
        'supp_avg': supplement_avg,

        # Individual task scores, combined using the correct metrics. 
        # NOTE: except for MultiRC, for which we need to compute EM (simple in practice, im out of time tho)
        **blimp, 
        **supplement,
        **glue_metrics,
    }


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

def add_to_wandb(result):
    ''' Log results to wandb. For this you need to map the run name to its id 
        in the table, by downloading name, id columns from wandb. '''
    
    # get the run ID from the model name
    run_ids = pd.read_csv('run-ids.csv')
    model = result['model']
    run_id = run_ids[run_ids['Name'] == model]['ID'].values[0]

    # resume the wandb run and log the result
    wandb.init(
        entity='tiny-transformers', project='baselines', id=run_id, resume='must'
    )
    wandb.log(result)
    wandb.finish()


if __name__ == '__main__':

    # Evaluate on multiple GPUs, but without sharding models across GPUs.

    # check if the current conda env is named 'babylm'
    if not os.environ['CONDA_DEFAULT_ENV'] == 'babylm':
        print('\033[1m WARNING: You are not in an environment named babylm \033[0m')

    # Create kwargs
    models = [
        {'index': 0, 'model': model, 'no_train': NO_INFERENCE}
        for i, model in enumerate(sorted(os.listdir(MODEL_DIR)))
    ]

    # do the actual multiprocessing
    results = [eval_and_aggregate(model) for model in models]
    # results = process_map(eval_and_aggregate, models, max_workers=4)

    is_missing = lambda result: 'blimp_avg' not in result
    missing = [r for r in results if is_missing(r)]
    results = [r for r in results if not is_missing(r)]

    # print a nice summary
    for result in results:
        model = result['model']
        blimp_avg, glue_avg = result['blimp_avg'], result['glue_avg']
        blimp_sub_avg, supplement_avg = result['base_avg'], result['supp_avg']

        print(f'''
            \033[1mFinal scores for {model} \033[0m
            BLiMP: {blimp_avg*100:.1f}%  \t ({blimp_sub_avg:.1f} base, {supplement_avg:.1f} supplement)
            GLUE : {glue_avg*100:.1f}    \t (multiplied by 100)
        ''')

    print(f'\033[1m{len(missing)} models in {MODEL_DIR} did not complete evaluation\033[0m')
    for miss in missing: 
        print(miss['model'])

    for result in results: add_to_wandb(result)
