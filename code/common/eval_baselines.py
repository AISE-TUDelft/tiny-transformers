import os, sys, subprocess, json, torch, wandb, pandas as pd, traceback, time
from tqdm.contrib.concurrent import process_map


MODEL_DIR       = 'models/rope'
N_CUDA_DEVICES  = torch.cuda.device_count()
ENV_NAME        = 'babylm'


TASKS = {
    "glue": ["cola", "sst2", "mrpc", "qqp", "mnli", "mnli-mm", "qnli", "rte",
             "boolq", "multirc", "wsc"],
    "blimp": ["anaphor_agreement", "argument_structure", "binding", "control_raising",
              "determiner_noun_agreement", "ellipsis", "filler_gap", "irregular_forms",
              "island_effects", "npi_licensing", "quantifiers", "subject_verb_agreement"],
    "supplement": ["hypernym", "qa_congruence_easy", "qa_congruence_tricky",
                   "subject_aux_inversion", "turn_taking"],
    # "msgs": ["main_verb_control", "control_raising_control", "syntactic_category_control",
    #          "relative_position_control", "lexical_content_the_control",
    #          "main_verb_lexical_content_the", "main_verb_relative_token_position",
    #          "control_raising_lexical_content_the", "control_raising_relative_token_position",
    #          "syntactic_category_lexical_content_the", "syntactic_category_relative_position"]
}

def run_babylm_pipeline(model_path: str, cuda_index: int = 0, verbose=True, debug=False): 
    ''' Run babylm pipeline, log training to a file `eval.log` under model_path 
        babylm creates `zeroshot` and `finetune` directories for the BLIMP and GLUE/MSGS tasks

        - model_path: directory containing the model (.safetensors and config.json)
        - cuda_index: index of the GPU to run the evaluation on
        - verbose: print the output of the evaluation script to console (rather than stderr alone)
    '''
    # check if the current conda env is named 'babylm' (corresponding to the pipeline's env)
    # if not os.environ['CONDA_DEFAULT_ENV'] == ENV_NAME:
    #     print(f'\033[1;31m WARNING: You are not in an environment named {ENV_NAME} \033[0m')

    log_file = os.path.join(model_path, 'eval.log')
    if debug: print('\t\033[31;1m EVALUATING IN DEBUG MODE (ONLY 1 EPOCH)\033[0m')
    print(f'\t\033[1m> Running BabyLM pipeline for {os.path.basename(model_path)} on GPU {cuda_index}\033[0m.\n')
    print(f'\033[90mFollow progress with: \ntail -f {log_file}')

    path_to_common = os.path.abspath(os.path.join(os.getcwd(), 'common')) \
            if not os.path.basename(os.getcwd()) == 'common' \
            else os.path.abspath(os.getcwd())
    print(f'path to common: {path_to_common}')

    command = ' '.join([
        f'CUDA_VISIBLE_DEVICES={cuda_index}',           # select the GPU
        f'conda run -n {ENV_NAME} --no-capture-output', # correct env & stream the output (rather than buffer)
        f'--cwd {path_to_common}',                      # run it in the `common` folder
        f'./evaluate.sh {model_path} {"debug" if debug else ""}',           # run the evaluation script
        f'2>&1 | tee {log_file}'                        # log the output & stderr to 'eval.log
    ])

    process = subprocess.Popen(
        command,
        stdout=sys.stdout if verbose else subprocess.PIPE, 
        # stderr=subprocess.STDOUT, 
        shell=True, 
        universal_newlines=True,
    )
    process.wait(60*60*12) # wait until the entire eval is done (or 12h, which is more than enough)
    print('\033[0m') # clear gray text escape seq

def is_evaluated(model_path, tasks=['zeroshot', 'finetune']):
    ''' Check if the model has already been evaluated; 
        i.e. `finetune` and `zeroshot` dirs contain all task subdirs 
    '''
    is_evaluated = True
    model = os.path.basename(model_path)

    if not os.path.exists(os.path.join(model_path, 'zeroshot')):
        print(f'\t{model} did not evaluate on any BLiMP tasks')
        is_evaluated = False

    else: # make sure all task directories exist, and that each task subdir contains `eval_results.json`
        zeroshot_dirs = [f for f in os.listdir(os.path.join(model_path, 'zeroshot'))]
        for task in TASKS['blimp'] + TASKS['supplement']:
            if task not in zeroshot_dirs:
                print(f'\t{model} did not evaluate on {task}')
                is_evaluated = False
            elif not os.path.exists(os.path.join(model_path, 'zeroshot', task, 'eval_results.json')):
                print(f'\t{model} did not complete evaluation on {task}')
                is_evaluated = False

    if not os.path.exists(os.path.join(model_path, 'finetune')):
        print(f'\t{model} did not evaluate on any GLUE tasks')
        is_evaluated = False

    else: # make sure all task directories exist, and that each task subdir contains `all_results.json`
        finetune_files = [f for f in os.listdir(os.path.join(model_path, 'finetune'))]
        finetune_tasks = TASKS['glue'] # + TASKS['msgs']
        for task in finetune_tasks:
            if task not in finetune_files:
                print(f'\t{model} did not evaluate on {task}')
                is_evaluated = False
            elif not os.path.exists(os.path.join(model_path, 'finetune', task, 'all_results.json')):
                print(f'\t{model} did not complete evaluation on {task}')
                is_evaluated = False

    return is_evaluated 

def get_scores(model_path, verbose=True) -> tuple[dict, dict, dict[str, dict], dict[str, dict]]:
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
        result_file_path = os.path.join(model_path, 'finetune', task, 'all_results.json')
        if not os.path.exists(result_file_path):
            print(f'\033[31;1m{os.path.join(model_path, "finetune", task)} exists, but doesn\'t contain all_results.json\033[0m')
            continue

        with open(result_file_path, 'r') as f:

            score = json.load(f)
            if task in TASKS['glue']: glue[task] = score
            # elif task in TASKS['msgs']: msgs[task] = score
            else: 
                print(f'{os.path.join("finetune", task)} is not in tasks defined at the top of eval_baselines.py')
    
    if verbose: # print these babies
        print(f'\n\033[1mScores found in {model_path}\033[0m')
        for name, benchmark in [('BLiMP', blimp), ('Supp.', supplement), ('GLUE', glue), ('MSGS', msgs)]: 

            print(f'\n\033[1m{name:>50s}  \tAcc.\t F1 \t MCC\033[0m')
            for task, score in sorted(benchmark.items()):

                acc = f'{score["eval_accuracy"]*100:.2f}%' if 'eval_accuracy' in score else '-'
                f1 = f'{score["eval_f1"]:.2f}' if 'eval_f1' in score else '-'
                mcc = f'{score["eval_mcc"]:.2f}' if 'eval_mcc' in score else '-'

                print(f'{task:>50s}: \t{acc}\t {f1}\t {mcc}')

    return blimp, supplement, glue, msgs

def aggregate_scores(blimp, supplement, glue, msgs, model) -> dict:
    ''' Aggregate the scores according to the BLiMP/GLUE papers '''

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

def add_to_wandb(result, step=None):
    ''' Log results to wandb. For this you need to map the run name to its id 
        in the table, by downloading name, id columns from wandb. 
        In hindsight, it's probably easier to save this under the model to avoid duplicates. 
    '''
    
    # get the run ID from the model name
    run_ids = pd.read_csv('run-ids.csv')
    model = result['model']
    run_id = run_ids[run_ids['Name'] == model]['ID'].values[0]

    # resume the wandb run and log the result
    wandb.init(
        entity='tiny-transformers', project='baselines', id=run_id, resume='must'
    )
    wandb.log(result, step=step)
    wandb.finish()

def eval_and_aggregate(model_path: str, index: int = 0, verbose=True, debug=False) -> dict:
    ''' Run evaluation, find all scores in the model dir, and aggregate them according 
        to how the BLiMP/GLUE/Super(GLUE)/MSGS papers describe. 
        Then, return aggregated scores as a big dic

        - model_path: path to the directory containing the model (.safetensors and config.json)
        - index: GPU index to run the evaluation on
    '''
    try: 
        model_path = os.path.abspath(model_path)
        model = os.path.basename(model_path)
        print(f'\t\033[1m{model}\033[0m')
        print(f'\t\033[1m> Checking for predictions...\033[0m')

        # if the model already has been evaluated, skip re-doing this. 
        if is_evaluated(model_path):
            print(f'\t{model} has already been trained; skipping training')

        elif not os.path.exists(os.path.join(model_path, 'config.json')) \
            or not os.path.exists(os.path.join(model_path, 'model.safetensors')):
            print(f'\t\033[1;31mModel files not found in {model_path}; skipping evaluation\033[0m')
            raise FileNotFoundError(f'Model files not found in {model_path}')

        else: 
            # t_0 = time.time()
            run_babylm_pipeline(model_path, cuda_index=index, verbose=verbose, debug=debug)
            # double check that the evaluation actually worked
            if not is_evaluated(model_path):
                raise FileNotFoundError(f'\033[1;31mEvaluation failed; skipping aggregation\033[0m (check eval.log in {model_path})')
            # time_taken = time.time() - t_0

        # Get raw scores for every benchmark's subtasks
        blimp, supplement, glue, msgs = get_scores(model_path, verbose=verbose)

        # aggregate by selecting the correct metric for each task, and compute task-averages
        result : dict = aggregate_scores(blimp, supplement, glue, msgs, model=model)
        # result.update({'eval_time': time_taken})

        return result 

    except Exception as e:
        print(f'\t\033[1;31mError in {model_path}:\033[0m\n')
        traceback.print_exc()
        return {'model': model_path, 'error': e}

def eval_multiprocess(kwargs: dict):
    return eval_and_aggregate(**kwargs)

if __name__ == '__main__':
    ''' Evaluate on multiple GPUs, but without sharding models across GPUs. '''

    # Create kwargs
    models : list[dict] = [ 
        {
            'model_path': os.path.join(os.path.abspath(MODEL_DIR), model),
            'index'     : 0, 
            'verbose'   : True,
        } for i, model in 
            # I'm mainly doing reversed sorted to start with the most recent model first
            enumerate(reversed(sorted(os.listdir(MODEL_DIR))))
    ]

    # evaluate and aggregate scores for all models 
    results = [eval_and_aggregate(**model) for model in models]
    # you can try multiprocessing, but babylm is giving me some errors 
    # you'll also need to unpack the kwargs into tuples for process_map to work 
    # results = process_map(eval_multiprocess, models, max_workers=2)

    # print a nice summary
    is_missing = lambda result: len(result) <= 2 
    missing = [r for r in results if is_missing(r)]
    results = sorted([r for r in results if not is_missing(r)], key=lambda r: r['blimp_avg'])

    print(f'\033[1mFinal Scores\n{"Model":>50s}  \tBLiMP\t GLUE\033[0m')
    for result in results:
        model = result['model']
        blimp_avg, glue_avg = result['blimp_avg'], result['glue_avg']
        blimp_sub_avg, supplement_avg = result['base_avg'], result['supp_avg']

        print(f'{model:>50s}  \t{blimp_avg*100:.1f}%\t {glue_avg*100:.1f}')

    print(f'\n\n\033[1m{len(missing)} models in {MODEL_DIR} did not complete evaluation\033[0m')
    for miss in missing: 
        print(f'{miss["model"]} \n{miss["error"]}')

    # for result in results: add_to_wandb(result)
