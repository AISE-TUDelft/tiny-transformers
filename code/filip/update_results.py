import wandb
# take comand line arguments first one is path second one is blimp third one is glue, everything string
import sys
id = sys.argv[1]
blimp = sys.argv[2]
glue = sys.argv[3]

api = wandb.Api()

run = api.run(f'tiny-transformers/filip/{id}')
run.config['blimp'] = blimp
run.config['glue'] = glue
run.update()