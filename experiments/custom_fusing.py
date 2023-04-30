import argparse

from transformers import AutoTokenizer

from fuse.main_model_fuse import ModelLoadingInfo, load_model, fuse_models
parser = argparse.ArgumentParser("hate speech classification using federated learning")
parser.add_argument('--fusing_models', required=True, type=str, help='folder_names of fusing models')
parser.add_argument('--base_model', required=True, type=str, help='base_model')

args = parser.parse_args()

print('model fusing started')
model_info = ModelLoadingInfo(name=args.base_model, tokenizer_name=args.base_model,
                              classification=True)
model_paths = args.fusing_models.split(',')
models_to_fuse = [ModelLoadingInfo(name=f'{model}/', tokenizer_name=f'{model}/', classification=True) for model in model_paths]
base_model = load_model(model_info)
fused_model = fuse_models(base_model, models_to_fuse)
# saving fused model for predictions
fused_model.save_pretrained('_'.join(model_paths))
tokenizer = AutoTokenizer.from_pretrained(f'{model_paths[0]}/')  # get the 1st model path to get the tokenizer
tokenizer.save_pretrained('_'.join(model_paths))
print('fused model saved')
