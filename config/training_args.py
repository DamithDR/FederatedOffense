train_args = {
    'evaluate_during_training': True,
    'logging_steps': 1000,
    'num_train_epochs': 3,
    'evaluate_during_training_steps': 100,
    'save_eval_checkpoints': False,
    # 'manual_seed': 777,
    'train_batch_size': 8,
    'eval_batch_size': 8,
    'overwrite_output_dir': True,
    'output_dir': 'outputs/',
    'fused_model_path': 'outputs/fused_model/',
    'fused_finetuned_model_path': 'outputs/fused_finetuned_model/',
}
