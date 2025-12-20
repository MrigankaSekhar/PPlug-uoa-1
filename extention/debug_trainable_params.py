
import torch

def print_trainable_params(model, max_rows=None):
    """
    Prints all parameters that have requires_grad=True, and total counts.
    Useful to debug zero grad_norm issues.
    """
    total_params = 0
    trainable_params = 0
    rows_printed = 0

    for name, p in model.named_parameters():
        numel = p.numel()
        total_params += numel
        if p.requires_grad:
            trainable_params += numel
            if max_rows is None or rows_printed < max_rows:
                print(f"TRAINABLE: {name} | shape={tuple(p.shape)} | numel={numel}")
                rows_printed += 1

    print("\n-------- SUMMARY --------")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    print("-------------------------\n")


# Example usage (in your main_profile or wherever you build the model):
if __name__ == "__main__":
    from extention.ModelForPer_slim_GNN import PersonalLLM_Slim
    from transformers import T5ForConditionalGeneration, AutoModel

    llm = T5ForConditionalGeneration.from_pretrained("../FlanT5-small")
    emb = AutoModel.from_pretrained("../bge-base-en-v1.5")

    model = PersonalLLM_Slim(llm, emb, max_input_len=256, max_new_len=10, task_id=3)

    print_trainable_params(model, max_rows=30)  # show first 30 trainable params
