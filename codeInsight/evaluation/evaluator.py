import math

def compute_metrics(eval_preds):
    eval_loss = eval_preds.loss
    perplexity = math.exp(eval_loss) if eval_loss < 20 else float("inf")
    return {"perplexity": perplexity}