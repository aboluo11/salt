def log_grad(writer, model, tag, global_step):
    if global_step:
        writer.add_scalar(f'{tag}_grad_mean', model.weight.grad.mean(), global_step)
        writer.add_scalar(f'{tag}_grad_std', model.weight.grad.std(), global_step)