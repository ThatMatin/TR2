import torch
import torch.nn as nn
import time
import training
from modules import Parameters
from config import get
from typing import Optional
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.profiler import ProfilerActivity, profiler, record_function
from torch.utils import data
from tqdm.auto import tqdm
from torch.cuda.amp import autocast, GradScaler
from logger import log_memory, log_profiler, setup_logger
from interrupt import InterruptHandler
from modules.transformer import TransNovo
from training import mean_batch_acc
from models import Model, beam_search_decode, greedy_decode
logger = setup_logger(__name__)

def train_step(model: nn.Module,
               optimizer: torch.optim.Optimizer,
               loss_fn: nn.Module,
               train_dl: data.DataLoader,
               scheduler: Optional[LambdaLR],
               scaler: GradScaler,
               interruptHandler: InterruptHandler):

    result_matrix = torch.zeros((len(train_dl), 3))
    is_profiling_on = bool(get("profile.is_active"))
    clip_value = float(get("train.clip_value"))
    accumulation_steps = int(get("train.accumulation_steps"))

    # target_mask = generate_causal_target_mask(2048)
    # batch_size = get( 'model.batch_size' )
    # target_mask = torch.stack( [ batch_size * target_mask] ).to( 'cuda' )
    model.train()
    for i, (X,Y,Ch,P) in tqdm(enumerate(train_dl),
                         total=len(train_dl),
                         desc="over training set"):
        with autocast():
            logits = model(X, Y[..., 1:], Ch, P )

            optimizer.zero_grad(True)

            # tgt_output = Y[ :, 1:]
            logits_flat = logits.transpose(-2, -1) #[..., :1]
            tgt_output = Y
            loss = loss_fn( logits_flat, tgt_output)
            loss = loss / accumulation_steps

        if is_profiling_on:
            with profiler.profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
                                  record_shapes=True, profile_memory=True) as prof:
                with record_function("model_infernece"):
                    scaler.scale(loss).backward()
            prof.export_chrome_trace("trace.json")
            log_profiler(prof.key_averages().table(sort_by="cuda_time_total"))
        else:
            scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0:
            nn.utils.clip_grad_value_(model.parameters(), clip_value=clip_value)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            if scheduler is not None:
                scheduler.step()


        result_matrix[i, 0] = loss.detach() * accumulation_steps
        
        result_matrix[i, 1] = mean_batch_acc(logits_flat.permute(0,2,1).detach(), tgt_output.detach())

        if interruptHandler.is_interrupted():
            break


    return result_matrix


def test_step(model: nn.Module,
              loss_fn: nn.Module,
              test_dl: data.DataLoader,
              interruptHandler: InterruptHandler):

    with torch.inference_mode():
        result_matrix = torch.zeros((len(test_dl), 2))
        model.eval()

        for i, (X,Y, Ch, P) in tqdm(enumerate(test_dl),
                             total=len(test_dl),
                             desc="over test set"):
            with autocast():
                logits = model(X, Y[..., 1:], Ch, P )

                tgt_output = Y
                logits_flat = logits.transpose(-2, -1)
                loss = loss_fn(logits_flat, tgt_output)

            result_matrix[i, 0] = loss
            result_matrix[i, 1] = mean_batch_acc(logits, tgt_output)

            if interruptHandler.is_interrupted():
                break

    return result_matrix

def update_lr(optimizer: torch.optim.Optimizer, lr: float):
    for p in optimizer.param_groups:
        p['lr'] = lr

def init_adam(model: nn.Module, p: Parameters):
    betas = p.optimizer_adam_betas
    eps = p.optimizer_adam_eps
    lr = p.new_learning_rate
    steps = int(p.data_point_count/p.batch_size)
    total_steps = (p.n_epochs + p.n_epochs_sofar) * steps
    warmup_steps = p.warmup_steps

    print(f"Optimizer init:\n\ttotal steps: {total_steps}\n\twarmup steps: {warmup_steps}")
    def polynomial_decay(current_step: int):
        return (1 - current_step / float(total_steps)) ** 3

    a = Adam(model.parameters(), lr, betas, eps, 1e-5)
    scheduler = LambdaLR(a, polynomial_decay)

    if len(p.optimizer_state_dict) != 0:
        a.load_state_dict(model.hyper_params.optimizer_state_dict)

    return a, scheduler


def train_loop(model: Model, p: Parameters ,optimizer: torch.optim.Optimizer,
                loss_fn, train_dl, test_dl, interruptHandler: InterruptHandler, scheduler: Optional[LambdaLR] = None):
    model.train( True )
    rm_idx = 0
    lr = optimizer.param_groups[0]['lr']
    scaler = GradScaler()

    start_epoch = p.n_epochs_sofar
    end_epoch = p.n_epochs
    ranger = tqdm(range(start_epoch, end_epoch), initial=start_epoch, total=end_epoch, desc="Epochs")

    s_time = time.time()
    # TODO: Find a proper place for it
    view_rate = 1 # p.n_epochs // 10

    # loss, acc, grad norms mean
    train_result_matrix = torch.zeros((p.n_epochs, len(train_dl), 3))
    # loss, acc
    test_result_matrix = torch.zeros((p.n_epochs, len(test_dl), 2))

    for epoch in ranger:

        rm_idx = epoch - start_epoch

        train_result_matrix[rm_idx] = training.train_step(model, optimizer, loss_fn, train_dl, scheduler, scaler, interruptHandler)
        test_result_matrix[rm_idx] = training.test_step(model, loss_fn, test_dl, interruptHandler)
        if epoch % 5 == 0:
            eval(model, loss_fn, test_dl, interruptHandler)
        # Update learning rate
        lr = optimizer.param_groups[0]['lr']
        p.last_learning_rate = lr

        # calculate batch loss
        train_epoch_loss_tensor = train_result_matrix[rm_idx, :, 0]
        tr_l = train_epoch_loss_tensor[train_epoch_loss_tensor != 0].mean().item()

        train_epoch_acc_tensor = train_result_matrix[rm_idx, :, 1]
        tr_a = train_epoch_acc_tensor[train_epoch_acc_tensor != 0].mean().item()

        tr_epoch_grads = train_result_matrix[rm_idx, :, 2].mean()

        test_epoch_loss_tensor = test_result_matrix[rm_idx, :, 0]
        te_l = test_epoch_loss_tensor[test_epoch_loss_tensor != 0].mean().item()

        test_epoch_acc_tensor = test_result_matrix[rm_idx, :, 1]
        te_a = test_epoch_acc_tensor[test_epoch_acc_tensor != 0].mean().item()

        if epoch % view_rate == 0:
            t = "\n\tEpoch |    lr    | train loss | train acc | train norms mean | test loss | test acc"
            t += f"\n\t{epoch:>6}|{lr:>8.8f}|{tr_l:>12.6f}|{tr_a:>11.6f}|{tr_epoch_grads:>18.6f}|{te_l:>11.6f}|{te_a:>10.6f}\n"
            print(t)

        model.save_model()
        # TODO: Save after evey N iterations
        if interruptHandler.is_interrupted():
            break

    print(f"training time: {time.time() - s_time: 0.1f}s")
    model.finish_training()#  (rm_idx,train_result_matrix,test_result_matrix,optimizer)

def eval(model: nn.Module,
            loss_fn: nn.Module,
            test_dl: data.DataLoader,
            interruptHandler: InterruptHandler):

    with torch.inference_mode():
        # result_matrix = torch.zeros((len(test_dl), 2))
        model.eval()

        for i, (X,Y, Ch, P) in tqdm(enumerate(test_dl),
                             total=len(test_dl),
                             desc="over test set"):
            
            pred = torch.zeros_like( Y )
            with autocast():
                # logits = model(X, Y[..., 1:], Ch, P )

                # tgt_output = Y
                # logits_flat = logits.transpose(-2, -1)
                # loss = loss_fn(logits_flat, tgt_output)
                for i in range( Y.shape[0] ):
                    pred_seq = beam_search_decode( model, X[i].unsqueeze(0), Ch[i].reshape((1,1)), P[i].reshape((1,1)) )
                    pred[i,:pred_seq.shape[1]] = pred_seq
            if interruptHandler.is_interrupted():
                break
            acc = (pred == Y).to(int)
            acc = (acc.sum(dim=1) / Y.shape[1]).mean()
        print( f'accuracy ------- > {acc}' )


    return 