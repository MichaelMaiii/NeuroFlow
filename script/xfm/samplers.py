import torch
import numpy as np
from torchdiffeq import odeint_adjoint as odeint
from functools import partial
import torch.nn as nn
import torch.nn.functional as F

def to_flattened_numpy(x):
  """Flatten a torch tensor `x` and convert it to numpy."""
  return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
  """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
  return torch.from_numpy(x.reshape(shape))

def euler_sampler_allstep(
        model,
        latents,
        t_steps,
        num_steps=20,
        heun=False,
        ):
    # setup conditioning
    _dtype = latents.dtype    
    t_steps = t_steps
    x_next = latents.to(torch.float64)
    device = x_next.device
    
    with torch.no_grad():
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next
            model_input = x_cur
            time_input = torch.ones(model_input.size(0)).to(device=device, dtype=torch.float64) * t_cur
            
            d_cur = model(
                model_input.to(dtype=_dtype), time_input.to(dtype=_dtype)
                )[0].to(torch.float64)             
            x_next = x_cur + (t_next - t_cur) * d_cur
            
            # Using heun
            if heun and (i < num_steps - 1):
                model_input = x_next
                time_input = torch.ones(model_input.size(0)).to(
                    device=model_input.device, dtype=torch.float64
                    ) * t_next
                d_prime = model(
                    model_input.to(dtype=_dtype), time_input.to(dtype=_dtype)
                    )[0].to(torch.float64)
                x_next = x_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)
                
            if i == 0:
                x_step = x_next
            else:
                x_step = torch.vstack((x_step, x_next))
                
    return x_step

def euler_sampler(
        model,
        latents,
        t_steps,
        num_steps=20,
        heun=False,
        ):
    # setup conditioning
    _dtype = latents.dtype    
    t_steps = t_steps
    x_next = latents.to(torch.float64)
    device = x_next.device
    
    with torch.no_grad():
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next
            model_input = x_cur
            time_input = torch.ones(model_input.size(0)).to(device=device, dtype=torch.float64) * t_cur
            
            d_cur = model(
                model_input.to(dtype=_dtype), time_input.to(dtype=_dtype)
                ).to(torch.float64)   #! not [0] except REPA            
            x_next = x_cur + (t_next - t_cur) * d_cur
            
            # Using heun
            if heun and (i < num_steps - 1):
                model_input = x_next
                time_input = torch.ones(model_input.size(0)).to(
                    device=model_input.device, dtype=torch.float64
                    ) * t_next
                d_prime = model(
                    model_input.to(dtype=_dtype), time_input.to(dtype=_dtype)
                    ).to(torch.float64)
                x_next = x_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)
                
    return x_next

def ode_sampler(
    model,
    latents,
    t_steps,
    solver="euler",   # "euler" | "heun" | "rk4"
):
    _dtype = latents.dtype
    device = latents.device

    x = latents.to(torch.float64)

    def f(x, t):
        return model(
            x.to(dtype=_dtype),
            t.to(dtype=_dtype)
        ).to(torch.float64)

    with torch.no_grad():
        for t_cur, t_next in zip(t_steps[:-1], t_steps[1:]):
            dt = t_next - t_cur
            t_cur_vec = torch.full((x.size(0),), t_cur, device=device, dtype=torch.float64)

            # ---------- Euler ----------
            if solver == "euler":
                k1 = f(x, t_cur_vec)
                x = x + dt * k1

            # ---------- Heun (RK2) ----------
            elif solver == "heun":
                k1 = f(x, t_cur_vec)
                x_pred = x + dt * k1
                t_next_vec = torch.full_like(t_cur_vec, t_next)
                k2 = f(x_pred, t_next_vec)
                x = x + dt * 0.5 * (k1 + k2)

            # ---------- RK4 ----------
            elif solver == "rk4":
                k1 = f(x, t_cur_vec)
                k2 = f(x + 0.5 * dt * k1, t_cur_vec + 0.5 * dt)
                k3 = f(x + 0.5 * dt * k2, t_cur_vec + 0.5 * dt)
                k4 = f(x + dt * k3, t_cur_vec + dt)
                x = x + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0

            else:
                raise ValueError(f"Unknown solver: {solver}")

    return x

def ode_sampler_bwd_reverse(model, fmri, num_steps, solver="euler"):
    
    z1 = fmri
    bw_steps = torch.linspace(1, 0, num_steps+1, dtype=torch.float64)
    z0_pred = ode_sampler(model, z1, bw_steps, solver=solver)
    
    return z0_pred.float()

def ode_sampler_fwd_reverse(model, image, num_steps, solver="euler"):
    
    z0 = image
    fw_steps = torch.linspace(0, 1, num_steps+1, dtype=torch.float64)
    z1_pred = ode_sampler(model, z0, fw_steps, solver=solver)
    
    return z1_pred.float()

def ode_sampler_cycle_reverse(model, fmri, image, num_steps, solver="euler"):

    z0 = image
    fw_steps = torch.linspace(0, 1, num_steps+1, dtype=torch.float64)
    z1_pred = ode_sampler(model, z0, fw_steps, solver=solver)
    
    z1 = fmri
    bw_steps = torch.linspace(1, 0, num_steps+1, dtype=torch.float64)
    z0_pred = ode_sampler(model, z1, bw_steps, solver=solver)
    
    return z0_pred.float(), z1_pred.float()

def euler_sampler_cycle(model, fmri, image, num_steps, heun=False):

    z0 = fmri
    fw_steps = torch.linspace(0, 1, num_steps+1, dtype=torch.float64)
    z1_pred = euler_sampler(model, z0, fw_steps, num_steps, heun)
    
    z1 = image
    bw_steps = torch.linspace(1, 0, num_steps+1, dtype=torch.float64)
    z0_pred = euler_sampler(model, z1, bw_steps, num_steps, heun)
    
    return z1_pred.float(), z0_pred.float()

def euler_sampler_cycle_reverse(model, fmri, image, num_steps, heun=False):

    z0 = image
    fw_steps = torch.linspace(0, 1, num_steps+1, dtype=torch.float64)
    z1_pred = euler_sampler(model, z0, fw_steps, num_steps, heun)
    
    z1 = fmri
    bw_steps = torch.linspace(1, 0, num_steps+1, dtype=torch.float64)
    z0_pred = euler_sampler(model, z1, bw_steps, num_steps, heun)
    
    return z0_pred.float(), z1_pred.float()

def euler_sampler_cycle_reverse_allstep(model, fmri, image, num_steps, heun=False):

    z0 = image
    fw_steps = torch.linspace(0, 1, num_steps+1, dtype=torch.float64)
    z1_pred = euler_sampler_allstep(model, z0, fw_steps, num_steps, heun)
    
    z1 = fmri
    bw_steps = torch.linspace(1, 0, num_steps+1, dtype=torch.float64)
    z0_pred = euler_sampler_allstep(model, z1, bw_steps, num_steps, heun)
    
    return z0_pred.float(), z1_pred.float()

def euler_sampler_bwd_allstep(model, image, num_steps, heun=False, sampling=False):
    
    z1 = image
    bw_steps = torch.linspace(1, 0, num_steps+1, dtype=torch.float64)
    z0_pred = euler_sampler_allstep(model, z1, bw_steps, num_steps, heun)
    
    return z0_pred.float()


def euler_sampler_bwd(model, image, num_steps, heun=False):
    
    z1 = image
    bw_steps = torch.linspace(1, 0, num_steps+1, dtype=torch.float64)
    z0_pred = euler_sampler(model, z1, bw_steps, num_steps, heun)
    
    return z0_pred.float()

def rfm_sampler_fwd(loss_fn, ema, fmri, num_steps):
    
    norm_ = torch.norm(fmri.flatten(1), dim=1)
    z0 = fmri
    
    fw_steps = torch.linspace(0, 1, num_steps+1, dtype=torch.float32)
    z1_pred = loss_fn.sampler(ema, z0, step_size=1/num_steps, T=fw_steps)
    
    return z1_pred[-1].view(-1, 256, 1664).float()

def rfm_sampler_bwd(loss_fn, ema, image, num_steps):
    norm_ = torch.norm(image.flatten(1), dim=1)
    
    z1 = image
    bw_steps = torch.linspace(1, 0, num_steps+1, dtype=torch.float32)
    z0_pred = loss_fn.sampler(ema, z1, step_size=1/num_steps, T=bw_steps)
    
    return z0_pred[-1].view(-1, 256, 1664).float()


def euler_sampler_bwd(model, image, num_steps, heun=False, sampling=False):
    
    z1 = image
    bw_steps = torch.linspace(1, 0, num_steps+1, dtype=torch.float64)
    z0_pred = euler_sampler(model, z1, bw_steps, num_steps, heun)
    
    return z0_pred.float()

def euler_sampler_bwd_reverse(model, fmri, num_steps, heun=False, sampling=False):
    
    z1 = fmri
    bw_steps = torch.linspace(1, 0, num_steps+1, dtype=torch.float64)
    z0_pred = euler_sampler(model, z1, bw_steps, num_steps, heun)
    
    return z0_pred.float()

def euler_sampler_fwd_reverse(model, image, num_steps, heun=False):
    
    z0 = image
    fw_steps = torch.linspace(0, 1, num_steps+1, dtype=torch.float64)
    z1_pred = euler_sampler(model, z0, fw_steps, num_steps, heun)
    
    return z1_pred.float()



def euler_sampler_fwd(model, fmri, num_steps, heun=False):
    
    z0 = fmri
    fw_steps = torch.linspace(0, 1, num_steps+1, dtype=torch.float64)
    z1_pred = euler_sampler(model, z0, fw_steps, num_steps, heun)
    
    return z1_pred.float()

def euler_sampler_fwd_allstep(model, fmri, num_steps, heun=False):
    
    z0 = fmri
    fw_steps = torch.linspace(0, 1, num_steps+1, dtype=torch.float64)
    z1_step = euler_sampler_allstep(model, z0, fw_steps, num_steps, heun)
    
    return z1_step.float()


def euler_maruyama_sampler(
        model,
        latents,
        y,
        num_steps=20,
        heun=False,  # not used, just for compatability
        cfg_scale=1.0,
        guidance_low=0.0,
        guidance_high=1.0,
        path_type="linear",
        ):
    # setup conditioning
    if cfg_scale > 1.0:
        y_null = torch.tensor([1000] * y.size(0), device=y.device)
            
    _dtype = latents.dtype
    
    t_steps = torch.linspace(1., 0.04, num_steps, dtype=torch.float64)
    t_steps = torch.cat([t_steps, torch.tensor([0.], dtype=torch.float64)])
    x_next = latents.to(torch.float64)
    device = x_next.device

    with torch.no_grad():
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-2], t_steps[1:-1])):
            dt = t_next - t_cur
            x_cur = x_next
            if cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low:
                model_input = torch.cat([x_cur] * 2, dim=0)
                y_cur = torch.cat([y, y_null], dim=0)
            else:
                model_input = x_cur
                y_cur = y            
            kwargs = dict(y=y_cur)
            time_input = torch.ones(model_input.size(0)).to(device=device, dtype=torch.float64) * t_cur
            diffusion = compute_diffusion(t_cur)            
            eps_i = torch.randn_like(x_cur).to(device)
            deps = eps_i * torch.sqrt(torch.abs(dt))

            # compute drift
            v_cur = model(
                model_input.to(dtype=_dtype), time_input.to(dtype=_dtype), **kwargs
                )[0].to(torch.float64)
            s_cur = get_score_from_velocity(v_cur, model_input, time_input, path_type=path_type)
            d_cur = v_cur - 0.5 * diffusion * s_cur
            if cfg_scale > 1. and t_cur <= guidance_high and t_cur >= guidance_low:
                d_cur_cond, d_cur_uncond = d_cur.chunk(2)
                d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)

            x_next =  x_cur + d_cur * dt + torch.sqrt(diffusion) * deps
    
    # last step
    t_cur, t_next = t_steps[-2], t_steps[-1]
    dt = t_next - t_cur
    x_cur = x_next
    if cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low:
        model_input = torch.cat([x_cur] * 2, dim=0)
        y_cur = torch.cat([y, y_null], dim=0)
    else:
        model_input = x_cur
        y_cur = y            
    kwargs = dict(y=y_cur)
    time_input = torch.ones(model_input.size(0)).to(
        device=device, dtype=torch.float64
        ) * t_cur
    
    # compute drift
    v_cur = model(
        model_input.to(dtype=_dtype), time_input.to(dtype=_dtype), **kwargs
        )[0].to(torch.float64)
    s_cur = get_score_from_velocity(v_cur, model_input, time_input, path_type=path_type)
    diffusion = compute_diffusion(t_cur)
    d_cur = v_cur - 0.5 * diffusion * s_cur
    if cfg_scale > 1. and t_cur <= guidance_high and t_cur >= guidance_low:
        d_cur_cond, d_cur_uncond = d_cur.chunk(2)
        d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)

    mean_x = x_cur + dt * d_cur
                    
    return mean_x
