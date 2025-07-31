from abc import ABC

import torch
import torch.nn.functional as F

# log = get_pylogger(__name__)


class BASECFM(torch.nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        # self.n_feats = n_feats
        # self.n_spks = n_spks
        # self.spk_emb_dim = spk_emb_dim
        # self.solver = cfm_params.solver
        # if hasattr(cfm_params, "sigma_min"):
        #     self.sigma_min = cfm_params.sigma_min
        # else:
        self.sigma_min = 1e-4  # 0.0#1e-4

    @torch.inference_mode()
    def forward(
        self, model, code, output_shape, ref_mels, n_timesteps=20, temperature=1.0
    ):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """
        z = torch.randn(output_shape, device=code.device) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=code.device)
        return self.solve_euler(model, z, t_span=t_span, code=code, ref_mels=ref_mels)

    def solve_euler(self, model, x, t_span, code, ref_mels):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes
        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]

        # I am storing this because I can later plot it by putting a debugger here and saving it to a file
        # Or in future might add like a return_all_steps flag
        sol = []

        for step in range(1, len(t_span)):
            dphi_dt = model(x, t.unsqueeze(0), code_emb=code, ref_clips=ref_mels)

            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        return sol[-1]

    def compute_loss(self, model, x1, mask, code, ref_mels):
        """Computes diffusion loss

        Args:
            x1 (torch.Tensor): Target
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): target mask
                shape: (batch_size, 1, mel_timesteps)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            spks (torch.Tensor, optional): speaker embedding. Defaults to None.
                shape: (batch_size, spk_emb_dim)

        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, n_feats, mel_timesteps)
        """
        b, _, t = x1.shape

        # random timestep
        t = torch.rand([b, 1, 1], device=x1.device, dtype=x1.dtype)
        # sample noise p(x_0)
        z = torch.randn_like(x1)

        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z

        # wrong weightage
        # loss = F.mse_loss(model(y,t.squeeze(),code_emb=code,ref_clips=ref_mels), u, reduction='none').mean(dim=-2) # B,80,t -> B,t
        # loss *= mask # B,t
        # loss = loss.sum(-1) / mask.sum(-1) # B,t -> B
        # loss = loss.sum()/loss.shape[0] # B -> 1

        loss = torch.sum(
            F.mse_loss(
                model(y, t.squeeze(), code_emb=code, ref_clips=ref_mels),
                u,
                reduction="none",
            )
            * mask.unsqueeze(1)
        ) / (mask.sum() * u.shape[1])

        return loss, y, t
