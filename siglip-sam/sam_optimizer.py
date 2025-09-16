"""
SAM (Sharpness-Aware Minimization) Optimizer 구현
참조: https://github.com/davda54/sam
"""

import torch
from typing import Any, Dict

class SAM(torch.optim.Optimizer):
    """
    SAM: Sharpness-Aware Minimization
    
    Args:
        params: 최적화할 파라미터들
        base_optimizer: 기본 옵티마이저 클래스 (torch.optim.SGD, torch.optim.AdamW 등)
        rho: SAM 반지름 파라미터 (기본값: 0.05)
        adaptive: Adaptive SAM 사용 여부 (기본값: False)
        **kwargs: 기본 옵티마이저에 전달할 추가 인자들
    """
    
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
    
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """
        첫 번째 스텝: 손실이 최대인 지점으로 이동
        """
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
        
        if zero_grad:
            self.zero_grad()
    
    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """
        두 번째 스텝: 원래 위치로 돌아가서 실제 업데이트 수행
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"
        
        self.base_optimizer.step()  # do the actual "sharpness-aware" update
        
        if zero_grad:
            self.zero_grad()
    
    @torch.no_grad()
    def step(self, closure=None):
        """
        SAM 스텝 (closure 방식)
        """
        assert closure is not None, "SAM requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)
        
        self.first_step(zero_grad=True)
        closure()
        self.second_step()
    
    def _grad_norm(self):
        """
        그래디언트 노름 계산
        """
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(dtype=torch.float32)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            dtype=torch.float32
        )
        return norm.to(shared_device)
    
    def load_state_dict(self, state_dict):
        """
        상태 딕셔너리 로드
        """
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
