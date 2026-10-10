"""vyton.core.trainer — 训练循环(唯一实现)
用法:
    from vyton.core.trainer import quick_train
    net, log = quick_train(net, loader, epochs=12, lr=3e-4, loss_fn=..., val_fn=...)
"""
import torch


def quick_train(net, loader, epochs=12, lr=3e-4, loss_fn=None, val_fn=None,
                save_path=None, log_every=1, device="cuda"):
    """标准训练循环:任何模型/任何loss"""
    net = net.to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    log = []
    for ep in range(epochs):
        net.train(); tot = 0; nb = 0
        for batch in loader:
            args = [b.to(device) if torch.is_tensor(b) else b for b in (batch if isinstance(batch, (list, tuple)) else [batch])]
            out = net(args[0])
            out = out[0] if isinstance(out, tuple) else out
            loss = loss_fn(out, args[1]) if len(args) > 1 and loss_fn else out.float().mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        sch.step()
        entry = {"ep": ep+1, "loss": tot/max(nb,1)}
        if val_fn:
            entry.update(val_fn(net))
        log.append(entry)
        if (ep+1) % log_every == 0:
            print(f"ep{ep+1} loss={entry['loss']:.4f}", flush=True)
        if save_path:
            torch.save(net.state_dict(), save_path)
    return net, log
