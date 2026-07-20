python ../../src/main.py --task mc --graph Gset --Gset_id 64 --batch 100 --lr_y 0.01 --dual_init 13.5 \
    --seed 0 --max_iters 20000 --primal_lr_mode configured \
    --primal_init center_uniform --rounding bernoulli \
    --spectral_animation --spectral_animation_every 100 --spectral_animation_modes 128 \
    --lr_x 0.02 --delta 1e-10
    # --primal_lr_mode spectral --spectral_step_fraction 0.99 \

    # 下一步任务：1. 确定最终解的 t_birth
    # 2. 绘制整个动态景观的可视化
    # 3. PDBO 远好于 SDP Gap 的原因
