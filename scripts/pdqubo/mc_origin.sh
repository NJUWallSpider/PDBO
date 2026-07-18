python ../../src/main.py --task mc --graph Gset --Gset_id 67 --batch 10 --lr_y 0.01 --dual_init 3.5 \
    --seed 0 --max_iters 20000 --primal_lr_mode configured --primal_init center_uniform \
    --spectral_animation --spectral_animation_every 10 --spectral_animation_modes 128 \
    --lr_x 0.135 --delta 1e-10
    # --primal_lr_mode spectral --spectral_step_fraction 0.99 \
