python ../../src/main.py --task mc --graph Gset --Gset_id 63 --batch 100 --lr_y 0.01 \
    --dual_init_mode spectral --dual_burn_in 0 \
    --seed 0 --max_iters 20000 --primal_lr_mode spectral \
    --spectral_step_fraction 0.99 \
    --primal_init center_uniform \
    --spectral_animation --spectral_animation_every 1000 --spectral_animation_modes 128 \
    --delta 0.01
