config = dict(
    beta_schedule = dict(
        train=dict(
            schedule='linear',
            linear_start=1e-6,
            linear_end=0.01,
            n_timestep=2000
        ),
        test=dict(
            schedule='linear',
            linear_start=1e-4,
            linear_end=0.09,
            n_timestep=1000
        )
    ),
    model = dict(
        in_channel=512,
        out_channel=256,
        inner_channel=256,
        norm_groups=32,
        channel_mults=(1, 2, 2, 4, 8),
        attn_res=[8],
        res_blocks=3,
        dropout=0,
        with_noise_level_emb=True,
        image_size=208
    ),
    epochs=1000,
    save_interval=20,
    val_interval=1,
    batch_size=1,
    learning_rate=1e-3,
    weight_decay=0.00,
    dataset="Nuscenes",
    architecture="PaletteDiffusion",
    T_0=200,
    T_mult=2,
    eta_min=1e-6,
    start_factor=0.1,
    warmup_steps=10,
    end_factor=1.0,
    linear_lr_total_iters=400,
    milestones=50,
)