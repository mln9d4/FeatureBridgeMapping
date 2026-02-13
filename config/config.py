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
            linear_start=1e-6,
            linear_end=0.01,
            n_timestep=2000
        )
    ),
    model = dict(
        in_channel=6,
        out_channel=3,
        inner_channel=64,
        norm_groups=32,
        channel_mults=(1, 2, 4, 8),
        attn_res=[16],
        res_blocks=2,
        dropout=0.2,
        image_size=208,
        eps=1e-3
    ),
    ema_scheduler = dict(
        ema_start=1,
        ema_iter=1, # NOt used
        ema_decay=0.5
    ),
    path = dict(
        network_checkpoint='/home/mingdayang/FeatureBridgeMapping/checkpoints/PaletteDiffusion/PaletteDiffusion_model_long_training_net.pth',
        ema_network_checkpoint='/home/mingdayang/FeatureBridgeMapping/checkpoints/PaletteDiffusion/PaletteDiffusion_model_long_training_ema_net.pth'
    ),
    resume=False,
    checkpoint_path='/home/mingdayang/FeatureBridgeMapping/checkpoints/PaletteDiffusion/PaletteDiffusion_model_long_training_ema_net.pth',
    epochs=10000,
    save_interval=10,
    val_interval=1000,
    batch_size=1,
    learning_rate=1e-5,
    weight_decay=0.01,
    dataset="Nuscenes",
    architecture="PaletteDiffusion",
    start_factor=0.1,
    warmup_steps=100,
    end_factor=1.0,
)