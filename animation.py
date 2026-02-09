import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import os


def create_animation(image_sequence, batch, epoch, val_i, fps=1):
    fig, ax = plt.subplots()
    writer = animation.FFMpegWriter(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
    ax.set_title(f'Batch: {batch}, Epoch: {epoch}, Validation Index: {val_i}')
    ims = []
    for img in image_sequence:
        img = img[0,:,:].cpu().numpy()  # Assuming img is a tensor of shape (1, 1, H, W)
        im = ax.imshow(img, animated=True)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=1000/fps, blit=True, repeat_delay=1000)
    if os.path.exists('/home/mingdayang/palette_diffusion/figures/animations') == False:
        os.makedirs('/home/mingdayang/palette_diffusion/figures/animations')
    ani.save(f'/home/mingdayang/palette_diffusion/figures/animations/animation_batch{batch}_epoch{epoch}_val{val_i}.mp4', writer=writer)