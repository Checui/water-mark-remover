import matplotlib.pyplot as plt


def sample_images(epoch, batch, gen_imgs, imgs):
    plt.rcParams["figure.figsize"] = [15, 5]
    fig, axs = plt.subplots(2, 5)
    fig.suptitle("Epoch: " + str(epoch) + ", Batch: " + str(batch), fontsize=16)
    for i in range(5):
        axs[0, i].imshow(gen_imgs[i][:, :, ::-1])
        axs[0, i].axis("off")

    for i in range(5):
        axs[1, i].imshow(imgs[i][:, :, ::-1])
        axs[1, i].axis("off")
    plt.show()
    plt.close()


def plot_losses(history):
    plt.rcParams["figure.figsize"] = [20, 5]
    # Create a single subplot
    fig, ax1 = plt.subplots()

    ax1.set_title("Losses")
    ax1.set_xlabel("epoch")
    ax1.legend(loc="upper right")
    ax1.grid()

    ax1.plot(history["g_loss"], label="G loss")
    ax1.legend()

    plt.show()
