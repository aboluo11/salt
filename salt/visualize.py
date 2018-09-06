from lightai.imps import *


def visualize(model, size=3):
    """
    :param size: number of img to display
    """
    tta_tsfms = [None, hflip]
    with torch.no_grad():
        model.eval()
        _, axes = plt.subplots(size, 4, figsize=(5 * size, 4 * size))
        val_ds = CsvDataset('inputs/all/3/val.csv', tsfm=to_np, tta_tsfms=tta_tsfms)
        for i, ax in zip(np.random.randint(0, len(val_ds), size=size), axes):
            tta_batch = val_ds[i]
            predicts = []
            for [img, mask], f in zip(tta_batch, tta_tsfms):
                if not f:
                    raw_img = img
                img = np.expand_dims(img, 0)
                predict = model(T(img))
                predicts.append(predict)
            p_mask = tta_mean_predict(predicts, [None, hflip])
            ax[0].imshow(raw_img.squeeze(), cmap='gray')
            ax[1].imshow(mask.squeeze(), cmap='gray')
            ax[2].imshow(p_mask.cpu().numpy().squeeze(), cmap='gray')
            ax[3].imshow(p_mask.cpu().numpy().squeeze() > 0.5, cmap='gray')
            for each, label in zip(ax, ['img', 'mask', 'predict', 'clip']):
                each.set_title(label)
                each.set_axis_off()
