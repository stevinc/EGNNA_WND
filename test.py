import logging
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def test(model, test_loader, loss_fn, device, params, metrics, excel_version="last"):
    start_time = time.time()
    # SET THE MODEL TO EVALUATION MODE
    model.eval()

    test_loss = 0

    with torch.no_grad():
        with tqdm(total=len(test_loader)) as t:
            for batch_idx, (in_bands, labels, blocks_adj_matrix, neighbours_numbers, points, dates, inf_dates, CoordAcc) in enumerate(test_loader):
                # move input data to GPU
                in_bands = in_bands.to(device)
                labels = labels.to(device)
                # accuracy_point = accuracy_point.to(device)
                blocks_adj_matrix = blocks_adj_matrix.to(device)
                # FORWARD PASS
                out = model(in_bands, blocks_adj_matrix, neighbours_numbers)

                # Cross-Entropy classification loss
                if params.weighted_loss:
                    loss = loss_fn(out, labels)
                    # loss = loss * accuracy_point
                    loss = loss.mean()
                else:
                    loss = loss_fn(out, labels)

                val_max, preds = torch.max(out, 1)

                # probability of the positive class
                val_softmax = torch.nn.Softmax(dim=1)(out).detach()

                test_loss += loss.item()

                # write loss
                t.set_postfix(loss='{:05.3f}'.format(loss.item()))
                t.update()

                # metrics
                if batch_idx == 0:
                    array_pred = preds.cpu().numpy()
                    array_labels = labels.cpu().numpy()
                    array_softmax = val_softmax.cpu().numpy()
                    lists_to_save = [points, inf_dates, dates, list(array_labels), list(array_pred),
                                     list(val_softmax[:, 0].cpu().numpy()), list(val_softmax[:, 1].cpu().numpy()), CoordAcc]
                    df_to_save = pd.DataFrame(lists_to_save).transpose()
                    df_to_save.columns = ['company_cod', 'InfectionDate', 'date_image', 'labels', 'pred', 'softmax_cls0',
                                          'softmax_cls1', 'CoordAcc']
                else:
                    array_pred = np.concatenate((array_pred, preds.cpu().numpy()), axis=0)
                    array_labels = np.concatenate((array_labels, labels.cpu().numpy()), axis=0)
                    array_softmax = np.concatenate((array_softmax, val_softmax.cpu().numpy()), axis=0)
                    lists_to_save = [points, inf_dates, dates, list(labels.cpu().numpy()), list(preds.cpu().numpy()),
                                     list(val_softmax[:, 0].cpu().numpy()), list(val_softmax[:, 1].cpu().numpy()), CoordAcc]
                    df_to_save_append = pd.DataFrame(lists_to_save).transpose()
                    df_to_save_append.columns = ['company_cod', 'InfectionDate', 'dates', 'labels', 'pred', 'softmax_cls0',
                                                 'softmax_cls1', 'CoordAcc']
                    df_to_save = df_to_save.append(df_to_save_append)

            # final metrics
            metrics_calc = {metric: metrics[metric](array_labels, array_pred) for metric in metrics}
            metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_calc.items())
            logging.info("- Test metrics micro: " + metrics_string)
            if params.name_excel:
                df_to_save.to_excel(params.name_excel + excel_version + ".xlsx")

            time_elapsed = time.time() - start_time
            logging.info('Test complete in {:.0f}m {:.0f}s. Avg test loss: {:05.3f}'.format(
                time_elapsed // 60, time_elapsed % 60, test_loss / len(test_loader)))
            return metrics_calc['f1']
